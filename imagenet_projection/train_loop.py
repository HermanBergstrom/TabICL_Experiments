"""Training loop for projection methods on sampled ImageNet episodes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from finetune_projection_head import FrozenTabICLConfig, build_frozen_tabicl_backbone

from .config import SamplingConfig
from .episodes import class_safe_support_query_indices, remap_labels_from_support
from .projection_methods import build_projection_module, project_episode_features
from .sampling import ShardedEmbeddingSampler, sample_one_dataset


def train_projection_head_with_sampler(
	*,
	sampling_config: SamplingConfig,
	val_sampling_config: SamplingConfig | None,
	val_heldout_classes_count: int | None,
	device: str | torch.device,
	num_steps: int,
	learning_rate: float,
	weight_decay: float,
	query_fraction_min: float,
	query_fraction_max: float,
	min_query_size: int,
	head_type: str,
	projection_dim: int | None,
	hidden_dim: int,
	dropout: float,
	grad_clip_norm: float | None,
	log_every: int,
	val_every: int,
	val_batches: int,
	tabicl_config: FrozenTabICLConfig,
	projection_method: str = "projection_head",
	zca_epsilon: float = 1e-5,
	hyper_top_k: int = 16,
	hyper_encoder_type: str = "mlp",
	hyper_attn_heads: int = 4,
	hyper_attn_layers: int = 2,
	gradient_accumulation_steps: int = 1,
) -> tuple[torch.nn.Module, dict[str, list[float]], dict[str, Any]]:
	"""Train projection module with one sampled dataset per optimization step."""
	if num_steps <= 0:
		raise ValueError("num_steps must be > 0")
	if learning_rate <= 0:
		raise ValueError("learning_rate must be > 0")
	if not (0.0 < query_fraction_min < 1.0):
		raise ValueError("query_fraction_min must be in (0, 1)")
	if not (0.0 < query_fraction_max < 1.0):
		raise ValueError("query_fraction_max must be in (0, 1)")
	if query_fraction_min > query_fraction_max:
		raise ValueError("query_fraction_min cannot exceed query_fraction_max")
	if min_query_size <= 0:
		raise ValueError("min_query_size must be > 0")
	if val_every < 0:
		raise ValueError("val_every must be >= 0")
	if val_batches <= 0:
		raise ValueError("val_batches must be > 0")
	if gradient_accumulation_steps <= 0:
		raise ValueError("gradient_accumulation_steps must be > 0")

	device_t = torch.device(device)

	heldout_classes: set[int] = set()
	if val_heldout_classes_count is not None and val_heldout_classes_count > 0:
		temp_sampler = ShardedEmbeddingSampler(
			output_dir=sampling_config.output_dir,
			split=sampling_config.split,
			seed=sampling_config.seed,
			max_cached_shards=sampling_config.max_cached_shards,
			sampler_backend=sampling_config.sampler_backend,
		)
		all_available = temp_sampler.available_classes
		n_to_holdout = min(val_heldout_classes_count, len(all_available) - 2)
		heldout_rng = np.random.default_rng(sampling_config.seed + 999)
		heldout_classes = set(heldout_rng.choice(all_available, size=n_to_holdout, replace=False).tolist())
		print(f"[info] Held-out {len(heldout_classes)} classes for validation on unseen classes")
		del temp_sampler

	sampler = ShardedEmbeddingSampler(
		output_dir=sampling_config.output_dir,
		split=sampling_config.split,
		seed=sampling_config.seed,
		max_cached_shards=sampling_config.max_cached_shards,
		sampler_backend=sampling_config.sampler_backend,
		excluded_classes=heldout_classes,
	)
	split_rng = np.random.default_rng(sampling_config.seed + 7)
	if val_sampling_config is not None and val_every > 0:
		val_sampler = ShardedEmbeddingSampler(
			output_dir=val_sampling_config.output_dir,
			split=val_sampling_config.split,
			seed=val_sampling_config.seed,
			max_cached_shards=val_sampling_config.max_cached_shards,
			sampler_backend=val_sampling_config.sampler_backend,
		)
		split_rng_val = np.random.default_rng(val_sampling_config.seed + 13)
	else:
		val_sampler = None
		split_rng_val = None

	val_heldout_sampler: ShardedEmbeddingSampler | None = None
	split_rng_heldout: np.random.Generator | None = None
	if len(heldout_classes) > 0 and val_every > 0:
		val_heldout_sampler = ShardedEmbeddingSampler(
			output_dir=sampling_config.output_dir,
			split=sampling_config.split,
			seed=sampling_config.seed + 17,
			max_cached_shards=sampling_config.max_cached_shards,
			sampler_backend=sampling_config.sampler_backend,
			excluded_classes=set(sampler.available_classes),
		)
		split_rng_heldout = np.random.default_rng(sampling_config.seed + 19)
		print(f"[info] Created held-out validation sampler with {len(val_heldout_sampler.available_classes)} held-out classes")

	val_cache: list[dict[str, Any]] = []
	if val_sampler is not None and val_sampling_config is not None and split_rng_val is not None:
		for _ in range(val_batches):
			X_np_val, y_np_val, meta_val = sample_one_dataset(
				sampler=val_sampler,
				min_dataset_size=val_sampling_config.min_dataset_size,
				max_dataset_size=val_sampling_config.max_dataset_size,
				min_classes=val_sampling_config.min_classes,
				max_classes=val_sampling_config.max_classes,
				min_per_class=val_sampling_config.min_per_class,
				remap_labels=val_sampling_config.remap_labels,
				allow_replacement=val_sampling_config.allow_replacement,
			)

			y_val_step_cpu = torch.as_tensor(y_np_val, dtype=torch.long)
			support_idx_np, query_idx_np = class_safe_support_query_indices(
				y=y_val_step_cpu,
				rng=split_rng_val,
				frac_min=query_fraction_min,
				frac_max=query_fraction_max,
				min_query_size=min_query_size,
			)
			val_cache.append(
				{
					"X": X_np_val,
					"y": y_np_val,
					"support_idx": support_idx_np,
					"query_idx": query_idx_np,
					"meta": meta_val,
				}
			)
		print(f"[info] Prepared {len(val_cache)} fixed validation episodes")

	val_heldout_cache: list[dict[str, Any]] = []
	if val_heldout_sampler is not None and split_rng_heldout is not None:
		for _ in range(val_batches):
			X_np_ho, y_np_ho, meta_ho = sample_one_dataset(
				sampler=val_heldout_sampler,
				min_dataset_size=sampling_config.min_dataset_size,
				max_dataset_size=sampling_config.max_dataset_size,
				min_classes=sampling_config.min_classes,
				max_classes=sampling_config.max_classes,
				min_per_class=sampling_config.min_per_class,
				remap_labels=sampling_config.remap_labels,
				allow_replacement=sampling_config.allow_replacement,
			)

			y_ho_step_cpu = torch.as_tensor(y_np_ho, dtype=torch.long)
			support_idx_ho_np, query_idx_ho_np = class_safe_support_query_indices(
				y=y_ho_step_cpu,
				rng=split_rng_heldout,
				frac_min=query_fraction_min,
				frac_max=query_fraction_max,
				min_query_size=min_query_size,
			)
			val_heldout_cache.append(
				{
					"X": X_np_ho,
					"y": y_np_ho,
					"support_idx": support_idx_ho_np,
					"query_idx": query_idx_ho_np,
					"meta": meta_ho,
				}
			)
		print(f"[info] Prepared {len(val_heldout_cache)} fixed held-out validation episodes")

	X0, y0, _meta0 = sample_one_dataset(
		sampler=sampler,
		min_dataset_size=sampling_config.min_dataset_size,
		max_dataset_size=sampling_config.max_dataset_size,
		min_classes=sampling_config.min_classes,
		max_classes=sampling_config.max_classes,
		min_per_class=sampling_config.min_per_class,
		remap_labels=sampling_config.remap_labels,
		allow_replacement=sampling_config.allow_replacement,
	)
	input_dim = int(X0.shape[1])

	backbone = build_frozen_tabicl_backbone(
		input_dim=int(projection_dim or input_dim),
		config=tabicl_config,
		device=device_t,
	)
	backbone = backbone.to(device_t)
	backbone.eval()

	head, projection_meta = build_projection_module(
		method=projection_method,
		input_dim=input_dim,
		output_dim=projection_dim,
		head_type=head_type,
		hidden_dim=hidden_dim,
		dropout=dropout,
		zca_epsilon=zca_epsilon,
		hyper_top_k=hyper_top_k,
		hyper_encoder_type=hyper_encoder_type,
		hyper_attn_heads=hyper_attn_heads,
		hyper_attn_layers=hyper_attn_layers,
		device=device_t,
	)

	pca_model: PCA | None = None
	rp_model: GaussianRandomProjection | None = None
	pca_heldout_model: PCA | None = None
	rp_heldout_model: GaussianRandomProjection | None = None

	if len(val_cache) > 0:
		support_features_list: list[np.ndarray] = []
		for episode in val_cache:
			X_episode = episode["X"]
			support_idx = episode["support_idx"]
			X_support = X_episode[support_idx]
			support_features_list.append(X_support)
		X_support_all = np.vstack(support_features_list)

		pca_model = PCA(n_components=min(projection_dim or 128, X_support_all.shape[0], X_support_all.shape[1]))
		pca_model.fit(X_support_all)

		rp_model = GaussianRandomProjection(n_components=projection_dim or 128, random_state=sampling_config.seed)
		rp_model.fit(X_support_all)

		print(f"[info] Fitted PCA and RandomProjection baselines on {X_support_all.shape[0]} support examples")

	if len(val_heldout_cache) > 0:
		heldout_support_features_list: list[np.ndarray] = []
		for episode in val_heldout_cache:
			X_episode = episode["X"]
			support_idx = episode["support_idx"]
			X_support = X_episode[support_idx]
			heldout_support_features_list.append(X_support)
		X_heldout_support_all = np.vstack(heldout_support_features_list)

		pca_heldout_model = PCA(n_components=min(projection_dim or 128, X_heldout_support_all.shape[0], X_heldout_support_all.shape[1]))
		pca_heldout_model.fit(X_heldout_support_all)

		rp_heldout_model = GaussianRandomProjection(n_components=projection_dim or 128, random_state=sampling_config.seed + 100)
		rp_heldout_model.fit(X_heldout_support_all)

		print(f"[info] Fitted held-out PCA and RandomProjection baselines on {X_heldout_support_all.shape[0]} support examples")

	optimizer = torch.optim.AdamW(
		head.parameters(),
		lr=learning_rate,
		weight_decay=weight_decay,
	)
	criterion = nn.CrossEntropyLoss()
	optimizer.zero_grad(set_to_none=True)
	accum_counter = 0
	accum_target = min(int(gradient_accumulation_steps), int(num_steps))

	history: dict[str, list[float]] = {
		"loss": [],
		"query_fraction": [],
		"query_size": [],
		"dataset_size": [],
		"n_classes": [],
		"val_step": [],
		"val_loss": [],
		"val_accuracy": [],
		"val_pca_accuracy": [],
		"val_rp_accuracy": [],
		"val_heldout_loss": [],
		"val_heldout_accuracy": [],
		"val_heldout_pca_accuracy": [],
		"val_heldout_rp_accuracy": [],
	}
	best_val_acc = float("-inf")
	best_val_loss = float("inf")
	best_state: dict[str, torch.Tensor] | None = None

	def _run_validation(step_for_eval: int) -> tuple[float, float, float, float]:
		if len(val_cache) == 0:
			raise RuntimeError("Validation called but validation sampler is not configured")

		head.eval()
		losses: list[float] = []
		accs: list[float] = []
		pca_accs: list[float] = []
		rp_accs: list[float] = []

		with torch.no_grad():
			for episode in val_cache:
				X_val_step = torch.as_tensor(episode["X"], dtype=torch.float32, device=device_t)
				y_val_step = torch.as_tensor(episode["y"], dtype=torch.long, device=device_t)
				support_idx = torch.as_tensor(episode["support_idx"], device=device_t, dtype=torch.long)
				query_idx = torch.as_tensor(episode["query_idx"], device=device_t, dtype=torch.long)

				X_proj_val = project_episode_features(
					method=projection_method,
					module=head,
					features=X_val_step,
					support_indices=support_idx,
					zca_epsilon=zca_epsilon,
				)
				X_support_val = X_proj_val.index_select(0, support_idx)
				y_support_val = y_val_step.index_select(0, support_idx)
				X_query_val = X_proj_val.index_select(0, query_idx)
				y_query_val = y_val_step.index_select(0, query_idx)
				y_support_local, y_query_local = remap_labels_from_support(y_support_val, y_query_val)

				logits_val = backbone(
					X_support_val,
					y_support_local,
					X_query_val,
					step_index=step_for_eval - 1,
				)
				loss_val = float(criterion(logits_val, y_query_local).detach().cpu().item())
				acc_val = float(
					(torch.argmax(logits_val, dim=1) == y_query_local)
					.float()
					.mean()
					.detach()
					.cpu()
					.item()
				)
				losses.append(loss_val)
				accs.append(acc_val)

				if pca_model is not None:
					X_val_np = episode["X"]
					support_idx_np = episode["support_idx"]
					query_idx_np = episode["query_idx"]

					X_support_pca_np = pca_model.transform(X_val_np[support_idx_np])
					X_query_pca_np = pca_model.transform(X_val_np[query_idx_np])

					X_support_pca = torch.as_tensor(X_support_pca_np, dtype=torch.float32, device=device_t)
					X_query_pca = torch.as_tensor(X_query_pca_np, dtype=torch.float32, device=device_t)
					y_support_pca = y_val_step.index_select(0, torch.as_tensor(support_idx_np, device=device_t, dtype=torch.long))
					y_query_pca = y_val_step.index_select(0, torch.as_tensor(query_idx_np, device=device_t, dtype=torch.long))
					y_support_pca_local, y_query_pca_local = remap_labels_from_support(y_support_pca, y_query_pca)

					logits_pca = backbone(
						X_support_pca,
						y_support_pca_local,
						X_query_pca,
						step_index=step_for_eval - 1,
					)
					acc_pca = float(
						(torch.argmax(logits_pca, dim=1) == y_query_pca_local)
						.float()
						.mean()
						.detach()
						.cpu()
						.item()
					)
					pca_accs.append(acc_pca)

				if rp_model is not None:
					X_val_np = episode["X"]
					support_idx_np = episode["support_idx"]
					query_idx_np = episode["query_idx"]

					X_support_rp_np = rp_model.transform(X_val_np[support_idx_np])
					X_query_rp_np = rp_model.transform(X_val_np[query_idx_np])

					X_support_rp = torch.as_tensor(X_support_rp_np, dtype=torch.float32, device=device_t)
					X_query_rp = torch.as_tensor(X_query_rp_np, dtype=torch.float32, device=device_t)
					y_support_rp = y_val_step.index_select(0, torch.as_tensor(support_idx_np, device=device_t, dtype=torch.long))
					y_query_rp = y_val_step.index_select(0, torch.as_tensor(query_idx_np, device=device_t, dtype=torch.long))
					y_support_rp_local, y_query_rp_local = remap_labels_from_support(y_support_rp, y_query_rp)

					logits_rp = backbone(
						X_support_rp,
						y_support_rp_local,
						X_query_rp,
						step_index=step_for_eval - 1,
					)
					acc_rp = float(
						(torch.argmax(logits_rp, dim=1) == y_query_rp_local)
						.float()
						.mean()
						.detach()
						.cpu()
						.item()
					)
					rp_accs.append(acc_rp)

		mean_pca_acc = float(np.mean(pca_accs)) if pca_accs else 0.0
		mean_rp_acc = float(np.mean(rp_accs)) if rp_accs else 0.0
		return float(np.mean(losses)), float(np.mean(accs)), mean_pca_acc, mean_rp_acc

	def _run_heldout_validation(step_for_eval: int) -> tuple[float, float, float, float]:
		if len(val_heldout_cache) == 0:
			raise RuntimeError("Held-out validation called but cache is empty")

		head.eval()
		losses: list[float] = []
		accs: list[float] = []
		pca_accs: list[float] = []
		rp_accs: list[float] = []

		with torch.no_grad():
			for episode in val_heldout_cache:
				X_val_step = torch.as_tensor(episode["X"], dtype=torch.float32, device=device_t)
				y_val_step = torch.as_tensor(episode["y"], dtype=torch.long, device=device_t)
				support_idx = torch.as_tensor(episode["support_idx"], device=device_t, dtype=torch.long)
				query_idx = torch.as_tensor(episode["query_idx"], device=device_t, dtype=torch.long)

				X_proj_val = project_episode_features(
					method=projection_method,
					module=head,
					features=X_val_step,
					support_indices=support_idx,
					zca_epsilon=zca_epsilon,
				)
				X_support_val = X_proj_val.index_select(0, support_idx)
				y_support_val = y_val_step.index_select(0, support_idx)
				X_query_val = X_proj_val.index_select(0, query_idx)
				y_query_val = y_val_step.index_select(0, query_idx)
				y_support_local, y_query_local = remap_labels_from_support(y_support_val, y_query_val)

				logits_val = backbone(
					X_support_val,
					y_support_local,
					X_query_val,
					step_index=step_for_eval - 1,
				)
				loss_val = float(criterion(logits_val, y_query_local).detach().cpu().item())
				acc_val = float(
					(torch.argmax(logits_val, dim=1) == y_query_local)
					.float()
					.mean()
					.detach()
					.cpu()
					.item()
				)
				losses.append(loss_val)
				accs.append(acc_val)

				if pca_heldout_model is not None:
					X_val_np = episode["X"]
					support_idx_np = episode["support_idx"]
					query_idx_np = episode["query_idx"]

					X_support_pca_np = pca_heldout_model.transform(X_val_np[support_idx_np])
					X_query_pca_np = pca_heldout_model.transform(X_val_np[query_idx_np])

					X_support_pca = torch.as_tensor(X_support_pca_np, dtype=torch.float32, device=device_t)
					X_query_pca = torch.as_tensor(X_query_pca_np, dtype=torch.float32, device=device_t)
					y_support_pca = y_val_step.index_select(0, torch.as_tensor(support_idx_np, device=device_t, dtype=torch.long))
					y_query_pca = y_val_step.index_select(0, torch.as_tensor(query_idx_np, device=device_t, dtype=torch.long))
					y_support_pca_local, y_query_pca_local = remap_labels_from_support(y_support_pca, y_query_pca)

					logits_pca = backbone(
						X_support_pca,
						y_support_pca_local,
						X_query_pca,
						step_index=step_for_eval - 1,
					)
					acc_pca = float(
						(torch.argmax(logits_pca, dim=1) == y_query_pca_local)
						.float()
						.mean()
						.detach()
						.cpu()
						.item()
					)
					pca_accs.append(acc_pca)

				if rp_heldout_model is not None:
					X_val_np = episode["X"]
					support_idx_np = episode["support_idx"]
					query_idx_np = episode["query_idx"]

					X_support_rp_np = rp_heldout_model.transform(X_val_np[support_idx_np])
					X_query_rp_np = rp_heldout_model.transform(X_val_np[query_idx_np])

					X_support_rp = torch.as_tensor(X_support_rp_np, dtype=torch.float32, device=device_t)
					X_query_rp = torch.as_tensor(X_query_rp_np, dtype=torch.float32, device=device_t)
					y_support_rp = y_val_step.index_select(0, torch.as_tensor(support_idx_np, device=device_t, dtype=torch.long))
					y_query_rp = y_val_step.index_select(0, torch.as_tensor(query_idx_np, device=device_t, dtype=torch.long))
					y_support_rp_local, y_query_rp_local = remap_labels_from_support(y_support_rp, y_query_rp)

					logits_rp = backbone(
						X_support_rp,
						y_support_rp_local,
						X_query_rp,
						step_index=step_for_eval - 1,
					)
					acc_rp = float(
						(torch.argmax(logits_rp, dim=1) == y_query_rp_local)
						.float()
						.mean()
						.detach()
						.cpu()
						.item()
					)
					rp_accs.append(acc_rp)

		mean_pca_acc = float(np.mean(pca_accs)) if pca_accs else 0.0
		mean_rp_acc = float(np.mean(rp_accs)) if rp_accs else 0.0
		return float(np.mean(losses)), float(np.mean(accs)), mean_pca_acc, mean_rp_acc

	for step in range(1, num_steps + 1):
		head.train()

		if step == 1:
			X_np, y_np = X0, y0
			meta = _meta0
		else:
			X_np, y_np, meta = sample_one_dataset(
				sampler=sampler,
				min_dataset_size=sampling_config.min_dataset_size,
				max_dataset_size=sampling_config.max_dataset_size,
				min_classes=sampling_config.min_classes,
				max_classes=sampling_config.max_classes,
				min_per_class=sampling_config.min_per_class,
				remap_labels=sampling_config.remap_labels,
				allow_replacement=sampling_config.allow_replacement,
			)

		X_step = torch.as_tensor(X_np, dtype=torch.float32, device=device_t)
		y_step = torch.as_tensor(y_np, dtype=torch.long, device=device_t)

		support_idx_np, query_idx_np = class_safe_support_query_indices(
			y=y_step,
			rng=split_rng,
			frac_min=query_fraction_min,
			frac_max=query_fraction_max,
			min_query_size=min_query_size,
		)
		support_idx = torch.as_tensor(support_idx_np, device=device_t, dtype=torch.long)
		query_idx = torch.as_tensor(query_idx_np, device=device_t, dtype=torch.long)

		X_proj = project_episode_features(
			method=projection_method,
			module=head,
			features=X_step,
			support_indices=support_idx,
			zca_epsilon=zca_epsilon,
		)
		X_support = X_proj.index_select(0, support_idx)
		y_support = y_step.index_select(0, support_idx)
		X_query = X_proj.index_select(0, query_idx)
		y_query = y_step.index_select(0, query_idx)
		y_support_local, y_query_local = remap_labels_from_support(y_support, y_query)

		logits = backbone(X_support, y_support_local, X_query, step_index=step - 1)
		loss = criterion(logits, y_query_local)
		(loss / float(accum_target)).backward()
		accum_counter += 1

		should_step_optimizer = accum_counter >= accum_target
		if should_step_optimizer:
			if grad_clip_norm is not None:
				nn.utils.clip_grad_norm_(head.parameters(), grad_clip_norm)
			optimizer.step()
			optimizer.zero_grad(set_to_none=True)
			accum_counter = 0
			remaining_steps = num_steps - step
			if remaining_steps > 0:
				accum_target = min(int(gradient_accumulation_steps), int(remaining_steps))

		query_frac = float(y_query.shape[0] / max(1, X_step.shape[0]))
		history["loss"].append(float(loss.detach().cpu().item()))
		history["query_fraction"].append(query_frac)
		history["query_size"].append(float(y_query.shape[0]))
		history["dataset_size"].append(float(meta["dataset_size"]))
		history["n_classes"].append(float(meta["n_classes"]))

		if step == 1 or step % log_every == 0 or step == num_steps:
			n_pending = accum_counter
			print(
				f"[step {step:04d}/{num_steps}] "
				f"loss={history['loss'][-1]:.4f} "
				f"dataset={int(meta['dataset_size'])} "
				f"classes={int(meta['n_classes'])} "
				f"query={int(y_query.shape[0])}/{int(X_step.shape[0])} ({query_frac:.3f}) "
				f"accum_pending={n_pending}/{accum_target}"
			)

		if val_sampler is not None and (step % val_every == 0 or step == num_steps):
			val_loss, val_acc, val_pca_acc, val_rp_acc = _run_validation(step)
			history["val_step"].append(float(step))
			history["val_loss"].append(val_loss)
			history["val_accuracy"].append(val_acc)
			history["val_pca_accuracy"].append(val_pca_acc)
			history["val_rp_accuracy"].append(val_rp_acc)
			print(
				f"[val  step {step:04d}/{num_steps}] "
				f"loss={val_loss:.4f} head_acc={val_acc:.4f} pca_acc={val_pca_acc:.4f} rp_acc={val_rp_acc:.4f}"
			)

			is_better = (val_acc > best_val_acc) or (
				val_acc == best_val_acc and val_loss < best_val_loss
			)
			if is_better:
				best_val_acc = val_acc
				best_val_loss = val_loss
				best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}

		if len(val_heldout_cache) > 0 and (step % val_every == 0 or step == num_steps):
			ho_loss, ho_acc, ho_pca_acc, ho_rp_acc = _run_heldout_validation(step)
			history["val_heldout_loss"].append(ho_loss)
			history["val_heldout_accuracy"].append(ho_acc)
			history["val_heldout_pca_accuracy"].append(ho_pca_acc)
			history["val_heldout_rp_accuracy"].append(ho_rp_acc)
			print(
				f"[heldout step {step:04d}/{num_steps}] "
				f"loss={ho_loss:.4f} head_acc={ho_acc:.4f} pca_acc={ho_pca_acc:.4f} rp_acc={ho_rp_acc:.4f}"
			)

	if best_state is not None:
		head.load_state_dict(best_state)
		print(
			f"[done] Restored best validation head "
			f"(acc={best_val_acc:.4f}, loss={best_val_loss:.4f})"
		)

	return head, history, projection_meta
