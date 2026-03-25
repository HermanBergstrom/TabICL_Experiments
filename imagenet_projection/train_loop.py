"""Training loop for projection methods on sampled ImageNet episodes."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from torch.optim.lr_scheduler import LambdaLR

try:
	import wandb
except ImportError:
	wandb = None

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
	use_random_projection_init: bool = True,
	gradient_accumulation_steps: int = 1,
	ortho_lambda: float = 0.01,
	enable_wandb: bool = True,
	wandb_project: str = "imagenet-projection",
	wandb_run_name: str | None = None,
	checkpoint_dir: str | Path = "model_weights/hypernetwork_checkpoints",
	resume_training: bool = False,
	checkpoint_interval_steps: int = 50,
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
	if ortho_lambda < 0:
		raise ValueError("ortho_lambda must be >= 0")
	if checkpoint_interval_steps <= 0:
		raise ValueError("checkpoint_interval_steps must be > 0")

	device_t = torch.device(device)

	if enable_wandb and wandb is None:
		print("[warn] wandb is not installed; continuing without W&B logging")
		enable_wandb = False
	if enable_wandb and wandb is not None and wandb.run is None:
		wandb.init(
			project=wandb_project,
			name=wandb_run_name,
			config={
				"num_steps": int(num_steps),
				"learning_rate": float(learning_rate),
				"weight_decay": float(weight_decay),
				"query_fraction_min": float(query_fraction_min),
				"query_fraction_max": float(query_fraction_max),
				"min_query_size": int(min_query_size),
				"projection_method": str(projection_method),
				"head_type": str(head_type),
				"projection_dim": int(projection_dim) if projection_dim is not None else None,
				"hidden_dim": int(hidden_dim),
				"dropout": float(dropout),
				"gradient_accumulation_steps": int(gradient_accumulation_steps),
				"ortho_lambda": float(ortho_lambda),
				"zca_epsilon": float(zca_epsilon),
				"hyper_top_k": int(hyper_top_k),
				"hyper_encoder_type": str(hyper_encoder_type),
				"hyper_attn_heads": int(hyper_attn_heads),
				"hyper_attn_layers": int(hyper_attn_layers),
				"use_random_projection_init": bool(use_random_projection_init),
				"sampling_min_dataset_size": int(sampling_config.min_dataset_size),
				"sampling_max_dataset_size": int(sampling_config.max_dataset_size),
				"sampling_min_classes": int(sampling_config.min_classes),
				"sampling_max_classes": int(sampling_config.max_classes),
				"sampling_min_per_class": int(sampling_config.min_per_class),
				"sampling_distribution": str(sampling_config.sampling_distribution),
				"sampling_backend": str(sampling_config.sampler_backend),
				"enable_hard_sampling": bool(sampling_config.enable_hard_sampling),
				"hard_sampling_temperature": float(sampling_config.hard_sampling_temperature),
				"hard_sampling_prob": float(sampling_config.hard_sampling_prob),
				"hard_sampling_device": str(sampling_config.hard_sampling_device),
			},
		)

	def _calculate_orthogonal_penalty(W: torch.Tensor | None) -> torch.Tensor:
		if W is None:
			return torch.zeros((), device=device_t, dtype=torch.float32)
		output_dim = int(W.shape[1])
		wt_w = torch.matmul(W.transpose(0, 1), W)
		identity = torch.eye(output_dim, device=W.device, dtype=W.dtype)
		return torch.nn.functional.mse_loss(wt_w, identity)

	def _global_grad_norm(module: torch.nn.Module) -> float:
		total_sq = 0.0
		for p in module.parameters():
			if p.grad is None:
				continue
			param_norm = p.grad.detach().data.norm(2).item()
			total_sq += param_norm ** 2
		return float(total_sq ** 0.5)

	def _synchronized_perf_counter() -> float:
		if device_t.type == "cuda":
			torch.cuda.synchronize(device_t)
		return time.perf_counter()

	enable_attention_spectral_timers = (
		projection_method == "spectral_hypernetwork" and hyper_encoder_type == "attention"
	)
	if enable_attention_spectral_timers:
		print("[info] Attention-spectral timing enabled: eigendecomp, hypernetwork encoder/decoder, TabICL forward")

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
		enable_hard_sampling=sampling_config.enable_hard_sampling,
		num_classes=sampling_config.num_classes,
		hard_sampling_temperature=sampling_config.hard_sampling_temperature,
		hard_sampling_prob=sampling_config.hard_sampling_prob,
		hard_sampling_device=sampling_config.hard_sampling_device,
	)
	split_rng = np.random.default_rng(sampling_config.seed + 7)
	if val_sampling_config is not None and val_every > 0:
		val_sampler = ShardedEmbeddingSampler(
			output_dir=val_sampling_config.output_dir,
			split=val_sampling_config.split,
			seed=val_sampling_config.seed,
			max_cached_shards=val_sampling_config.max_cached_shards,
			sampler_backend=val_sampling_config.sampler_backend,
			enable_hard_sampling=val_sampling_config.enable_hard_sampling,
			num_classes=val_sampling_config.num_classes,
			hard_sampling_temperature=val_sampling_config.hard_sampling_temperature,
			hard_sampling_prob=val_sampling_config.hard_sampling_prob,
			hard_sampling_device=val_sampling_config.hard_sampling_device,
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
			enable_hard_sampling=sampling_config.enable_hard_sampling,
			num_classes=sampling_config.num_classes,
			hard_sampling_temperature=sampling_config.hard_sampling_temperature,
			hard_sampling_prob=sampling_config.hard_sampling_prob,
			hard_sampling_device=sampling_config.hard_sampling_device,
		)
		split_rng_heldout = np.random.default_rng(sampling_config.seed + 19)
		print(f"[info] Created held-out validation sampler with {len(val_heldout_sampler.available_classes)} held-out classes")

	val_cache: list[dict[str, Any]] = []
	if val_sampler is not None and val_sampling_config is not None and split_rng_val is not None:
		for _ in range(val_batches):
			val_sample_t0 = time.perf_counter()
			X_np_val, y_np_val, meta_val = sample_one_dataset(
				sampler=val_sampler,
				min_dataset_size=val_sampling_config.min_dataset_size,
				max_dataset_size=val_sampling_config.max_dataset_size,
				min_classes=val_sampling_config.min_classes,
				max_classes=val_sampling_config.max_classes,
				min_per_class=val_sampling_config.min_per_class,
				remap_labels=val_sampling_config.remap_labels,
				allow_replacement=val_sampling_config.allow_replacement,
				sampling_distribution=val_sampling_config.sampling_distribution,
			)
			val_sample_t1 = time.perf_counter()
			meta_val["sampling_time_ms"] = (val_sample_t1 - val_sample_t0) * 1000.0

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
			ho_sample_t0 = time.perf_counter()
			X_np_ho, y_np_ho, meta_ho = sample_one_dataset(
				sampler=val_heldout_sampler,
				min_dataset_size=sampling_config.min_dataset_size,
				max_dataset_size=sampling_config.max_dataset_size,
				min_classes=sampling_config.min_classes,
				max_classes=sampling_config.max_classes,
				min_per_class=sampling_config.min_per_class,
				remap_labels=sampling_config.remap_labels,
				allow_replacement=sampling_config.allow_replacement,
				sampling_distribution=sampling_config.sampling_distribution,
			)
			ho_sample_t1 = time.perf_counter()
			meta_ho["sampling_time_ms"] = (ho_sample_t1 - ho_sample_t0) * 1000.0

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
		sampling_distribution=sampling_config.sampling_distribution,
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
		use_random_projection_init=use_random_projection_init,
		device=device_t,
	)

	pca_models: list[PCA] = []
	rp_model: GaussianRandomProjection | None = None
	pca_heldout_models: list[PCA] = []
	rp_heldout_model: GaussianRandomProjection | None = None

	if len(val_cache) > 0:
		support_features_list: list[np.ndarray] = []
		for episode in val_cache:
			X_episode = episode["X"]
			support_idx = episode["support_idx"]
			X_support = X_episode[support_idx]
			support_features_list.append(X_support)
			pca_model_episode = PCA(
				n_components=min(
					projection_dim or 128,
					X_support.shape[0],
					X_support.shape[1],
				)
			)
			pca_model_episode.fit(X_support)
			pca_models.append(pca_model_episode)

		X_support_all = np.vstack(support_features_list)

		rp_model = GaussianRandomProjection(n_components=projection_dim or 128, random_state=sampling_config.seed)
		rp_model.fit(X_support_all)

		print(
			f"[info] Fitted {len(pca_models)} per-episode PCA models and "
			f"RandomProjection baseline on {X_support_all.shape[0]} support examples"
		)

	if len(val_heldout_cache) > 0:
		heldout_support_features_list: list[np.ndarray] = []
		for episode in val_heldout_cache:
			X_episode = episode["X"]
			support_idx = episode["support_idx"]
			X_support = X_episode[support_idx]
			heldout_support_features_list.append(X_support)
			pca_heldout_model_episode = PCA(
				n_components=min(
					projection_dim or 128,
					X_support.shape[0],
					X_support.shape[1],
				)
			)
			pca_heldout_model_episode.fit(X_support)
			pca_heldout_models.append(pca_heldout_model_episode)
		X_heldout_support_all = np.vstack(heldout_support_features_list)

		rp_heldout_model = GaussianRandomProjection(n_components=projection_dim or 128, random_state=sampling_config.seed + 100)
		rp_heldout_model.fit(X_heldout_support_all)

		print(
			f"[info] Fitted {len(pca_heldout_models)} held-out per-episode PCA models and "
			f"RandomProjection baseline on {X_heldout_support_all.shape[0]} support examples"
		)

	optimizer = torch.optim.AdamW(
		head.parameters(),
		lr=learning_rate,
		weight_decay=weight_decay,
	)
	total_optimizer_steps = max(1, int(math.ceil(float(num_steps) / float(gradient_accumulation_steps))))
	warmup_steps = max(1, int(total_optimizer_steps * 0.10))

	def _lr_lambda(current_step: int) -> float:
		if current_step < warmup_steps:
			return float(current_step) / float(max(1, warmup_steps))
		return 1.0

	scheduler = LambdaLR(optimizer, _lr_lambda)
	criterion = nn.CrossEntropyLoss()
	optimizer.zero_grad(set_to_none=True)
	accum_counter = 0
	accum_target = min(int(gradient_accumulation_steps), int(num_steps))
	optimizer_step_count = 0
	start_step = 1

	checkpoint_dir_path = Path(checkpoint_dir)
	checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
	latest_checkpoint_path = checkpoint_dir_path / "latest.pt"
	best_heldout_checkpoint_path = checkpoint_dir_path / "best_heldout.pt"

	history: dict[str, list[float]] = {
		"loss": [],
		"ce_loss": [],
		"ortho_loss": [],
		"total_loss": [],
		"step_time_ms": [],
		"timing_eigendecomp_ms": [],
		"timing_hyper_encoder_ms": [],
		"timing_hyper_decoder_ms": [],
		"timing_hyper_forward_ms": [],
		"timing_tabicl_forward_ms": [],
		"grad_norm": [],
		"W_norm": [],
		"learning_rate": [],
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
		"timing_sample_ms": [],
		"timing_val_sample_ms": [],
		"timing_heldout_sample_ms": [],
		"used_hard_sampling": [],
	}
	best_val_acc = float("-inf")
	best_val_loss = float("inf")
	best_state: dict[str, torch.Tensor] | None = None
	best_heldout_acc = float("-inf")
	best_heldout_loss = float("inf")

	def _checkpoint_payload(*, step_value: int, which: str) -> dict[str, Any]:
		return {
			"which": which,
			"step": int(step_value),
			"num_steps": int(num_steps),
			"optimizer_step_count": int(optimizer_step_count),
			"projection_method": str(projection_method),
			"projection_meta": projection_meta,
			"head_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"history": history,
			"best_val_acc": float(best_val_acc),
			"best_val_loss": float(best_val_loss),
			"best_heldout_acc": float(best_heldout_acc),
			"best_heldout_loss": float(best_heldout_loss),
		}

	if resume_training and latest_checkpoint_path.exists():
		checkpoint = torch.load(latest_checkpoint_path, map_location="cpu", weights_only=False)
		if not isinstance(checkpoint, dict):
			raise ValueError(f"Invalid checkpoint format at {latest_checkpoint_path}")
		head_state_dict = checkpoint.get("head_state_dict")
		if isinstance(head_state_dict, dict):
			head.load_state_dict(head_state_dict)
		optimizer_state_dict = checkpoint.get("optimizer_state_dict")
		if isinstance(optimizer_state_dict, dict):
			optimizer.load_state_dict(optimizer_state_dict)
		scheduler_state_dict = checkpoint.get("scheduler_state_dict")
		if isinstance(scheduler_state_dict, dict):
			scheduler.load_state_dict(scheduler_state_dict)

		checkpoint_history = checkpoint.get("history")
		if isinstance(checkpoint_history, dict):
			for key in history:
				loaded_values = checkpoint_history.get(key)
				if isinstance(loaded_values, list):
					history[key] = loaded_values

		best_val_acc = float(checkpoint.get("best_val_acc", best_val_acc))
		best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
		best_heldout_acc = float(checkpoint.get("best_heldout_acc", best_heldout_acc))
		best_heldout_loss = float(checkpoint.get("best_heldout_loss", best_heldout_loss))
		optimizer_step_count = int(checkpoint.get("optimizer_step_count", optimizer_step_count))
		start_step = int(checkpoint.get("step", 0)) + 1
		if start_step > num_steps:
			print(
				f"[info] Resume checkpoint already at step {start_step - 1}; "
				f"num_steps={num_steps}, nothing to train."
			)
			return head, history, projection_meta
		print(
			f"[info] Resumed training from {latest_checkpoint_path} "
			f"at step={start_step}"
		)

	def _run_validation(step_for_eval: int) -> tuple[float, float, float, float]:
		if len(val_cache) == 0:
			raise RuntimeError("Validation called but validation sampler is not configured")

		head.eval()
		losses: list[float] = []
		accs: list[float] = []
		pca_accs: list[float] = []
		rp_accs: list[float] = []

		with torch.no_grad():
			for episode_idx, episode in enumerate(val_cache):
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

				if episode_idx < len(pca_models):
					pca_model = pca_models[episode_idx]
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
			for episode_idx, episode in enumerate(val_heldout_cache):
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

				if episode_idx < len(pca_heldout_models):
					pca_heldout_model = pca_heldout_models[episode_idx]
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

	for step in range(start_step, num_steps + 1):
		step_t0 = time.perf_counter()
		head.train()

		if step == 1:
			X_np, y_np = X0, y0
			meta = _meta0
			sample_time_ms = 0.0
		else:
			sample_t0 = time.perf_counter()
			X_np, y_np, meta = sample_one_dataset(
				sampler=sampler,
				min_dataset_size=sampling_config.min_dataset_size,
				max_dataset_size=sampling_config.max_dataset_size,
				min_classes=sampling_config.min_classes,
				max_classes=sampling_config.max_classes,
				min_per_class=sampling_config.min_per_class,
				remap_labels=sampling_config.remap_labels,
				allow_replacement=sampling_config.allow_replacement,
				sampling_distribution=sampling_config.sampling_distribution,
			)
			sample_t1 = time.perf_counter()
			sample_time_ms = (sample_t1 - sample_t0) * 1000.0

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

		proj_out = project_episode_features(
			method=projection_method,
			module=head,
			features=X_step,
			support_indices=support_idx,
			zca_epsilon=zca_epsilon,
			return_projection_matrix=True,
		)
		if isinstance(proj_out, tuple):
			X_proj, W_generated = proj_out
		else:
			X_proj = proj_out
			W_generated = None

		if enable_attention_spectral_timers:
			hyper_timing = getattr(head, "last_profile", {})
			history["timing_eigendecomp_ms"].append(float(hyper_timing.get("eigendecomp_ms", 0.0)))
			history["timing_hyper_encoder_ms"].append(float(hyper_timing.get("encoder_ms", 0.0)))
			history["timing_hyper_decoder_ms"].append(float(hyper_timing.get("decoder_ms", 0.0)))
			history["timing_hyper_forward_ms"].append(float(hyper_timing.get("hypernetwork_forward_ms", 0.0)))
		else:
			history["timing_eigendecomp_ms"].append(0.0)
			history["timing_hyper_encoder_ms"].append(0.0)
			history["timing_hyper_decoder_ms"].append(0.0)
			history["timing_hyper_forward_ms"].append(0.0)
		X_support = X_proj.index_select(0, support_idx)
		y_support = y_step.index_select(0, support_idx)
		X_query = X_proj.index_select(0, query_idx)
		y_query = y_step.index_select(0, query_idx)
		y_support_local, y_query_local = remap_labels_from_support(y_support, y_query)

		tabicl_t0 = _synchronized_perf_counter() if enable_attention_spectral_timers else 0.0
		logits = backbone(X_support, y_support_local, X_query, step_index=step - 1)
		if enable_attention_spectral_timers:
			tabicl_t1 = _synchronized_perf_counter()
			history["timing_tabicl_forward_ms"].append(float((tabicl_t1 - tabicl_t0) * 1000.0))
		else:
			history["timing_tabicl_forward_ms"].append(0.0)
		ce_loss = criterion(logits, y_query_local)
		ortho_loss = _calculate_orthogonal_penalty(W_generated)
		total_loss = ce_loss + (float(ortho_lambda) * ortho_loss)
		(total_loss / float(accum_target)).backward()
		accum_counter += 1

		should_step_optimizer = accum_counter >= accum_target
		if should_step_optimizer:
			grad_norm = _global_grad_norm(head)
			w_norm = float(W_generated.detach().norm(2).item()) if W_generated is not None else 0.0
			if grad_clip_norm is not None:
				nn.utils.clip_grad_norm_(head.parameters(), grad_clip_norm)
			optimizer.step()
			scheduler.step()
			optimizer_step_count += 1
			current_lr = float(scheduler.get_last_lr()[0])
			history["grad_norm"].append(grad_norm)
			history["W_norm"].append(w_norm)
			history["learning_rate"].append(current_lr)
			if enable_wandb and wandb is not None:
				log_payload: dict[str, float | int] = {
					"train/ce_loss": float(ce_loss.detach().cpu().item()),
					"train/ortho_loss": float(ortho_loss.detach().cpu().item()),
					"train/total_loss": float(total_loss.detach().cpu().item()),
					"metrics/grad_norm": grad_norm,
					"metrics/W_norm": w_norm,
					"metrics/query_fraction": float(y_query.shape[0] / max(1, X_step.shape[0])),
					"metrics/learning_rate": current_lr,
					"metrics/optimizer_step": optimizer_step_count,
					"timing/sample_ms": sample_time_ms,
					"timing/used_hard_sampling": float(meta.get("used_hard_sampling", False)),
					"step": step,
				}
				if enable_attention_spectral_timers:
					log_payload["timing/eigendecomp_ms"] = history["timing_eigendecomp_ms"][-1]
					log_payload["timing/hyper_encoder_ms"] = history["timing_hyper_encoder_ms"][-1]
					log_payload["timing/hyper_decoder_ms"] = history["timing_hyper_decoder_ms"][-1]
					log_payload["timing/hyper_forward_ms"] = history["timing_hyper_forward_ms"][-1]
					log_payload["timing/tabicl_forward_ms"] = history["timing_tabicl_forward_ms"][-1]
				wandb.log(log_payload)
			if (step % checkpoint_interval_steps == 0) or (step == num_steps):
				torch.save(
					_checkpoint_payload(step_value=step, which="latest"),
					latest_checkpoint_path,
				)
			optimizer.zero_grad(set_to_none=True)
			accum_counter = 0
			remaining_steps = num_steps - step
			if remaining_steps > 0:
				accum_target = min(int(gradient_accumulation_steps), int(remaining_steps))

		query_frac = float(y_query.shape[0] / max(1, X_step.shape[0]))
		loss_value = float(total_loss.detach().cpu().item())
		history["loss"].append(loss_value)
		history["ce_loss"].append(float(ce_loss.detach().cpu().item()))
		history["ortho_loss"].append(float(ortho_loss.detach().cpu().item()))
		history["total_loss"].append(loss_value)
		history["query_fraction"].append(query_frac)
		history["query_size"].append(float(y_query.shape[0]))
		history["dataset_size"].append(float(meta["dataset_size"]))
		history["n_classes"].append(float(meta["n_classes"]))
		history["step_time_ms"].append(float((time.perf_counter() - step_t0) * 1000.0))
		history["timing_sample_ms"].append(float(sample_time_ms))
		history["used_hard_sampling"].append(float(meta.get("used_hard_sampling", False)))

		if step == 1 or step % log_every == 0 or step == num_steps:
			n_pending = accum_counter
			timing_suffix = ""
			if enable_attention_spectral_timers:
				timing_suffix = (
					f" eig_ms={history['timing_eigendecomp_ms'][-1]:.2f}"
					f" hyper_enc_ms={history['timing_hyper_encoder_ms'][-1]:.2f}"
					f" hyper_dec_ms={history['timing_hyper_decoder_ms'][-1]:.2f}"
					f" tabicl_ms={history['timing_tabicl_forward_ms'][-1]:.2f}"
				)
			print(
				f"[step {step:04d}/{num_steps}] "
				f"loss={history['loss'][-1]:.4f} "
				f"dataset={int(meta['dataset_size'])} "
				f"classes={int(meta['n_classes'])} "
				f"query={int(y_query.shape[0])}/{int(X_step.shape[0])} ({query_frac:.3f}) "
				f"accum_pending={n_pending}/{accum_target}"
				f"{timing_suffix}"
			)

		if val_sampler is not None and (step % val_every == 0 or step == num_steps):
			val_loss, val_acc, val_pca_acc, val_rp_acc = _run_validation(step)
			avg_step_time_ms = float(np.mean(history["step_time_ms"])) if history["step_time_ms"] else 0.0
			history["val_step"].append(float(step))
			history["val_loss"].append(val_loss)
			history["val_accuracy"].append(val_acc)
			history["val_pca_accuracy"].append(val_pca_acc)
			history["val_rp_accuracy"].append(val_rp_acc)
			avg_val_sample_time_ms = float(np.mean([m.get("sampling_time_ms", 0.0) for m in val_cache])) if val_cache else 0.0
			history["timing_val_sample_ms"].append(avg_val_sample_time_ms)
			print(
				f"[val  step {step:04d}/{num_steps}] "
				f"loss={val_loss:.4f} head_acc={val_acc:.4f} pca_acc={val_pca_acc:.4f} rp_acc={val_rp_acc:.4f} "
				f"avg_step_ms={avg_step_time_ms:.2f} avg_val_sample_ms={avg_val_sample_time_ms:.2f}"
			)
			if enable_wandb and wandb is not None:
				wandb.log(
					{
						"val/loss": float(val_loss),
						"val/accuracy": float(val_acc),
						"val/pca_accuracy": float(val_pca_acc),
						"val/rp_accuracy": float(val_rp_acc),
						"timing/avg_step_ms": avg_step_time_ms,
						"timing/avg_val_sample_ms": avg_val_sample_time_ms,
						"step": step,
					}
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
			avg_heldout_sample_time_ms = float(np.mean([m.get("sampling_time_ms", 0.0) for m in val_heldout_cache])) if val_heldout_cache else 0.0
			history["timing_heldout_sample_ms"].append(avg_heldout_sample_time_ms)
			print(
				f"[heldout step {step:04d}/{num_steps}] "
				f"loss={ho_loss:.4f} head_acc={ho_acc:.4f} pca_acc={ho_pca_acc:.4f} rp_acc={ho_rp_acc:.4f} "
				f"avg_heldout_sample_ms={avg_heldout_sample_time_ms:.2f}"
			)
			if enable_wandb and wandb is not None:
				wandb.log(
					{
						"val_heldout/loss": float(ho_loss),
						"val_heldout/accuracy": float(ho_acc),
						"val_heldout/pca_accuracy": float(ho_pca_acc),
						"val_heldout/rp_accuracy": float(ho_rp_acc),
						"timing/avg_heldout_sample_ms": avg_heldout_sample_time_ms,
						"step": step,
					}
				)

			is_best_heldout = (ho_acc > best_heldout_acc) or (
				ho_acc == best_heldout_acc and ho_loss < best_heldout_loss
			)
			if is_best_heldout:
				best_heldout_acc = ho_acc
				best_heldout_loss = ho_loss
				torch.save(
					_checkpoint_payload(step_value=step, which="best_heldout"),
					best_heldout_checkpoint_path,
				)
				print(
					f"[info] Saved new best held-out checkpoint "
					f"(acc={best_heldout_acc:.4f}, loss={best_heldout_loss:.4f}) "
					f"to {best_heldout_checkpoint_path}"
				)

	if best_state is not None:
		head.load_state_dict(best_state)
		print(
			f"[done] Restored best validation head "
			f"(acc={best_val_acc:.4f}, loss={best_val_loss:.4f})"
		)

	torch.save(
		_checkpoint_payload(step_value=num_steps, which="latest"),
		latest_checkpoint_path,
	)
	print(f"[done] Saved latest training checkpoint to: {latest_checkpoint_path}")
	if best_heldout_checkpoint_path.exists():
		print(f"[done] Best held-out checkpoint path: {best_heldout_checkpoint_path}")

	return head, history, projection_meta
