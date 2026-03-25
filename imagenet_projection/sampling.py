"""Sampling utilities over persisted ImageNet embedding shards."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import SamplingConfig


def _manifest_path(output_dir: Path, split: str) -> Path:
	return output_dir / f"{split}_manifest.json"


def _sample_from_distribution(
	rng: np.random.Generator,
	min_val: int,
	max_val: int,
	distribution: str = "log-uniform",
) -> int:
	"""Sample a value from specified distribution.

	Args:
		rng: Random number generator.
		min_val: Minimum value (inclusive).
		max_val: Maximum value (inclusive).
		distribution: Sampling strategy ("uniform" or "log-uniform").

	Returns:
		Sampled value.
	"""
	if distribution == "log-uniform":
		# Sample uniformly in log-space and exponentiate back.
		log_min = math.log(float(min_val))
		log_max = math.log(float(max_val))
		log_sample = float(rng.uniform(log_min, log_max))
		return int(math.exp(log_sample))
	elif distribution == "uniform":
		return int(rng.integers(min_val, max_val + 1))
	else:
		raise ValueError(
			f"Unknown sampling_distribution: {distribution}. "
			"Supported: 'uniform', 'log-uniform'"
		)


class HardNegativeClassSampler:
	"""Hard negative class sampler using cosine similarity of class centroids.
	
	Selects challenging classification tasks by sampling classes that are
	difficult to distinguish (similar in feature space) using a 50/50 strategy:
	- 50% of the time: uniform random class selection
	- 50% of the time: hard negatives guided by class centroid similarity
	"""

	def __init__(
		self,
		class_centroids: torch.Tensor,
		device: torch.device | str = "cpu",
		temperature: float = 0.1,
	) -> None:
		"""Initialize with class centroids and precompute similarity matrix.
		
		Args:
			class_centroids: Tensor of shape [num_classes, feature_dim].
			device: Device to place the similarity matrix on (GPU recommended).
			temperature: Temperature parameter for softmax. Lower values enforce
				stricter nearest neighbor selection. Default 0.1.
		"""
		self.device = torch.device(device) if isinstance(device, str) else device
		self.temperature = temperature
		self.num_classes = class_centroids.shape[0]

		# L2-normalize centroids for cosine similarity computation
		centroids_norm = F.normalize(class_centroids.to(self.device), p=2, dim=1)
		
		# Precompute the num_classes x num_classes cosine similarity matrix
		# Value range: [-1, 1], where 1 indicates identical classes
		self.sim_matrix = torch.matmul(centroids_norm, centroids_norm.T)

	@torch.no_grad()
	def sample_classes(
		self,
		k_classes: int,
		hard_prob: float = 0.5,
	) -> torch.Tensor:
		"""Sample k_classes using hard negative sampling or uniform selection.
		
		With probability hard_prob, samples classes based on cosine similarity
		to already-selected classes (hard negatives). Otherwise, samples uniformly.
		
		Args:
			k_classes: Number of classes to sample.
			hard_prob: Probability of using hard sampling (vs. uniform).
				Default 0.5 (50/50 split).
		
		Returns:
			Tensor of shape [k_classes] containing selected class indices.
		
		Raises:
			ValueError: If k_classes > num_classes or k_classes <= 0.
		"""
		if k_classes <= 0:
			raise ValueError(f"k_classes must be > 0, got {k_classes}")
		if k_classes > self.num_classes:
			raise ValueError(
				f"k_classes={k_classes} exceeds num_classes={self.num_classes}"
			)

		# 50/50 coin flip: uniform or hard sampling
		if torch.rand(1, device=self.device).item() > hard_prob:
			return torch.randperm(self.num_classes, device=self.device)[:k_classes]

		# Hard negative sampling: iteratively select difficult classes
		selected = torch.zeros(k_classes, dtype=torch.long, device=self.device)

		# Step 1: Pick the first class uniformly at random
		selected[0] = torch.randint(0, self.num_classes, (1,), device=self.device)

		# Track which classes have been selected
		available_mask = torch.ones(self.num_classes, dtype=torch.bool, device=self.device)
		available_mask[selected[0]] = False

		# Track the maximum similarity to ANY already-selected class
		# Initialized with the similarity row of the first selected class
		current_max_sim = self.sim_matrix[selected[0]].clone()

		# Step 2-K: Iteratively sample hard negatives
		for i in range(1, k_classes):
			# Create logits from similarity scores
			logits = current_max_sim.clone()
			
			# Mask out already-selected classes by setting logits to -inf
			# This ensures they get probability 0 after softmax
			logits[~available_mask] = -float("inf")

			# Apply temperature and convert to probabilities
			probs = F.softmax(logits / self.temperature, dim=0)

			# Sample the next class according to similarity-based probabilities
			next_class = torch.multinomial(probs, num_samples=1)[0]
			selected[i] = next_class

			# Update: mark this class as selected
			available_mask[next_class] = False
			
			# Magic step: Update running maximum similarity
			# For each candidate class, track its maximum similarity to any
			# already-selected class. Classes similar to already-selected ones
			# become more likely to be selected (higher difficulty).
			current_max_sim = torch.maximum(
				current_max_sim,
				self.sim_matrix[next_class]
			)

		return selected


def _compute_class_centroids(
	output_dir: Path,
	split: str,
	shards: list[dict[str, Any]],
	num_classes: int,
	device: torch.device | str = "cpu",
) -> torch.Tensor:
	"""Compute class centroids from shard data.
	
	Computes the mean feature vector for each class across all shards.
	
	Args:
		output_dir: Path to directory containing shard files.
		split: Data split name (e.g., "train", "val").
		shards: List of shard metadata dictionaries.
		num_classes: Total number of classes (e.g., 1000 for ImageNet).
		device: Device to place centroids on (CPU by default, GPU for faster sampling).
	
	Returns:
		Tensor of shape [num_classes, feature_dim] containing class centroids.
	"""
	if device is not None and not isinstance(device, torch.device):
		device = torch.device(device)

	# Accumulate features and counts per class
	class_sum: dict[int, torch.Tensor] = {}
	class_count: dict[int, int] = {}

	with torch.no_grad():
		for shard_meta in tqdm(shards, desc=f"compute centroids {split}"):
			shard_id = int(shard_meta["shard_id"])
			shard_path = output_dir / str(shard_meta["file"])
			
			if not shard_path.exists():
				raise FileNotFoundError(f"Shard listed in manifest is missing: {shard_path}")

			shard = torch.load(shard_path, map_location="cpu", weights_only=True)
			features = shard["features"].float()  # [num_samples, feature_dim]
			targets = shard["targets"].long()  # [num_samples]

			if not isinstance(features, torch.Tensor) or not isinstance(targets, torch.Tensor):
				raise TypeError(f"Shard must contain tensor features and targets: {shard_path}")

			# Accumulate by class
			for class_id in range(num_classes):
				mask = targets == class_id
				if mask.any():
					class_features = features[mask]
					class_sum[class_id] = (
						class_sum.get(class_id, torch.zeros_like(class_features[0]).cpu())
						+ class_features.sum(dim=0).cpu()
					)
					class_count[class_id] = class_count.get(class_id, 0) + int(mask.sum().item())

	# Compute centroids by averaging
	centroids_list: list[torch.Tensor] = []
	for class_id in range(num_classes):
		if class_id in class_count and class_count[class_id] > 0:
			centroid = class_sum[class_id] / class_count[class_id]
		else:
			# Handle missing classes with zero vector
			# (Infer feature dim from first available centroid)
			if centroids_list:
				centroid = torch.zeros_like(centroids_list[0])
			else:
				# Fallback: assume 768-dim features (typical for ViT-B)
				centroid = torch.zeros(768, dtype=torch.float32)
		centroids_list.append(centroid)

	centroids = torch.stack(centroids_list, dim=0).to(device)
	return centroids


class ShardedEmbeddingSampler:
	"""Class-first sampler over persisted shard embeddings."""

	def __init__(
		self,
		*,
		output_dir: Path,
		split: str,
		seed: int = 0,
		max_cached_shards: int = 8,
		sampler_backend: str = "shard",
		excluded_classes: set[int] | None = None,
		enable_hard_sampling: bool = False,
		num_classes: int = 1000,
		hard_sampling_temperature: float = 0.1,
		hard_sampling_prob: float = 0.5,
		hard_sampling_device: str = "cpu",
	) -> None:
		self.output_dir = output_dir
		self.split = split
		self.rng = np.random.default_rng(seed)
		self.max_cached_shards = max(1, int(max_cached_shards))
		if sampler_backend not in {"shard", "preload"}:
			raise ValueError("sampler_backend must be one of: shard, preload")
		self.sampler_backend = sampler_backend
		self.excluded_classes = excluded_classes if excluded_classes is not None else set()

		manifest_file = _manifest_path(output_dir, split)
		if not manifest_file.exists():
			raise FileNotFoundError(f"Manifest not found: {manifest_file}")
		with manifest_file.open("r", encoding="utf-8") as f:
			self.manifest = json.load(f)

		self.shards = list(self.manifest.get("shards", []))
		if len(self.shards) == 0:
			raise ValueError(f"No shards listed in manifest: {manifest_file}")
		self.shard_by_id = {int(s["shard_id"]): s for s in self.shards}

		if self.sampler_backend == "preload":
			self.features_all, self.class_to_locations = self._preload_all_data()
		else:
			self.features_all = None
			self.class_to_locations = self._build_class_index()
		all_classes = sorted(self.class_to_locations.keys())
		self.available_classes = sorted([c for c in all_classes if c not in self.excluded_classes])
		if len(self.available_classes) < 2:
			raise ValueError("Need at least 2 classes available for class-first sampling after excluding classes")

		self._shard_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

		# Optional hard negative sampling
		self.hard_sampler: HardNegativeClassSampler | None = None
		self.hard_sampling_prob = hard_sampling_prob
		if enable_hard_sampling:
			centroids = _compute_class_centroids(
				output_dir=output_dir,
				split=split,
				shards=self.shards,
				num_classes=num_classes,
				device=hard_sampling_device,
			)
			self.hard_sampler = HardNegativeClassSampler(
				class_centroids=centroids,
				device=hard_sampling_device,
				temperature=hard_sampling_temperature,
			)

	def _preload_all_data(self) -> tuple[torch.Tensor, dict[int, list[int]]]:
		"""Load all shard features into one tensor and build class->global-row index."""
		features_parts: list[torch.Tensor] = []
		class_to_rows: dict[int, list[int]] = {}
		row_offset = 0
		for shard_meta in tqdm(self.shards, desc=f"preload {self.split} shards"):
			shard_id = int(shard_meta["shard_id"])
			meta = self.shard_by_id.get(shard_id)
			if meta is None:
				raise KeyError(f"Missing shard metadata for shard_id={shard_id}")
			shard_path = self.output_dir / str(meta["file"])
			if not shard_path.exists():
				raise FileNotFoundError(f"Shard listed in manifest is missing: {shard_path}")

			shard = torch.load(shard_path, map_location="cpu", weights_only=True)
			features = shard["features"]
			targets = shard["targets"]
			if not isinstance(features, torch.Tensor) or not isinstance(targets, torch.Tensor):
				raise TypeError(f"Shard must contain tensor features and targets: {shard_path}")

			features_parts.append(features.cpu())
			targets_np = np.asarray(targets.cpu().numpy()).reshape(-1)
			for local_idx, class_id in enumerate(targets_np.tolist()):
				class_to_rows.setdefault(int(class_id), []).append(row_offset + int(local_idx))
			row_offset += int(features.shape[0])

		if len(features_parts) == 0:
			raise RuntimeError("No feature shards available to preload")
		return torch.cat(features_parts, dim=0), class_to_rows

	def _build_class_index(self) -> dict[int, list[tuple[int, int]]]:
		class_to_locations: dict[int, list[tuple[int, int]]] = {}
		for shard_meta in tqdm(self.shards, desc=f"index {self.split} classes"):
			shard_id = int(shard_meta["shard_id"])
			shard_path = self.output_dir / str(shard_meta["file"])
			if not shard_path.exists():
				raise FileNotFoundError(f"Shard listed in manifest is missing: {shard_path}")

			shard = torch.load(shard_path, map_location="cpu", weights_only=True)
			targets = shard["targets"]
			if not isinstance(targets, torch.Tensor):
				raise TypeError(f"Shard targets must be a tensor: {shard_path}")

			targets_np = np.asarray(targets.cpu().numpy()).reshape(-1)
			for local_idx, class_id in enumerate(targets_np.tolist()):
				class_to_locations.setdefault(int(class_id), []).append((shard_id, int(local_idx)))

		return class_to_locations

	def _get_shard(self, shard_id: int) -> dict[str, Any]:
		if self.sampler_backend == "preload":
			raise RuntimeError("_get_shard should not be called when sampler_backend='preload'")
		cached = self._shard_cache.get(shard_id)
		if cached is not None:
			self._shard_cache.move_to_end(shard_id)
			return cached

		meta = self.shard_by_id.get(shard_id)
		if meta is None:
			raise KeyError(f"Unknown shard_id={shard_id}")
		shard_path = self.output_dir / str(meta["file"])
		shard = torch.load(shard_path, map_location="cpu", weights_only=True)
		self._shard_cache[shard_id] = shard
		self._shard_cache.move_to_end(shard_id)

		while len(self._shard_cache) > self.max_cached_shards:
			self._shard_cache.popitem(last=False)

		return shard

	def _sample_class_counts(self, dataset_size: int, n_classes: int, min_per_class: int) -> np.ndarray:
		if n_classes * min_per_class > dataset_size:
			raise ValueError(
				"Requested min_per_class and class count exceed dataset size: "
				f"{n_classes} * {min_per_class} > {dataset_size}"
			)

		counts = np.full((n_classes,), min_per_class, dtype=np.int64)
		remaining = int(dataset_size - counts.sum())
		if remaining <= 0:
			return counts

		probs = self.rng.dirichlet(np.ones(n_classes, dtype=np.float64))
		extra = self.rng.multinomial(remaining, probs)
		counts += extra
		return counts

	def _fit_counts_to_capacity(
		self,
		counts: np.ndarray,
		capacities: np.ndarray,
	) -> np.ndarray:
		"""Adjust desired class counts to available rows without replacement."""
		if counts.shape != capacities.shape:
			raise ValueError("counts and capacities must have the same shape")

		adjusted = np.minimum(counts, capacities).astype(np.int64, copy=True)
		deficit = int(counts.sum() - adjusted.sum())
		if deficit <= 0:
			return adjusted

		spare = (capacities - adjusted).astype(np.int64, copy=False)
		while deficit > 0:
			candidates = np.flatnonzero(spare > 0)
			if candidates.size == 0:
				break
			pick = int(self.rng.choice(candidates))
			adjusted[pick] += 1
			spare[pick] -= 1
			deficit -= 1

		return adjusted

	def sample_dataset(
		self,
		*,
		dataset_size: int,
		n_classes: int,
		min_per_class: int = 1,
		remap_labels: bool = True,
		allow_replacement: bool = False,
	) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
		if dataset_size <= 0:
			raise ValueError("dataset_size must be > 0")
		if n_classes <= 1:
			raise ValueError("n_classes must be > 1")
		if n_classes > len(self.available_classes):
			raise ValueError(
				f"Requested {n_classes} classes but only {len(self.available_classes)} are available"
			)

		# Select classes: use hard negative sampling if available
		if self.hard_sampler is not None:
			# Hard sampling from all 1000 classes, then filter to available
			all_classes_set = set(self.available_classes)
			max_attempts = 10  # Prevent infinite loops
			for attempt in range(max_attempts):
				sampled = self.hard_sampler.sample_classes(
					k_classes=n_classes,
					hard_prob=self.hard_sampling_prob,
				)
				# Convert to available classes only
				class_ids_list = [int(c) for c in sampled.cpu().tolist() if int(c) in all_classes_set]
				if len(class_ids_list) >= n_classes:
					class_ids = np.asarray(class_ids_list[:n_classes], dtype=np.int64)
					break
			else:
				# Fallback to uniform sampling if hard sampling doesn't yield enough valid classes
				class_ids = self.rng.choice(self.available_classes, size=n_classes, replace=False)
				class_ids = np.asarray(class_ids, dtype=np.int64)
		else:
			# Uniform sampling from available classes
			class_ids = self.rng.choice(self.available_classes, size=n_classes, replace=False)
			class_ids = np.asarray(class_ids, dtype=np.int64)

		counts = self._sample_class_counts(dataset_size, n_classes, min_per_class)
		capacities = np.asarray(
			[len(self.class_to_locations[int(cls)]) for cls in class_ids.tolist()],
			dtype=np.int64,
		)
		requested_size = int(counts.sum())
		if not allow_replacement:
			counts = self._fit_counts_to_capacity(counts, capacities)
			effective_size = int(counts.sum())
			if effective_size <= 0:
				raise RuntimeError("No rows available for sampled classes without replacement")
		else:
			effective_size = requested_size

		if self.sampler_backend == "preload":
			assert self.features_all is not None
			global_rows: list[int] = []
			label_rows: list[int] = []
			for cls, n_take in zip(class_ids.tolist(), counts.tolist()):
				if n_take <= 0:
					continue
				pool = self.class_to_locations[int(cls)]
				chosen_idx = self.rng.choice(len(pool), size=n_take, replace=allow_replacement)
				chosen_rows = [int(pool[int(j)]) for j in np.asarray(chosen_idx, dtype=np.int64).tolist()]
				global_rows.extend(chosen_rows)
				label_rows.extend([int(cls)] * len(chosen_rows))

			if len(global_rows) == 0:
				raise RuntimeError("No rows sampled; check sampling settings")

			row_idx = torch.as_tensor(global_rows, dtype=torch.long)
			X = self.features_all.index_select(0, row_idx)
			y = torch.as_tensor(label_rows, dtype=torch.long)
		else:
			requested_by_shard: dict[int, list[int]] = {}
			requested_labels: dict[int, list[int]] = {}
			for cls, n_take in zip(class_ids.tolist(), counts.tolist()):
				if n_take <= 0:
					continue
				pool = self.class_to_locations[int(cls)]
				if len(pool) < n_take and not allow_replacement:
					raise RuntimeError(
						"Internal sampling inconsistency: n_take exceeds capacity after adjustment"
					)

				chosen_idx = self.rng.choice(len(pool), size=n_take, replace=allow_replacement)
				for idx in np.asarray(chosen_idx, dtype=np.int64).tolist():
					shard_id, local_idx = pool[int(idx)]
					requested_by_shard.setdefault(int(shard_id), []).append(int(local_idx))
					requested_labels.setdefault(int(shard_id), []).append(int(cls))

			features_parts: list[torch.Tensor] = []
			labels_parts: list[torch.Tensor] = []
			for shard_id, local_rows in requested_by_shard.items():
				shard = self._get_shard(shard_id)
				row_idx = torch.as_tensor(local_rows, dtype=torch.long)
				feats = shard["features"].index_select(0, row_idx)
				lbls = torch.as_tensor(requested_labels[shard_id], dtype=torch.long)
				features_parts.append(feats)
				labels_parts.append(lbls)

			if len(features_parts) == 0:
				raise RuntimeError("No rows sampled; check sampling settings")

			X = torch.cat(features_parts, dim=0)
			y = torch.cat(labels_parts, dim=0)

		perm = torch.randperm(X.shape[0])
		X = X.index_select(0, perm)
		y = y.index_select(0, perm)

		y_np = np.asarray(y.numpy(), dtype=np.int64)
		if remap_labels:
			uniq = np.unique(y_np)
			mapping = {int(c): i for i, c in enumerate(uniq.tolist())}
			y_np = np.asarray([mapping[int(v)] for v in y_np], dtype=np.int64)

		meta = {
			"requested_dataset_size": requested_size,
			"dataset_size": int(X.shape[0]),
			"n_classes": int(n_classes),
			"class_ids": class_ids.tolist(),
			"class_counts": counts.tolist(),
			"class_capacities": capacities.tolist(),
			"shrunk_for_capacity": bool((not allow_replacement) and (int(X.shape[0]) < requested_size)),
			"sampler_backend": self.sampler_backend,
			"used_hard_sampling": self.hard_sampler is not None,
		}
		return np.asarray(X.numpy()), y_np, meta


def sample_one_dataset(
	*,
	sampler: ShardedEmbeddingSampler,
	min_dataset_size: int,
	max_dataset_size: int,
	min_classes: int,
	max_classes: int,
	min_per_class: int,
	remap_labels: bool,
	allow_replacement: bool,
	sampling_distribution: str = "log-uniform",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
	max_classes_allowed = min(max_classes, len(sampler.available_classes))
	if min_classes > max_classes_allowed:
		raise ValueError(
			f"min_classes={min_classes} exceeds available class count={max_classes_allowed}"
		)

	dataset_size = _sample_from_distribution(
		rng=sampler.rng,
		min_val=min_dataset_size,
		max_val=max_dataset_size,
		distribution=sampling_distribution,
	)
	n_classes = _sample_from_distribution(
		rng=sampler.rng,
		min_val=min_classes,
		max_val=max_classes_allowed,
		distribution=sampling_distribution,
	)
	return sampler.sample_dataset(
		dataset_size=dataset_size,
		n_classes=n_classes,
		min_per_class=min_per_class,
		remap_labels=remap_labels,
		allow_replacement=allow_replacement,
	)


def sample_datasets_loop(config: SamplingConfig):
	"""Yield variable-size, class-first sampled datasets."""
	sampler = ShardedEmbeddingSampler(
		output_dir=config.output_dir,
		split=config.split,
		seed=config.seed,
		max_cached_shards=config.max_cached_shards,
		sampler_backend=config.sampler_backend,
		enable_hard_sampling=config.enable_hard_sampling,
		num_classes=config.num_classes,
		hard_sampling_temperature=config.hard_sampling_temperature,
		hard_sampling_prob=config.hard_sampling_prob,
		hard_sampling_device=config.hard_sampling_device,
	)

	max_classes_allowed = min(config.max_classes, len(sampler.available_classes))
	if config.min_classes > max_classes_allowed:
		raise ValueError(
			f"min_classes={config.min_classes} exceeds available class count={max_classes_allowed}"
		)

	for i in range(config.n_datasets):
		dataset_size = _sample_from_distribution(
			rng=sampler.rng,
			min_val=config.min_dataset_size,
			max_val=config.max_dataset_size,
			distribution=config.sampling_distribution,
		)
		n_classes = _sample_from_distribution(
			rng=sampler.rng,
			min_val=config.min_classes,
			max_val=max_classes_allowed,
			distribution=config.sampling_distribution,
		)
		X, y, meta = sampler.sample_dataset(
			dataset_size=dataset_size,
			n_classes=n_classes,
			min_per_class=config.min_per_class,
			remap_labels=config.remap_labels,
			allow_replacement=config.allow_replacement,
		)
		yield i, X, y, meta