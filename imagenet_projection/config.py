"""Configuration dataclasses for ImageNet projection workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractConfig:
	"""CLI configuration for ImageNet embedding extraction."""

	data_dir: Path
	output_dir: Path
	splits: list[str]
	batch_size: int
	num_workers: int
	img_size: int
	resize_size: int
	shard_size: int
	repo_dir: Path
	weights_path: Path
	device: str
	dtype: str
	amp_dtype: str
	resume: bool
	pin_memory: bool


@dataclass
class SamplingConfig:
	"""Configuration for class-first dataset sampling from saved shards."""

	output_dir: Path
	split: str
	n_datasets: int
	min_dataset_size: int
	max_dataset_size: int
	min_classes: int
	max_classes: int
	min_per_class: int
	seed: int
	remap_labels: bool
	allow_replacement: bool
	max_cached_shards: int
	sampler_backend: str
	sampling_distribution: str = "log-uniform"
	enable_hard_sampling: bool = False
	num_classes: int = 1000
	hard_sampling_temperature: float = 0.1
	hard_sampling_prob: float = 0.5
	hard_sampling_device: str = "cpu"
