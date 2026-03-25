"""Reusable components for ImageNet projection experiments."""

from .save_checkpoints import (
	extract_projection_state_dict,
	infer_projection_head_from_state_dict,
	load_projection_head_checkpoint,
	save_projection_checkpoint,
)
from .config import ExtractConfig, SamplingConfig
from .episodes import class_safe_support_query_indices, remap_labels_from_support, sample_support_query_indices
from .extraction import extract_split, load_dinov3, resolve_device
from .projection_methods import build_projection_module, project_episode_features
from .sampling import ShardedEmbeddingSampler, sample_datasets_loop, sample_one_dataset
from .train_loop import train_projection_head_with_sampler

__all__ = [
	"build_projection_module",
	"project_episode_features",
	"ExtractConfig",
	"SamplingConfig",
	"ShardedEmbeddingSampler",
	"class_safe_support_query_indices",
	"extract_split",
	"extract_projection_state_dict",
	"infer_projection_head_from_state_dict",
	"load_dinov3",
	"load_projection_head_checkpoint",
	"remap_labels_from_support",
	"resolve_device",
	"sample_datasets_loop",
	"sample_one_dataset",
	"sample_support_query_indices",
	"save_projection_checkpoint",
	"train_projection_head_with_sampler",
]
