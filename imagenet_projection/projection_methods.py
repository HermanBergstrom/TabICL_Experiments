"""Projection method builders for ImageNet training loops.

The goal is to keep the main loop agnostic to the projection implementation.
For now, this module exposes a projection-head builder and a stable extension
point for future methods (for example, hypernetwork-generated projections).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from finetune_projection_head import ProjectionHead
from .hypernetworks import SpectralHypernetworkAdapter, VanillaStatsHypernetworkAdapter


def transform_projection_inputs(
	*,
	method: str,
	features: torch.Tensor,
	support_indices: torch.Tensor | None = None,
	zca_epsilon: float = 1e-5,
) -> torch.Tensor:
	"""Apply method-specific preprocessing before the learned projection head.

	For `zca_projection_head`, ZCA whitening is fit on support features only and
	then applied to all features from the same episode.
	"""
	if method == "projection_head":
		return features
	if method == "zca_projection_head":
		if support_indices is None:
			raise ValueError("support_indices are required for zca_projection_head")
		return _zca_whiten_from_support(
			features=features,
			support_indices=support_indices,
			eps=zca_epsilon,
		)

	raise ValueError(
		f"Unsupported projection method: {method}. "
		"Supported methods: ['projection_head', 'zca_projection_head', 'spectral_hypernetwork', 'vanilla_hypernetwork']"
	)


def project_episode_features(
	*,
	method: str,
	module: torch.nn.Module,
	features: torch.Tensor,
	support_indices: torch.Tensor,
	zca_epsilon: float,
	return_projection_matrix: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
	"""Project all rows in an episode for a given projection method."""
	if method in {"projection_head", "zca_projection_head"}:
		X_proj_in = transform_projection_inputs(
			method=method,
			features=features,
			support_indices=support_indices,
			zca_epsilon=zca_epsilon,
		)
		projected = module(X_proj_in)
		if return_projection_matrix:
			return projected, None
		return projected

	if method == "spectral_hypernetwork":
		if not isinstance(module, SpectralHypernetworkAdapter):
			raise TypeError(
				"spectral_hypernetwork requires SpectralHypernetworkAdapter module"
			)
		projected = module(
			features=features,
			support_indices=support_indices,
			return_projection_matrix=return_projection_matrix,
		)
		return projected

	if method == "vanilla_hypernetwork":
		if not isinstance(module, VanillaStatsHypernetworkAdapter):
			raise TypeError(
				"vanilla_hypernetwork requires VanillaStatsHypernetworkAdapter module"
			)
		projected = module(
			features=features,
			support_indices=support_indices,
			return_projection_matrix=return_projection_matrix,
		)
		return projected

	raise ValueError(
		f"Unsupported projection method: {method}. "
		"Supported methods: ['projection_head', 'zca_projection_head', 'spectral_hypernetwork', 'vanilla_hypernetwork']"
	)


def _zca_whiten_from_support(
	*,
	features: torch.Tensor,
	support_indices: torch.Tensor,
	eps: float,
) -> torch.Tensor:
	"""Compute ZCA transform from support rows and apply to all rows."""
	if features.ndim != 2:
		raise ValueError(f"Expected 2D features [N, D], got shape={tuple(features.shape)}")
	if eps <= 0:
		raise ValueError("ZCA epsilon must be > 0")
	if support_indices.ndim != 1:
		raise ValueError("support_indices must be a 1D tensor")

	X_support = features.index_select(0, support_indices)
	n_support = int(X_support.shape[0])
	if n_support == 0:
		raise ValueError("support_indices produced an empty support set")

	support_mean = X_support.mean(dim=0, keepdim=True)
	centered_all = features - support_mean
	centered_support = X_support - support_mean
	if n_support <= 1:
		return centered_all

	cov = (centered_support.T @ centered_support) / float(max(1, n_support - 1))
	# Keep covariance symmetric for stable eigendecomposition under fp noise.
	cov = 0.5 * (cov + cov.T)

	eigvals, eigvecs = torch.linalg.eigh(cov)
	inv_sqrt = torch.rsqrt(torch.clamp(eigvals, min=eps))
	whitening = (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T
	return centered_all @ whitening


def build_projection_module(
	*,
	method: str,
	input_dim: int,
	output_dim: int | None,
	head_type: str,
	hidden_dim: int,
	dropout: float,
	zca_epsilon: float,
	hyper_top_k: int,
	hyper_encoder_type: str,
	hyper_attn_heads: int,
	hyper_attn_layers: int,
	hyper_attn_use_pos_embed: bool,
	use_random_projection_init: bool,
	device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
	"""Build a projection module and return (module, metadata)."""
	if method not in {"projection_head", "zca_projection_head", "spectral_hypernetwork", "vanilla_hypernetwork"}:
		raise ValueError(
			f"Unsupported projection method: {method}. "
			"Supported methods: ['projection_head', 'zca_projection_head', 'spectral_hypernetwork', 'vanilla_hypernetwork']"
		)
	if zca_epsilon <= 0:
		raise ValueError("zca_epsilon must be > 0")
	if hyper_top_k <= 0:
		raise ValueError("hyper_top_k must be > 0")
	if hyper_encoder_type not in {"mlp", "attention"}:
		raise ValueError("hyper_encoder_type must be one of ['mlp', 'attention']")
	if hyper_attn_heads <= 0:
		raise ValueError("hyper_attn_heads must be > 0")
	if hyper_attn_layers <= 0:
		raise ValueError("hyper_attn_layers must be > 0")

	if method == "spectral_hypernetwork":
		actual_output_dim = int(output_dim if output_dim is not None else input_dim)
		module = SpectralHypernetworkAdapter(
			input_dim=int(input_dim),
			output_dim=actual_output_dim,
			top_k_components=int(hyper_top_k),
			context_hidden_dim=int(hidden_dim),
			encoder_type=str(hyper_encoder_type),
			attention_num_heads=int(hyper_attn_heads),
			attention_num_layers=int(hyper_attn_layers),
			attention_use_positional_embeddings=bool(hyper_attn_use_pos_embed),
			use_random_projection_init=bool(use_random_projection_init),
		).to(device)
	elif method == "vanilla_hypernetwork":
		actual_output_dim = int(output_dim if output_dim is not None else input_dim)
		module = VanillaStatsHypernetworkAdapter(
			input_dim=int(input_dim),
			output_dim=actual_output_dim,
			context_hidden_dim=int(hidden_dim),
			use_random_projection_init=bool(use_random_projection_init),
		).to(device)
	else:
		module = ProjectionHead(
			input_dim=input_dim,
			output_dim=output_dim,
			head_type=head_type,
			hidden_dim=hidden_dim,
			dropout=dropout,
		).to(device)

		actual_output_dim = _infer_projection_output_dim(
			module=module,
			input_dim=input_dim,
			requested_output_dim=output_dim,
		)

	meta: dict[str, Any] = {
		"projection_method": method,
		"head_type": head_type,
		"input_dim": int(input_dim),
		"output_dim": actual_output_dim,
		"head_hidden_dim": int(hidden_dim),
		"head_dropout": float(dropout),
		"zca_epsilon": float(zca_epsilon),
		"hyper_top_k": int(hyper_top_k),
		"hyper_encoder_type": str(hyper_encoder_type),
		"hyper_attn_heads": int(hyper_attn_heads),
		"hyper_attn_layers": int(hyper_attn_layers),
		"hyper_attn_use_pos_embed": bool(hyper_attn_use_pos_embed),
		"hyper_use_random_projection_init": bool(use_random_projection_init),
	}
	return module, meta


def _infer_projection_output_dim(
	*,
	module: nn.Module,
	input_dim: int,
	requested_output_dim: int | None,
) -> int:
	"""Infer projection output dim across ProjectionHead implementations."""
	if requested_output_dim is not None:
		return int(requested_output_dim)

	module_out = getattr(module, "output_dim", None)
	if module_out is not None:
		return int(module_out)

	net = getattr(module, "net", None)
	if isinstance(net, nn.Linear):
		return int(net.out_features)
	if isinstance(net, nn.Sequential):
		for layer in reversed(net):
			if isinstance(layer, nn.Linear):
				return int(layer.out_features)

	# ProjectionHead defaults to identity-width when output_dim is not provided.
	return int(input_dim)
