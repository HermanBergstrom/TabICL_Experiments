"""Checkpoint helpers for projection experiments.

These helpers unify checkpoint formats between training scripts and evaluation
scripts so new projection methods can reuse the same loading logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from finetune_projection_head import ProjectionHead


def extract_projection_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
	"""Extract projection weights from known checkpoint formats.

	Supported formats:
	- {"state_dict": ...}
	- {"head_state_dict": ...}
	- raw state dict containing model tensors directly
	"""
	state_dict = checkpoint.get("state_dict")
	if state_dict is None:
		state_dict = checkpoint.get("head_state_dict")
	if state_dict is not None:
		return state_dict

	if "net.weight" in checkpoint or "net.0.weight" in checkpoint:
		return checkpoint  # raw state dict

	raise ValueError(
		"Checkpoint missing 'state_dict'/'head_state_dict' and is not a raw state dict"
	)


def infer_projection_head_from_state_dict(
	state_dict: dict[str, torch.Tensor],
) -> tuple[str, int, int, int, float]:
	"""Infer ProjectionHead hyperparameters from saved weights."""
	if "net.weight" in state_dict and "net.bias" in state_dict:
		w = state_dict["net.weight"]
		return "linear", int(w.shape[1]), int(w.shape[0]), 0, 0.0

	if "net.0.weight" in state_dict and "net.3.weight" in state_dict:
		w0 = state_dict["net.0.weight"]
		w3 = state_dict["net.3.weight"]
		return "mlp", int(w0.shape[1]), int(w3.shape[0]), int(w0.shape[0]), 0.0

	raise ValueError(
		"Could not infer projection head architecture from checkpoint state_dict. "
		"Expected either linear (net.weight) or MLP (net.0.weight/net.3.weight)."
	)


def load_projection_head_checkpoint(
	checkpoint_path: Path,
	device: str,
) -> tuple[ProjectionHead, dict[str, Any]]:
	"""Load ProjectionHead from mixed checkpoint formats.

	Returns (model, metadata).
	"""
	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	if not isinstance(checkpoint, dict):
		raise ValueError(f"Checkpoint at {checkpoint_path} must be a dict")

	state_dict = extract_projection_state_dict(checkpoint)

	head_type = checkpoint.get("head_type")
	input_dim = checkpoint.get("input_dim")
	output_dim = checkpoint.get("output_dim")
	hidden_dim = checkpoint.get("head_hidden_dim")
	dropout = checkpoint.get("head_dropout")

	if head_type is None or input_dim is None or output_dim is None:
		head_type, input_dim, output_dim, hidden_dim_inf, dropout_inf = infer_projection_head_from_state_dict(state_dict)
		hidden_dim = hidden_dim if hidden_dim is not None else hidden_dim_inf
		dropout = dropout if dropout is not None else dropout_inf

	head_type = str(head_type)
	input_dim = int(input_dim)
	output_dim = int(output_dim)
	hidden_dim = int(hidden_dim) if hidden_dim is not None else 256
	dropout = float(dropout) if dropout is not None else 0.0

	head = ProjectionHead(
		input_dim=input_dim,
		output_dim=output_dim,
		head_type=head_type,
		hidden_dim=hidden_dim,
		dropout=dropout,
	).to(device)
	head.load_state_dict(state_dict)
	head.eval()

	meta: dict[str, Any] = {
		"head_type": head_type,
		"input_dim": input_dim,
		"output_dim": output_dim,
		"head_hidden_dim": hidden_dim,
		"head_dropout": dropout,
		"checkpoint_path": str(checkpoint_path),
		"checkpoint_dataset": checkpoint.get("dataset"),
		"checkpoint_mimic_task": checkpoint.get("mimic_task"),
		"checkpoint_projection_method": checkpoint.get("projection_method", "projection_head"),
	}
	return head, meta


def save_projection_checkpoint(
	*,
	path: Path,
	projection_module: torch.nn.Module,
	history: dict[str, Any],
	extra: dict[str, Any] | None = None,
) -> None:
	"""Save checkpoint with canonical and backward-compatible weight keys."""
	state_dict_cpu = {k: v.detach().cpu() for k, v in projection_module.state_dict().items()}
	payload: dict[str, Any] = {
		"projection_method": "projection_head",
		"state_dict": state_dict_cpu,
		"head_state_dict": state_dict_cpu,
		"history": history,
	}
	if extra:
		payload.update(extra)
	torch.save(payload, path)
