"""Episode split utilities for support/query sampling and label remapping."""

from __future__ import annotations

import numpy as np
import torch


def sample_support_query_indices(
	n_rows: int,
	rng: np.random.Generator,
	frac_min: float,
	frac_max: float,
	min_query_size: int,
) -> tuple[np.ndarray, np.ndarray]:
	if n_rows < 3:
		raise ValueError("Need at least 3 rows to create support/query splits")

	q_frac = float(rng.uniform(frac_min, frac_max))
	q_size = int(round(q_frac * n_rows))
	q_size = max(min_query_size, q_size)
	q_size = min(q_size, n_rows - 1)

	perm = rng.permutation(n_rows)
	query_idx = np.asarray(perm[:q_size], dtype=np.int64)
	support_idx = np.asarray(perm[q_size:], dtype=np.int64)
	if support_idx.size == 0:
		support_idx = query_idx[-1:]
		query_idx = query_idx[:-1]
	return support_idx, query_idx


def class_safe_support_query_indices(
	y: torch.Tensor,
	rng: np.random.Generator,
	frac_min: float,
	frac_max: float,
	min_query_size: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Split rows so every class in query is present in support."""
	y_np = np.asarray(y.detach().cpu().numpy()).reshape(-1)
	n_rows = int(y_np.shape[0])
	if n_rows < 3:
		raise ValueError("Need at least 3 rows to create support/query splits")

	classes, inverse = np.unique(y_np, return_inverse=True)
	if classes.size <= 1:
		return sample_support_query_indices(n_rows, rng, frac_min, frac_max, min_query_size)

	support_keep: list[int] = []
	remaining: list[int] = []
	for cls_id in range(classes.size):
		cls_idx = np.flatnonzero(inverse == cls_id)
		rng.shuffle(cls_idx)
		support_keep.append(int(cls_idx[0]))
		if cls_idx.size > 1:
			remaining.extend(int(i) for i in cls_idx[1:])

	remaining_np = np.asarray(remaining, dtype=np.int64)
	rng.shuffle(remaining_np)

	q_frac = float(rng.uniform(frac_min, frac_max))
	q_size = int(round(q_frac * n_rows))
	max_query = int(remaining_np.size)
	q_size = min(max_query, max(1, q_size))
	if max_query > 0:
		q_size = max(min_query_size, q_size)
		q_size = min(max_query, q_size)

	query_idx = remaining_np[:q_size] if q_size > 0 else np.empty((0,), dtype=np.int64)
	remaining_after_query = remaining_np[q_size:]

	support_idx = np.concatenate(
		[
			np.asarray(support_keep, dtype=np.int64),
			remaining_after_query,
		]
	)
	rng.shuffle(support_idx)

	if query_idx.size == 0:
		if support_idx.size <= classes.size:
			return sample_support_query_indices(n_rows, rng, frac_min, frac_max, min_query_size)
		move_idx = int(support_idx[-1])
		support_idx = support_idx[:-1]
		query_idx = np.asarray([move_idx], dtype=np.int64)

	return support_idx, query_idx


def remap_labels_from_support(
	y_support: torch.Tensor,
	y_query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Map labels to contiguous ids based on support-set classes."""
	support_classes = torch.unique(y_support)
	if support_classes.numel() <= 0:
		raise ValueError("Support labels are empty")

	min_cls = int(torch.min(support_classes).item())
	max_cls = int(torch.max(support_classes).item())
	mapping = torch.full(
		(max_cls - min_cls + 1,),
		-1,
		dtype=torch.long,
		device=y_support.device,
	)
	mapping[support_classes - min_cls] = torch.arange(
		support_classes.numel(), dtype=torch.long, device=y_support.device
	)

	y_support_mapped = mapping[y_support - min_cls]
	y_query_mapped = mapping[y_query - min_cls]
	if torch.any(y_query_mapped < 0):
		raise ValueError(
			"Query contains classes absent from support after split; "
			"cannot compute class-consistent loss"
		)
	return y_support_mapped, y_query_mapped
