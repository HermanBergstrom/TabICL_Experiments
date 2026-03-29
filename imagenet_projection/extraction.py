"""ImageNet embedding extraction and shard persistence utilities."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from .config import ExtractConfig


class IndexedImageFolder(Dataset[dict[str, Any]]):
	"""ImageFolder wrapper that returns index and file path metadata."""

	def __init__(self, root: Path, transform: transforms.Compose):
		self.base = datasets.ImageFolder(str(root), transform=transform)

	def __len__(self) -> int:
		return len(self.base)

	def __getitem__(self, idx: int) -> dict[str, Any] | None:
		path, _class_id = self.base.samples[idx]
		try:
			image, target = self.base[idx]
		except Exception as e:
			print(f"\n[warning] Skipping corrupted/unreadable image {path}: {e}")
			return None

		return {
			"image": image,
			"target": int(target),
			"index": int(idx),
			"path": str(path),
		}

	@property
	def class_to_idx(self) -> dict[str, int]:
		return dict(self.base.class_to_idx)


def collate_indexed(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
	batch = [item for item in batch if item is not None]
	if not batch:
		return None
	images = torch.stack([item["image"] for item in batch], dim=0)
	targets = torch.tensor([item["target"] for item in batch], dtype=torch.long)
	indices = torch.tensor([item["index"] for item in batch], dtype=torch.long)
	paths = [item["path"] for item in batch]
	return {
		"images": images,
		"targets": targets,
		"indices": indices,
		"paths": paths,
	}


def resolve_device(device: str) -> torch.device:
	if device == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device == "cuda" and torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def output_dtype(dtype_name: str) -> torch.dtype:
	if dtype_name == "float16":
		return torch.float16
	if dtype_name == "bfloat16":
		return torch.bfloat16
	if dtype_name == "float32":
		return torch.float32
	raise ValueError("dtype must be one of: float16, bfloat16, float32")


def autocast_context(device: torch.device, amp_dtype: str):
	if device.type != "cuda":
		return nullcontext()
	if amp_dtype == "float16":
		return torch.autocast("cuda", dtype=torch.float16)
	if amp_dtype == "bfloat16":
		return torch.autocast("cuda", dtype=torch.bfloat16)
	raise ValueError("amp-dtype must be one of: float16, bfloat16")


def make_transform(resize_size: int, img_size: int) -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.Resize(resize_size),
			transforms.CenterCrop(img_size),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=(0.485, 0.456, 0.406),
				std=(0.229, 0.224, 0.225),
			),
		]
	)


def load_dinov3(repo_dir: Path, weights_path: Path, device: torch.device) -> torch.nn.Module:
	print(f"[info] Loading DINOv3 from repo={repo_dir} weights={weights_path}")
	model = torch.hub.load(
		str(repo_dir),
		"dinov3_vitb16",
		source="local",
		weights=str(weights_path),
	)
	model.eval()
	model.to(device)
	return model


def extract_model_features(model_output: Any) -> torch.Tensor:
	"""Normalize DINO output variants into a [B, D] tensor."""
	if isinstance(model_output, torch.Tensor):
		feats = model_output
	elif isinstance(model_output, (list, tuple)) and len(model_output) > 0:
		first = model_output[0]
		if not isinstance(first, torch.Tensor):
			raise TypeError("Unsupported non-tensor first element in model output")
		feats = first
	elif isinstance(model_output, dict):
		candidate = model_output.get("x_norm_clstoken", None)
		if candidate is None:
			candidate = model_output.get("features", None)
		if not isinstance(candidate, torch.Tensor):
			raise TypeError("Could not extract tensor features from model output dict")
		feats = candidate
	else:
		raise TypeError(f"Unsupported model output type: {type(model_output)}")

	if feats.ndim == 3:
		# Common transformer shape [B, T, D]; take CLS token.
		feats = feats[:, 0, :]
	if feats.ndim != 2:
		raise ValueError(f"Expected 2D features after normalization, got shape {tuple(feats.shape)}")
	return feats


def _manifest_path(output_dir: Path, split: str) -> Path:
	return output_dir / f"{split}_manifest.json"


def _shard_path(output_dir: Path, split: str, shard_id: int) -> Path:
	return output_dir / f"{split}_shard_{shard_id:06d}.pt"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
	tmp = path.with_suffix(path.suffix + ".tmp")
	with tmp.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)
	tmp.replace(path)


def _init_manifest(
	*,
	output_dir: Path,
	split: str,
	class_to_idx: dict[str, int],
	total_dataset_size: int,
	feature_dtype: str,
	resume: bool,
) -> dict[str, Any]:
	path = _manifest_path(output_dir, split)
	if resume and path.exists():
		with path.open("r", encoding="utf-8") as f:
			manifest = json.load(f)
		if manifest.get("split") != split:
			raise ValueError(f"Existing manifest split mismatch in {path}")
		return manifest

	manifest = {
		"split": split,
		"total_dataset_size": int(total_dataset_size),
		"processed_samples": 0,
		"next_shard_id": 0,
		"feature_dtype": feature_dtype,
		"class_to_idx": class_to_idx,
		"shards": [],
	}
	_write_json(path, manifest)
	return manifest


def _flush_shard(
	*,
	output_dir: Path,
	split: str,
	manifest: dict[str, Any],
	features: torch.Tensor,
	targets: torch.Tensor,
	indices: torch.Tensor,
	paths: list[str],
) -> None:
	if features.shape[0] <= 0:
		return

	shard_id = int(manifest["next_shard_id"])
	shard_file = _shard_path(output_dir, split, shard_id)
	if shard_file.exists():
		raise FileExistsError(
			f"Refusing to overwrite existing shard: {shard_file}. "
			"Use --resume or a different output directory."
		)

	payload = {
		"split": split,
		"shard_id": shard_id,
		"num_samples": int(features.shape[0]),
		"features": features,
		"targets": targets,
		"indices": indices,
		"paths": paths,
	}
	torch.save(payload, shard_file)

	start_idx = int(indices.min().item()) if indices.numel() > 0 else -1
	end_idx = int(indices.max().item()) if indices.numel() > 0 else -1
	manifest["processed_samples"] = max(int(manifest["processed_samples"]), end_idx + 1)
	manifest["next_shard_id"] = shard_id + 1
	manifest["shards"].append(
		{
			"shard_id": shard_id,
			"file": shard_file.name,
			"num_samples": int(features.shape[0]),
			"min_index": start_idx,
			"max_index": end_idx,
		}
	)

	_write_json(_manifest_path(output_dir, split), manifest)


def extract_split(config: ExtractConfig, split: str, model: torch.nn.Module, device: torch.device) -> None:
	if split == "all":
		split_dir = config.data_dir
	else:
		split_dir = config.data_dir / split
	if not split_dir.exists():
		raise FileNotFoundError(f"ImageNet split directory not found: {split_dir}")

	print(f"[info] Preparing split={split} from {split_dir}")
	transform = make_transform(config.resize_size, config.img_size)
	full_dataset = IndexedImageFolder(root=split_dir, transform=transform)

	manifest = _init_manifest(
		output_dir=config.output_dir,
		split=split,
		class_to_idx=full_dataset.class_to_idx,
		total_dataset_size=len(full_dataset),
		feature_dtype=config.dtype,
		resume=config.resume,
	)

	processed = int(manifest.get("processed_samples", 0))
	if processed < 0 or processed > len(full_dataset):
		raise ValueError(
			f"Manifest processed_samples={processed} is invalid for split size={len(full_dataset)}"
		)
	if processed == len(full_dataset):
		print(f"[info] split={split} already complete ({processed} samples). Skipping.")
		return

	if processed > 0:
		print(f"[info] split={split} resuming from sample index {processed}")
		dataset: Dataset[dict[str, Any]] = Subset(full_dataset, range(processed, len(full_dataset)))
	else:
		dataset = full_dataset

	loader = DataLoader(
		dataset,
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=config.num_workers,
		pin_memory=config.pin_memory and device.type == "cuda",
		collate_fn=collate_indexed,
	)

	out_dtype = output_dtype(config.dtype)
	autocast_ctx = autocast_context(device, config.amp_dtype)

	buf_feats: list[torch.Tensor] = []
	buf_targets: list[torch.Tensor] = []
	buf_indices: list[torch.Tensor] = []
	buf_paths: list[str] = []
	buf_count = 0

	def flush_ready_shards(force: bool = False) -> None:
		nonlocal buf_feats, buf_targets, buf_indices, buf_paths, buf_count
		if buf_count == 0:
			return
		if not force and buf_count < config.shard_size:
			return

		feats_all = torch.cat(buf_feats, dim=0)
		targets_all = torch.cat(buf_targets, dim=0)
		indices_all = torch.cat(buf_indices, dim=0)
		paths_all = list(buf_paths)

		while feats_all.shape[0] >= config.shard_size:
			n = config.shard_size
			_flush_shard(
				output_dir=config.output_dir,
				split=split,
				manifest=manifest,
				features=feats_all[:n].contiguous(),
				targets=targets_all[:n].contiguous(),
				indices=indices_all[:n].contiguous(),
				paths=paths_all[:n],
			)
			feats_all = feats_all[n:]
			targets_all = targets_all[n:]
			indices_all = indices_all[n:]
			paths_all = paths_all[n:]

		if force and feats_all.shape[0] > 0:
			_flush_shard(
				output_dir=config.output_dir,
				split=split,
				manifest=manifest,
				features=feats_all.contiguous(),
				targets=targets_all.contiguous(),
				indices=indices_all.contiguous(),
				paths=paths_all,
			)
			feats_all = feats_all[:0]
			targets_all = targets_all[:0]
			indices_all = indices_all[:0]
			paths_all = []

		buf_feats = [feats_all] if feats_all.shape[0] > 0 else []
		buf_targets = [targets_all] if targets_all.shape[0] > 0 else []
		buf_indices = [indices_all] if indices_all.shape[0] > 0 else []
		buf_paths = paths_all
		buf_count = int(feats_all.shape[0])

	print(
		f"[info] Extracting split={split} | remaining={len(dataset)} "
		f"batch_size={config.batch_size} shard_size={config.shard_size}"
	)

	with torch.inference_mode():
		for batch in tqdm(loader, desc=f"split={split}"):
			if batch is None:
				continue
			images = batch["images"].to(device, non_blocking=True)
			targets = batch["targets"].cpu().to(torch.long)
			indices = batch["indices"].cpu().to(torch.long)
			paths = batch["paths"]

			with autocast_ctx:
				model_output = model(images)
				feats = extract_model_features(model_output)

			feats = feats.detach().to(dtype=out_dtype).cpu()
			buf_feats.append(feats)
			buf_targets.append(targets)
			buf_indices.append(indices)
			buf_paths.extend(paths)
			buf_count += int(feats.shape[0])

			if buf_count >= config.shard_size:
				flush_ready_shards(force=False)

	flush_ready_shards(force=True)

	manifest["processed_samples"] = len(full_dataset)
	_write_json(_manifest_path(config.output_dir, split), manifest)

	print(
		f"[done] split={split} processed={manifest['processed_samples']} "
		f"shards={len(manifest['shards'])}"
	)
