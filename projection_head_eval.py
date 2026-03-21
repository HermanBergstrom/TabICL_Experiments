"""Evaluate a pretrained projection head on a target dataset with TabICL.

This script is intended for transfer/generalization checks: a projection head
trained on one dataset can be evaluated on another dataset without retraining.
It compares three feature spaces on the target dataset:
1. Pretrained projection head features
2. PCA baseline
3. Gaussian random projection baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.random_projection import GaussianRandomProjection
from tabicl import TabICLClassifier

from experiments import DATASET_CONFIGS, DATASET_NAMES, _load_data, _sample_indices
from finetune_projection_head import ProjectionHead
from imagenet_projection.checkpoints import load_projection_head_checkpoint

try:
	from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional dependency at import time
	hf_hub_download = None


def _enable_live_output() -> None:
	"""Best-effort line-buffering so logs appear promptly in batch jobs."""
	try:
		sys.stdout.reconfigure(line_buffering=True, write_through=True)
		sys.stderr.reconfigure(line_buffering=True, write_through=True)
	except Exception:
		pass


def _log(msg: str) -> None:
	print(msg, flush=True)


def _resolve_device(device: str) -> str:
	if device == "cuda" and torch.cuda.is_available():
		return "cuda"
	if device == "auto":
		return "cuda" if torch.cuda.is_available() else "cpu"
	return "cpu"


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float | None]:
	"""Compute accuracy/F1/AUROC for binary or multiclass classification."""
	auroc = None
	try:
		if y_proba.ndim == 2 and y_proba.shape[1] == 2:
			auroc = float(roc_auc_score(y_true, y_proba[:, 1]))
		elif y_proba.ndim == 1:
			auroc = float(roc_auc_score(y_true, y_proba))
		else:
			auroc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
	except Exception:
		auroc = None

	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
		"auroc": auroc,
	}


def _fit_eval_tabicl(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	*,
	n_estimators: int,
	seed: int,
	device: str,
) -> dict[str, float | None]:
	"""Fit TabICLClassifier and evaluate on test split."""
	clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
	try:
		clf.set_params(device=device)
	except Exception:
		pass
	clf.fit(X_train, y_train)
	y_pred = np.asarray(clf.predict(X_test)).reshape(-1)
	y_proba = np.asarray(clf.predict_proba(X_test))
	return _classification_metrics(y_test, y_pred, y_proba)


def _label_balance_summary(y: np.ndarray) -> str:
	"""Format class counts and percentages for quick balance checks."""
	y_arr = np.asarray(y).reshape(-1)
	classes, counts = np.unique(y_arr, return_counts=True)
	total = int(y_arr.shape[0])
	parts = [f"n={total}"]
	for cls, cnt in zip(classes.tolist(), counts.tolist()):
		pct = 100.0 * float(cnt) / float(max(1, total))
		parts.append(f"{cls}:{cnt} ({pct:.1f}%)")
	return " | ".join(parts)


def _resolve_head_checkpoint(
	head_path: str | Path,
	*,
	hf_repo_id: str | None,
	hf_filename: str | None,
) -> Path:
	"""Resolve head checkpoint locally or download from Hugging Face when requested."""
	p = Path(head_path)
	if p.exists():
		return p

	if hf_repo_id is None:
		raise FileNotFoundError(
			f"Projection-head checkpoint not found at {p}. "
			"Provide a valid local path or --hf-repo-id/--hf-filename for download."
		)
	if hf_hub_download is None:
		raise ImportError("huggingface_hub is required for --hf-repo-id download")

	filename = hf_filename if hf_filename is not None else p.name
	cache_path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
	return Path(cache_path)


def _load_projection_head(checkpoint_path: Path, device: str) -> tuple[ProjectionHead, dict]:
	"""Load projection head checkpoint using shared checkpoint utilities."""
	return load_projection_head_checkpoint(checkpoint_path, device)


def _project_with_head(head: torch.nn.Module, X: np.ndarray, device: str) -> np.ndarray:
	"""Apply a pretrained projection head to numpy features."""
	head.eval()
	with torch.no_grad():
		x = torch.as_tensor(X, dtype=torch.float32, device=device)
		z = head(x)
	return np.asarray(z.detach().cpu().numpy(), dtype=np.float32)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Evaluate a pretrained projection head on a target dataset")
	p.add_argument("--dataset", type=str, choices=DATASET_NAMES, default="dvm")
	p.add_argument(
		"--mimic-task",
		type=str,
		choices=["mortality", "los_classification", "los_regression", "cxr"],
		default="los_classification",
		help="Task to use when --dataset mimic (default: los_classification)",
	)
	p.add_argument(
		"--feature-source",
		type=str,
		choices=["dinov3", "vertexai", "dinov3_text"],
		default="dinov3",
		help=(
			"Embedding source for loaders that support it (notably petfinder)."
		),
	)
	p.add_argument(
		"--use-text",
		action="store_true",
		help="Request text embeddings from loaders that support them (forwarded to experiments._load_data)",
	)
	p.add_argument("--head-path", type=Path, required=True, help="Path to pretrained projection head checkpoint")
	p.add_argument("--hf-repo-id", type=str, default=None,
		help="Optional Hugging Face repo id to download --head-path/--hf-filename when local file is absent")
	p.add_argument("--hf-filename", type=str, default=None,
		help="Optional Hugging Face filename override (defaults to basename of --head-path)")
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
	p.add_argument("--tabicl-eval-estimators", type=int, default=1)
	p.add_argument("--max-train-samples", type=int, default=20000,
		help="Cap on train rows (<=0 for all), sampled like experiments.py")
	p.add_argument("--max-test-samples", type=int, default=20000,
		help="Cap on test rows (<=0 for all), sampled like experiments.py")
	p.add_argument("--results-path", type=Path, default=None)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	_enable_live_output()
	device = _resolve_device(args.device)

	ckpt_path = _resolve_head_checkpoint(
		head_path=args.head_path,
		hf_repo_id=args.hf_repo_id,
		hf_filename=args.hf_filename,
	)
	head, head_meta = _load_projection_head(ckpt_path, device=device)

	ds_cfg = DATASET_CONFIGS[args.dataset]
	_log(f"[info] Loading {args.dataset} image-embedding splits...")
	_data_root, splits, metadata = _load_data(
		dataset=args.dataset,
		data_dir=ds_cfg["data_dir"],
		module_path=ds_cfg["module_path"],
		need_images=True,
		feature_source=args.feature_source,
		use_text=args.use_text,
		mimic_task=args.mimic_task,
	)
	task = str(metadata.get("task", "classification")).lower()
	if task != "classification":
		raise ValueError(
			"This script expects classification data. "
			f"Got task={task}. If using --dataset mimic, set --mimic-task to a classification task "
			"(e.g. los_classification, mortality, or cxr)."
		)

	_X_tab_train, X_img_train, _X_text_train, y_train = splits["train"]
	_X_tab_val, X_img_val, _X_text_val, y_val = splits["val"]
	_X_tab_test, X_img_test, _X_text_test, y_test = splits["test"]

	if X_img_train is None or X_img_val is None or X_img_test is None:
		raise ValueError("Dataset loader did not provide image embeddings for one or more splits")

	_log("[info] Label balance (full splits):")
	_log(f"  train: {_label_balance_summary(y_train)}")
	_log(f"  val:   {_label_balance_summary(y_val)}")
	_log(f"  test:  {_label_balance_summary(y_test)}")

	X_train = X_img_train
	X_test = X_img_test

	max_train = None if args.max_train_samples is None or args.max_train_samples <= 0 else int(args.max_train_samples)
	max_test = None if args.max_test_samples is None or args.max_test_samples <= 0 else int(args.max_test_samples)

	train_idx = _sample_indices(y_train, max_train, seed=args.seed, task=task)
	test_idx = _sample_indices(y_test, max_test, seed=args.seed + 1, task=task)
	X_train, y_train = X_train[train_idx], y_train[train_idx]
	X_test, y_test = X_test[test_idx], y_test[test_idx]

	_log("[info] Label balance (sampled splits):")
	_log(f"  train: {_label_balance_summary(y_train)}")
	_log(f"  test:  {_label_balance_summary(y_test)}")

	if X_train.shape[1] != int(head_meta["input_dim"]):
		raise ValueError(
			"Projection head input dim does not match target dataset image embedding dim: "
			f"head_input_dim={head_meta['input_dim']} vs dataset_dim={X_train.shape[1]}. "
			"Use a checkpoint trained on the same embedding dimensionality."
		)

	head_dim = int(head_meta["output_dim"])
	pca_max_dim = int(min(X_train.shape[0], X_train.shape[1]))
	pca_dim = min(head_dim, pca_max_dim)
	if pca_dim != head_dim:
		_log(f"[warning] PCA dim capped for compatibility: {head_dim} -> {pca_dim}")

	_log(
		f"[info] Evaluating transfer head from checkpoint={ckpt_path} | "
		f"head_type={head_meta['head_type']} input_dim={head_meta['input_dim']} output_dim={head_dim}"
	)

	X_train_proj = _project_with_head(head, X_train, device=device)
	X_test_proj = _project_with_head(head, X_test, device=device)

	_log("[info] Evaluating TabICL on projection-head features...")
	projection_metrics = _fit_eval_tabicl(
		X_train=X_train_proj,
		y_train=y_train,
		X_test=X_test_proj,
		y_test=y_test,
		n_estimators=args.tabicl_eval_estimators,
		seed=args.seed,
		device=device,
	)

	_log("[info] Evaluating PCA baseline...")
	pca = PCA(n_components=pca_dim, random_state=args.seed)
	X_train_pca = np.asarray(pca.fit_transform(X_train), dtype=np.float32)
	X_test_pca = np.asarray(pca.transform(X_test), dtype=np.float32)
	pca_metrics = _fit_eval_tabicl(
		X_train=X_train_pca,
		y_train=y_train,
		X_test=X_test_pca,
		y_test=y_test,
		n_estimators=args.tabicl_eval_estimators,
		seed=args.seed,
		device=device,
	)

	_log("[info] Evaluating random projection baseline...")
	rp = GaussianRandomProjection(n_components=head_dim, random_state=args.seed)
	X_train_rp = np.asarray(rp.fit_transform(X_train), dtype=np.float32)
	X_test_rp = np.asarray(rp.transform(X_test), dtype=np.float32)
	random_projection_metrics = _fit_eval_tabicl(
		X_train=X_train_rp,
		y_train=y_train,
		X_test=X_test_rp,
		y_test=y_test,
		n_estimators=args.tabicl_eval_estimators,
		seed=args.seed,
		device=device,
	)

	results = {
		"metadata": {
			"dataset": args.dataset,
			"mimic_task": args.mimic_task if args.dataset == "mimic" else None,
			"feature_source": args.feature_source,
			"use_text": bool(args.use_text),
			"feature_mode": "image",
			"task": task,
			"seed": args.seed,
			"device": device,
			"train_shape": list(X_train.shape),
			"test_shape": list(X_test.shape),
			"tabicl_eval_estimators": args.tabicl_eval_estimators,
			"projection_dim": head_dim,
			"pca_dim": pca_dim,
			"checkpoint": head_meta,
		},
		"projection_head": projection_metrics,
		"pca_baseline": pca_metrics,
		"random_projection_baseline": random_projection_metrics,
	}

	_log("\n=== Results ===")
	print(json.dumps(results, indent=2), flush=True)

	if args.results_path is not None:
		args.results_path.parent.mkdir(parents=True, exist_ok=True)
		args.results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
		_log(f"\nSaved results to: {args.results_path}")


if __name__ == "__main__":
	main()
