"""Train and evaluate projection-head preprocessing on DVM image embeddings.

Workflow:
1. Load DVM train/val/test image embeddings.
2. Train projection head on the training split only.
3. Fit TabICLClassifier on projected train features and evaluate on projected test.
4. Compare against a PCA baseline with the same feature dimensionality.
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

from experiments import DATASET_CONFIGS, _load_data, _sample_indices
from finetune_projection_head import (
	ProjectionTrainingConfig,
	FrozenTabICLConfig,
	train_projection_head,
)


def _enable_live_output() -> None:
	"""Best-effort line-buffering so logs appear promptly in batch jobs."""
	try:
		sys.stdout.reconfigure(line_buffering=True, write_through=True)
		sys.stderr.reconfigure(line_buffering=True, write_through=True)
	except Exception:
		# Fallback for environments where reconfigure is unavailable.
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


def _project_with_head(head: torch.nn.Module, X: np.ndarray, device: str) -> np.ndarray:
	"""Apply trained projection head to numpy features."""
	head.eval()
	with torch.no_grad():
		x = torch.as_tensor(X, dtype=torch.float32, device=device)
		z = head(x)
	return np.asarray(z.detach().cpu().numpy(), dtype=np.float32)


def _cpu_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
	"""Clone a model state dict onto CPU for portable checkpointing."""
	return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


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


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Projection-head vs PCA baseline on image embeddings + TabICL")
	p.add_argument("--dataset", type=str, choices=["dvm", "mimic"], default="dvm")
	p.add_argument(
		"--mimic-task",
		type=str,
		choices=["mortality", "los_classification", "los_regression", "cxr"],
		default="los_classification",
		help="Task to use when --dataset mimic (default: los_classification)",
	)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
	p.add_argument("--projection-dim", type=int, default=128)
	p.add_argument("--head-type", type=str, choices=["linear", "mlp"], default="mlp")
	p.add_argument("--head-hidden-dim", type=int, default=256)
	p.add_argument("--head-dropout", type=float, default=0.0)
	p.add_argument("--train-steps", type=int, default=500)
	p.add_argument("--learning-rate", type=float, default=1e-3)
	p.add_argument("--weight-decay", type=float, default=0.0)
	p.add_argument("--min-query-frac", type=float, default=0.1)
	p.add_argument("--max-query-frac", type=float, default=0.4)
	p.add_argument("--min-query-size", type=int, default=8)
	p.add_argument(
		"--query-minority-weight",
		type=float,
		default=1.0,
		help=(
			"Weight multiplier (>=1.0) for least-frequent class(es) when sampling "
			"query rows; 1.0 disables reweighting"
		),
	)
	p.add_argument("--max-step-samples", type=int, default=None)
	p.add_argument("--tabicl-eval-estimators", type=int, default=1)
	p.add_argument(
		"--tabicl-train-shuffles",
		"--tabicl-train-estimators",
		dest="tabicl_train_shuffles",
		type=int,
		default=2,
		help=(
			"Number of feature-shuffle patterns used while training the frozen "
			"TabICL backbone (legacy alias: --tabicl-train-estimators)"
		),
	)
	p.add_argument("--tabicl-shuffle", type=str, choices=["none", "random", "shift"], default="random")
	p.add_argument("--model-path", type=Path, default=None,
		help="Optional local TabICL checkpoint path")
	p.add_argument("--max-train-samples", type=int, default=20000,
		help="Cap on train rows (<=0 for all), sampled like experiments.py")
	p.add_argument("--max-val-samples", type=int, default=5000,
		help="Cap on val rows (<=0 for all), used only when validation selection is enabled")
	p.add_argument("--max-test-samples", type=int, default=20000,
		help="Cap on test rows (<=0 for all), sampled like experiments.py")
	p.add_argument(
		"--use-validation-selection",
		action="store_true",
		help=(
			"If set, evaluate projection head every log_every iterations using "
			"train-as-support and val-as-query, and keep the best checkpoint"
		),
	)
	p.add_argument(
		"--save-projection-head-dir",
		type=Path,
		default=None,
		help=(
			"Optional output directory for projection-head checkpoints. "
			"Saves final model by default; with --use-validation-selection, saves "
			"both best and last models."
		),
	)
	p.add_argument("--results-path", type=Path, default=None)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	_enable_live_output()
	device = _resolve_device(args.device)

	ds_cfg = DATASET_CONFIGS[args.dataset]
	_log(f"[info] Loading {args.dataset} image-embedding splits...")
	_data_root, splits, metadata = _load_data(
		dataset=args.dataset,
		data_dir=ds_cfg["data_dir"],
		module_path=ds_cfg["module_path"],
		need_images=True,
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
	_X_tab_val, X_img_val, _X_text_val, _y_val = splits["val"]
	_X_tab_test, X_img_test, _X_text_test, y_test = splits["test"]
	_log("[info] Label balance (full splits):")
	_log(f"  train: {_label_balance_summary(y_train)}")
	_log(f"  val:   {_label_balance_summary(_y_val)}")
	_log(f"  test:  {_label_balance_summary(y_test)}")

	if X_img_train is None or X_img_val is None or X_img_test is None:
		raise ValueError("DVM loader did not provide image embeddings for one or more splits")

	X_train = X_img_train
	X_val = X_img_val
	X_test = X_img_test

	max_train = None if args.max_train_samples is None or args.max_train_samples <= 0 else int(args.max_train_samples)
	max_val = None if args.max_val_samples is None or args.max_val_samples <= 0 else int(args.max_val_samples)
	max_test = None if args.max_test_samples is None or args.max_test_samples <= 0 else int(args.max_test_samples)

	train_idx = _sample_indices(y_train, max_train, seed=args.seed, task=task)
	val_idx = _sample_indices(_y_val, max_val, seed=args.seed + 2, task=task)
	test_idx = _sample_indices(y_test, max_test, seed=args.seed + 1, task=task)
	X_train, y_train = X_train[train_idx], y_train[train_idx]
	X_val, _y_val = X_val[val_idx], _y_val[val_idx]
	X_test, y_test = X_test[test_idx], y_test[test_idx]
	_log("[info] Label balance (sampled splits):")
	_log(f"  train: {_label_balance_summary(y_train)}")
	_log(f"  val:   {_label_balance_summary(_y_val)}")
	_log(f"  test:  {_label_balance_summary(y_test)}")

	requested_dim = int(args.projection_dim)
	pca_max_dim = int(min(X_train.shape[0], X_train.shape[1]))
	effective_dim = min(requested_dim, pca_max_dim)
	if effective_dim <= 0:
		raise ValueError("Invalid projected dimension")
	if effective_dim != requested_dim:
		_log(
			f"[warning] projection_dim capped for PCA compatibility: "
			f"{requested_dim} -> {effective_dim}"
		)

	_log(
		f"[info] Shapes | train={X_train.shape} test={X_test.shape} "
		f"target_dim={effective_dim} device={device}"
	)

	proj_cfg = ProjectionTrainingConfig(
		num_steps=args.train_steps,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		query_fraction_min=args.min_query_frac,
		query_fraction_max=args.max_query_frac,
		min_query_size=args.min_query_size,
		query_minority_weight=args.query_minority_weight,
		max_step_samples=args.max_step_samples,
		seed=args.seed,
	)
	tabicl_cfg = FrozenTabICLConfig(
		n_models=1,
		n_feature_shuffles=args.tabicl_train_shuffles,
		feature_shuffle_method=args.tabicl_shuffle,
		shuffle_seed=args.seed,
		model_path=args.model_path,
	)

	_log("[info] Training projection head...")
	save_requested = args.save_projection_head_dir is not None
	if save_requested:
		head, history, model_states = train_projection_head(
			X_train=X_train,
			y_train=y_train,
			X_val=X_val if args.use_validation_selection else None,
			y_val=_y_val if args.use_validation_selection else None,
			device=device,
			head_type=args.head_type,
			projection_dim=effective_dim,
			hidden_dim=args.head_hidden_dim,
			dropout=args.head_dropout,
			tabicl_config=tabicl_cfg,
			config=proj_cfg,
			verbose=True,
			return_model_states=True,
		)
	else:
		head, history = train_projection_head(
			X_train=X_train,
			y_train=y_train,
			X_val=X_val if args.use_validation_selection else None,
			y_val=_y_val if args.use_validation_selection else None,
			device=device,
			head_type=args.head_type,
			projection_dim=effective_dim,
			hidden_dim=args.head_hidden_dim,
			dropout=args.head_dropout,
			tabicl_config=tabicl_cfg,
			config=proj_cfg,
			verbose=True,
		)

	X_train_proj = _project_with_head(head, X_train, device=device)
	X_test_proj = _project_with_head(head, X_test, device=device)

	_log("[info] Evaluating TabICL on projected features...")
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
	pca = PCA(n_components=effective_dim, random_state=args.seed)
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
	rp = GaussianRandomProjection(n_components=effective_dim, random_state=args.seed)
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

	saved_projection_heads: dict[str, str] = {}
	if save_requested:
		assert args.save_projection_head_dir is not None
		args.save_projection_head_dir.mkdir(parents=True, exist_ok=True)
		checkpoint_base = {
			"dataset": args.dataset,
			"mimic_task": args.mimic_task if args.dataset == "mimic" else None,
			"feature_mode": "image",
			"seed": int(args.seed),
			"input_dim": int(X_train.shape[1]),
			"output_dim": int(effective_dim),
			"head_type": args.head_type,
			"head_hidden_dim": int(args.head_hidden_dim),
			"head_dropout": float(args.head_dropout),
			"use_validation_selection": bool(args.use_validation_selection),
		}

		if args.use_validation_selection:
			best_state = model_states.get("best_state_dict")
			last_state = model_states.get("last_state_dict")

			if best_state is None:
				best_state = _cpu_state_dict(head.state_dict())
			if last_state is None:
				last_state = _cpu_state_dict(head.state_dict())

			best_path = args.save_projection_head_dir / "projection_head_best.pt"
			last_path = args.save_projection_head_dir / "projection_head_last.pt"
			torch.save({**checkpoint_base, "which": "best", "state_dict": best_state}, best_path)
			torch.save({**checkpoint_base, "which": "last", "state_dict": last_state}, last_path)
			saved_projection_heads["best"] = str(best_path)
			saved_projection_heads["last"] = str(last_path)
			_log(f"[info] Saved projection heads: best={best_path} last={last_path}")
		else:
			final_state = model_states.get("last_state_dict")
			if final_state is None:
				final_state = _cpu_state_dict(head.state_dict())
			final_path = args.save_projection_head_dir / "projection_head_final.pt"
			torch.save({**checkpoint_base, "which": "final", "state_dict": final_state}, final_path)
			saved_projection_heads["final"] = str(final_path)
			_log(f"[info] Saved projection head: final={final_path}")

	results = {
		"metadata": {
			"dataset": args.dataset,
			"mimic_task": args.mimic_task if args.dataset == "mimic" else None,
			"feature_mode": "image",
			"task": task,
			"seed": args.seed,
			"device": device,
			"requested_dim": requested_dim,
			"effective_dim": effective_dim,
			"train_shape": list(X_train.shape),
			"test_shape": list(X_test.shape),
			"head_type": args.head_type,
			"head_hidden_dim": args.head_hidden_dim,
			"head_dropout": args.head_dropout,
			"train_steps": args.train_steps,
			"tabicl_train_shuffles": args.tabicl_train_shuffles,
			"tabicl_train_estimators": args.tabicl_train_shuffles,
			"tabicl_eval_estimators": args.tabicl_eval_estimators,
			"tabicl_shuffle": args.tabicl_shuffle,
			"query_minority_weight": float(args.query_minority_weight),
			"use_validation_selection": bool(args.use_validation_selection),
		},
		"projection_head": projection_metrics,
		"pca_baseline": pca_metrics,
		"random_projection_baseline": random_projection_metrics,
		"projection_head_training": {
			"final_loss": float(history["loss"][-1]) if history["loss"] else None,
			"mean_loss": float(np.mean(history["loss"])) if history["loss"] else None,
			"best_val_accuracy": float(np.max(history["val_accuracy"])) if history["val_accuracy"] else None,
			"best_val_loss": float(np.min(history["val_loss"])) if history["val_loss"] else None,
			"num_val_evals": int(len(history["val_step"])),
			"saved_checkpoints": saved_projection_heads,
		},
	}

	_log("\n=== Results ===")
	print(json.dumps(results, indent=2), flush=True)

	if args.results_path is not None:
		args.results_path.parent.mkdir(parents=True, exist_ok=True)
		args.results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
		_log(f"\nSaved results to: {args.results_path}")


if __name__ == "__main__":
	main()
