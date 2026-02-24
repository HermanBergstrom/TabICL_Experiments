"""Multi-dataset experiment runner.

Benchmarks classification models on tabular + image-embedding datasets (DVM cars,
PetFinder adoption, etc.) using tabular features, DINOv2/v3 image embeddings, or
their concatenation.  Supports optional dimensionality reduction (PCA / ICA /
random projection) on image features and a "probing mode" that evaluates
LinearProbe / MLP heads on pre-extracted TabICL representations.

See DVM_EXPERIMENTS.md for a full walkthrough.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.random_projection import GaussianRandomProjection
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from tabicl import TabICLClassifier, TabICLRegressor
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor

from baseline_heads import (
	LinearProbe,
	MLPHead,
	LinearRegressionHead,
	MLPRegressionHead,
	predict_head,
	predict_head_regression,
	train_head,
	train_head_regression,
)

# ---------------------------------------------------------------------------
# Dataset registry & constants
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, dict] = {
	"dvm": {
		"module_path": Path(
			"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset/dvm_dataset_with_dinov2.py"
		),
		"loader_fn": "load_dvm_dataset",
		"data_dir": Path(
			"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset"
		),
		"tabicl_features_dir": Path(
			"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset/tabiclv2_features"
		),
	},
	"petfinder": {
		"module_path": Path(
			"/home/hermanb/projects/aip-rahulgk/image_icl_project/petfinder/petfinder_dataset_with_dinov3.py"
		),
		"loader_fn": "load_petfinder_dataset",
		"data_dir": Path(
			"/home/hermanb/projects/aip-rahulgk/image_icl_project/petfinder"
		),
		"tabicl_features_dir": Path(
			"/home/hermanb/projects/aip-rahulgk/image_icl_project/petfinder/tabiclv2_features"
		),
	},
	"paintings": {
		"module_path": Path(
			"/home/hermanb/projects/aip-rahulgk/image_icl_project/paintings/paintings_dataset_with_dinov3.py"
		),
		"loader_fn": "load_paintings_dataset",
		"data_dir": Path(
			"/home/hermanb/projects/aip-rahulgk/image_icl_project/paintings"
		),
		"tabicl_features_dir": Path(
			"/home/hermanb/projects/aip-rahulgk/image_icl_project/paintings/tabiclv2_features"
		),
	},
}

DATASET_NAMES = list(DATASET_CONFIGS.keys())

STANDARD_METHODS = ["tabicl", "decision_tree", "random_forest", "xgboost"]
PROBING_METHODS = ["linear_probe", "mlp"]
ALL_METHODS = STANDARD_METHODS + PROBING_METHODS


# ===================================================================
# 1. Data loading
# ===================================================================


def _import_dataset_loader(module_path: Path, loader_fn: str):
	"""Dynamically import *loader_fn* from the given module path."""
	if not module_path.exists():
		raise FileNotFoundError(f"Dataset module not found: {module_path}")

	spec = importlib.util.spec_from_file_location("dataset_module", str(module_path))
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module spec from {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	if not hasattr(module, loader_fn):
		raise AttributeError(
			f"Module {module_path} does not define {loader_fn}"
		)

	return getattr(module, loader_fn)


def _extract_modalities_from_dataset(dataset, task: str):
	"""Walk *dataset* and return ``(X_tab, X_img | None, y)`` as numpy arrays."""
	tabular_features, image_features, targets = [], [], []

	for index in tqdm(range(len(dataset))):
		item = dataset[index]
		tabular = item.get("tabular")
		if tabular is None:
			raise ValueError("Dataset item is missing tabular features")

		tabular_features.append(np.asarray(tabular, dtype=np.float32))
		emb = item.get("image_embedding")
		image_features.append(None if emb is None else np.asarray(emb, dtype=np.float32))
		if task == "regression":
			targets.append(float(item["target"]))
		else:
			targets.append(int(item["target"]))

	X_tab = np.stack(tabular_features)
	X_img = None if any(f is None for f in image_features) else np.stack(image_features)
	y_dtype = np.float32 if task == "regression" else np.int64
	y = np.asarray(targets, dtype=y_dtype)
	return X_tab, X_img, y


def _load_data(
	dataset: str, data_dir: Path, module_path: Path, need_images: bool,
):
	"""Load a dataset and extract numpy arrays for each split.

	Dispatches to the correct loader based on *dataset* (e.g. ``'dvm'``,
	``'petfinder'``).  Returns ``(data_root, splits)`` where *splits* maps
	split name to ``(X_tab, X_img | None, y)``.
	"""
	cfg = DATASET_CONFIGS[dataset]
	load_fn = _import_dataset_loader(module_path, cfg["loader_fn"])

	# Each loader has a slightly different call signature.
	if dataset == "dvm":
		data_root = data_dir.parent if data_dir.name == "preprocessed_csvs" else data_dir
		train_loader, val_loader, test_loader, _metadata = load_fn(
			data_dir=str(data_root), batch_size=2048, num_workers=0,
			use_images=need_images,
		)
	elif dataset == "petfinder":
		data_root = data_dir
		train_loader, val_loader, test_loader, _metadata = load_fn(
			processed_dir=data_dir / "preprocessed",
			pt_path=data_dir / "petfinder_dinov3_features.pt",
			batch_size=2048, num_workers=0, use_images=need_images,
		)
	elif dataset == "paintings":
		data_root = data_dir
		train_loader, val_loader, test_loader, _metadata = load_fn(
			processed_dir=data_dir / "preprocessed",
			pt_path=data_dir / "paintings_dinov3_features.pt",
			batch_size=2048, num_workers=0, use_images=need_images,
		)
	else:
		raise ValueError(f"Unknown dataset: {dataset}")

	task = str(_metadata.get("task", "classification")).lower()

	print(f"Extracting full train / val / test features from {dataset} dataset (task={task})...")
	splits = {
		name: _extract_modalities_from_dataset(loader.dataset, task=task)
		for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]
	}
	return data_root, splits, _metadata


# ===================================================================
# 2. Sampling helpers
# ===================================================================


def _stratified_indices(y: np.ndarray, n: int | None, seed: int) -> np.ndarray:
	"""Return stratified sample indices (or all indices when *n* is None / large enough)."""
	if n is None or len(y) <= n:
		return np.arange(len(y), dtype=np.int64)
	if n < len(np.unique(y)):
		raise ValueError(
			f"Requested n={n} is smaller than the number of classes ({len(np.unique(y))})."
		)
	idx, _ = next(
		StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
		.split(np.zeros((len(y), 1)), y)
	)
	return idx


def _sample_indices(y: np.ndarray, n: int | None, seed: int, task: str) -> np.ndarray:
	"""Sample indices for classification (stratified) or regression (uniform)."""
	if task == "classification":
		return _stratified_indices(y, n, seed)
	if n is None or len(y) <= n:
		return np.arange(len(y), dtype=np.int64)
	rng = np.random.default_rng(seed)
	return np.asarray(rng.choice(len(y), size=n, replace=False), dtype=np.int64)


def _maybe_index(arr: np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
	"""Index into *arr* if it is not ``None``."""
	return None if arr is None else arr[idx]


# ===================================================================
# 3. TabICL pre-extracted representation loading
# ===================================================================


def _load_representations(features_dir: Path, split: str, n_rows: int) -> np.ndarray:
	"""Load ``{split}_representations.pt`` and mean-pool across estimators.

	Returns ``np.ndarray`` of shape ``(n_rows, embed_dim)``.
	"""
	pt_path = features_dir / f"{split}_representations.pt"
	if not pt_path.exists():
		raise FileNotFoundError(
			f"Pre-extracted TabICL features not found at {pt_path}. "
			"Run extract_tabicl_features.py first."
		)
	rep_dict: dict[int, torch.Tensor] = torch.load(
		pt_path, map_location="cpu", weights_only=True
	)
	rows = []
	for idx in range(n_rows):
		if idx not in rep_dict:
			raise KeyError(
				f"Row index {idx} missing from {pt_path}. "
				"Re-run extract_tabicl_features.py to regenerate."
			)
		rows.append(rep_dict[idx].mean(dim=0).numpy())
	return np.stack(rows).astype(np.float32)


def _load_all_representations(
	features_dir: Path, n_train: int, n_val: int, n_test: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Load representations for train / val / test splits."""
	reps = tuple(
		_load_representations(features_dir, s, n)
		for s, n in [("train", n_train), ("val", n_val), ("test", n_test)]
	)
	print(
		f"[info] Loaded TabICL representations  "
		f"train={reps[0].shape}  val={reps[1].shape}  test={reps[2].shape}"
	)
	return reps  # type: ignore[return-value]


def _extract_representations_via_subprocess(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_val: np.ndarray,
	X_test: np.ndarray,
	n_estimators: int,
	seed: int,
	n_folds: int = 10,
	chunk_size: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Extract probing representations by calling ``extract_tabicl_features.py``.

	This enforces fair probing for each train subsample: train representations are
	computed with CV on the *current* subsampled train split, and val/test
	representations are computed from a model fitted on that same subsample.
	"""
	script_path = Path(__file__).with_name("extract_tabicl_features.py")
	if not script_path.exists():
		raise FileNotFoundError(f"Required script not found: {script_path}")

	with tempfile.TemporaryDirectory(prefix="tabicl_probe_reps_") as tmpdir:
		tmp = Path(tmpdir)
		arrays_path = tmp / "arrays_input.npz"
		reps_path = tmp / "reps_output.npz"

		np.savez_compressed(
			arrays_path,
			X_train=np.asarray(X_train, dtype=np.float32),
			y_train=np.asarray(y_train, dtype=np.int64),
			X_val=np.asarray(X_val, dtype=np.float32),
			X_test=np.asarray(X_test, dtype=np.float32),
		)

		cmd = [
			sys.executable,
			str(script_path),
			"--from-arrays-npz", str(arrays_path),
			"--save-arrays-npz", str(reps_path),
			"--n-estimators", str(n_estimators),
			"--n-folds", str(n_folds),
			"--chunk-size", str(chunk_size),
			"--seed", str(seed),
		]
		proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
		if proc.returncode != 0:
			raise RuntimeError(
				"On-the-fly probing representation extraction failed.\n"
				f"Command: {' '.join(cmd)}\n"
				f"stdout:\n{proc.stdout}\n"
				f"stderr:\n{proc.stderr}"
			)

		reps = np.load(reps_path)
		return (
			np.asarray(reps["train_representations"], dtype=np.float32).copy(),
			np.asarray(reps["val_representations"], dtype=np.float32).copy(),
			np.asarray(reps["test_representations"], dtype=np.float32).copy(),
		)


# ===================================================================
# 4. Image dimensionality reduction
# ===================================================================


def _fit_image_reducer(X_train: np.ndarray, name: str, dim: int, seed: int):
	"""Return a fitted reducer or ``None`` when *name* is ``'none'``."""
	if name == "none":
		return None
	if dim <= 0:
		raise ValueError("--image-reducer-dim must be > 0 when reducer is enabled")

	# PCA and ICA share the same component-capping logic.
	if name in ("pca", "ica"):
		max_dim = min(X_train.shape[0], X_train.shape[1])
		effective = min(dim, max_dim)
		if effective != dim:
			print(f"[warning] Capping {name.upper()} components: {dim} -> {effective}")
		cls = PCA if name == "pca" else FastICA
		return cls(n_components=effective, random_state=seed)

	if name == "random_projection":
		return GaussianRandomProjection(n_components=dim, random_state=seed)

	raise ValueError(f"Unknown image reducer: {name}")


def _apply_reducer(reducer, X_train, X_val, X_test):
	"""Fit on *X_train* and transform all three splits.  Pass-through when ``None``."""
	if reducer is None:
		return X_train, X_val, X_test
	return reducer.fit_transform(X_train), reducer.transform(X_val), reducer.transform(X_test)


# ===================================================================
# 5. Feature assembly
# ===================================================================


def _build_features(
	mode: str,
	X_tab: tuple[np.ndarray, np.ndarray, np.ndarray],
	X_img: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
	reducer_name: str,
	reducer_dim: int,
	seed: int,
	X_rep: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] = (None, None, None),
):
	"""Assemble final feature matrices for ``(train, val, test)``.

	In probing mode (*X_rep* provided), representations *replace* raw tabular
	features.  Image features are processed with optional reduction.

	Returns ``(X_train, X_val, X_test, fitted_reducer | None)``.
	"""
	tab_train, tab_val, tab_test = X_tab
	rep_train, rep_val, rep_test = X_rep

	# Probing mode: replace tabular with pre-extracted representations.
	if rep_train is not None:
		tab_train, tab_val, tab_test = rep_train, rep_val, rep_test

	if mode == "tabular":
		return tab_train, tab_val, tab_test, None

	img_train, img_val, img_test = X_img
	if img_train is None or img_val is None or img_test is None:
		raise ValueError(
			f"feature_mode='{mode}' requires image embeddings, but the dataset has none."
		)

	reducer = _fit_image_reducer(img_train, reducer_name, reducer_dim, seed)
	img_train, img_val, img_test = _apply_reducer(reducer, img_train, img_val, img_test)

	if mode == "image":
		return img_train, img_val, img_test, reducer

	if mode == "concat":
		return (
			np.concatenate([tab_train, img_train], axis=1),
			np.concatenate([tab_val, img_val], axis=1),
			np.concatenate([tab_test, img_test], axis=1),
			reducer,
		)

	raise ValueError(f"Unknown feature_mode: {mode}")


# ===================================================================
# 6. Feature experiment configs
# ===================================================================


def _resolve_feature_experiments(
	feature_mode: str,
	image_reducer: str,
	image_reducer_dim: int,
	feature_suite: bool,
	suite_reducer_dim: int,
) -> list[dict]:
	"""Build the list of feature experiment dicts to iterate over."""
	if not feature_suite:
		return [
			{
				"label": f"{feature_mode}_{image_reducer}",
				"feature_mode": feature_mode,
				"image_reducer": image_reducer,
				"image_reducer_dim": image_reducer_dim,
			}
		]

	d = suite_reducer_dim
	configs = [
		("tabular_only",     "tabular", "none"),
		("image_only",       "image",   "none"),
		(f"image_pca{d}",    "image",   "pca"),
		(f"image_ica{d}",    "image",   "ica"),
		(f"image_rp{d}",     "image",   "random_projection"),
		("concat",           "concat",  "none"),
		(f"concat_pca{d}",   "concat",  "pca"),
		(f"concat_ica{d}",   "concat",  "ica"),
		(f"concat_rp{d}",    "concat",  "random_projection"),
	]
	return [
		{"label": label, "feature_mode": mode, "image_reducer": reducer, "image_reducer_dim": d}
		for label, mode, reducer in configs
	]


# ===================================================================
# 7. Model construction
# ===================================================================


def _build_standard_models(
	task: str, seed: int, n_estimators: int, xgb_progress: bool, xgb_verbose_every: int,
) -> dict[str, object]:
	"""Instantiate the standard (non-probing) models for the given task."""
	if task == "regression":
		models: dict[str, object] = {
			"tabicl": TabICLRegressor(n_estimators=n_estimators, random_state=seed),
			"decision_tree": DecisionTreeRegressor(random_state=seed),
			"random_forest": RandomForestRegressor(
				n_estimators=100, random_state=seed, n_jobs=-1,
			),
			"xgboost": XGBRegressor(
				n_estimators=500, max_depth=8, learning_rate=0.1,
				tree_method="hist", device="cuda",
			),
		}
	else:
		models = {
			"tabicl": TabICLClassifier(n_estimators=n_estimators, random_state=seed),
			"decision_tree": DecisionTreeClassifier(random_state=seed),
			"random_forest": RandomForestClassifier(
				n_estimators=100, random_state=seed, min_samples_split=10, n_jobs=-1,
			),
			"xgboost": XGBClassifier(
				n_estimators=500, max_depth=8, learning_rate=0.1,
				tree_method="hist", device="cuda",
			),
		}
	xgb = models["xgboost"]
	xgb._fit_kwargs = {"verbose": xgb_verbose_every if xgb_progress else False}
	xgb._xgb_use_eval_set = True
	return models


# ===================================================================
# 8. Training & evaluation helpers
# ===================================================================


def _resolve_device(device: str) -> str:
	"""Map the ``--device`` flag to ``cuda`` or ``cpu``."""
	if device == "cpu":
		return "cpu"
	if device in ("cuda", "auto") and torch.cuda.is_available():
		return "cuda"
	return "cpu"


def _fit_sklearn_model(
	name: str, clf, X_train, y_train, X_val, y_val, device: str,
) -> dict[str, float]:
	"""Fit an sklearn-compatible model and return timing metrics.

	Models that expose a ``device`` parameter (TabICL, XGBoost) will be set to
	the requested device.  XGBoost additionally receives an eval set.
	"""
	try:
		clf.set_params(device=_resolve_device(device))
	except (ValueError, TypeError):
		pass  # model has no device parameter

	fit_kwargs = dict(getattr(clf, "_fit_kwargs", {}))
	if getattr(clf, "_xgb_use_eval_set", False):
		fit_kwargs["eval_set"] = [(X_val, y_val)]

	start = time.time()
	clf.fit(X_train, y_train, **fit_kwargs)
	return {"fit_seconds": float(time.time() - start)}


def _evaluate_model(task: str, clf, X, y) -> dict[str, float]:
	"""Predict with *clf* and return task-appropriate metrics / timing."""
	start = time.time()
	y_pred = np.asarray(clf.predict(X)).reshape(-1)
	y_true = np.asarray(y).reshape(-1)
	if task == "regression":
		return {
			"mse": float(mean_squared_error(y_true, y_pred)),
			"r2": float(r2_score(y_true, y_pred)),
			"eval_seconds": float(time.time() - start),
		}
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
		"eval_seconds": float(time.time() - start),
	}


def _run_probing_head(
	model_name: str,
	X_train: np.ndarray, y_train: np.ndarray,
	X_val: np.ndarray, y_val: np.ndarray,
	X_test: np.ndarray, y_test: np.ndarray,
	device: str,
	task: str,
) -> dict:
	"""Train and evaluate a probing head (``linear_probe`` or ``mlp``).

	Returns ``{fit: {...}, val: {...}, test: {...}}``.
	"""
	probe_device = _resolve_device(device)

	if task == "regression":
		if model_name == "linear_probe":
			head = LinearRegressionHead(input_dim=X_train.shape[1])
			lr = 1e-3
		else:
			head = MLPRegressionHead(input_dim=X_train.shape[1], hidden_dim=256)
			lr = 1e-4

		start = time.time()
		head, _ = train_head_regression(
			model=head,
			X_train=X_train, y_train=y_train,
			X_val=X_val, y_val=y_val,
			learning_rate=lr,
			num_epochs=100, batch_size=32,
			early_stopping_patience=10,
			device=probe_device,
			verbose=False,
		)
		fit_metrics = {"fit_seconds": float(time.time() - start)}

		def _eval_reg(X, y):
			t = time.time()
			preds = predict_head_regression(head, X, device=probe_device)
			return {
				"mse": float(mean_squared_error(y, preds)),
				"r2": float(r2_score(y, preds)),
				"eval_seconds": float(time.time() - t),
			}

		return {"fit": fit_metrics, "val": _eval_reg(X_val, y_val), "test": _eval_reg(X_test, y_test)}

	# Classification probing
	num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
	if model_name == "linear_probe":
		head = LinearProbe(input_dim=X_train.shape[1], num_classes=num_classes)
		lr = 1e-3
	else:
		head = MLPHead(input_dim=X_train.shape[1], num_classes=num_classes, hidden_dim=256)
		lr = 1e-4

	start = time.time()
	head, _ = train_head(
		model=head,
		X_train=X_train, y_train=y_train,
		X_val=X_val, y_val=y_val,
		num_classes=num_classes,
		learning_rate=lr,
		num_epochs=100, batch_size=32,
		early_stopping_patience=10,
		device=probe_device,
		verbose=False,
	)
	fit_metrics = {"fit_seconds": float(time.time() - start)}

	def _eval_cls(X, y):
		t = time.time()
		preds, _ = predict_head(head, X, device=probe_device)
		return {
			"accuracy": float(accuracy_score(y, preds)),
			"f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
			"eval_seconds": float(time.time() - t),
		}

	return {"fit": fit_metrics, "val": _eval_cls(X_val, y_val), "test": _eval_cls(X_test, y_test)}


def _print_model_results(name: str, results: dict) -> None:
	"""Pretty-print fit / val / test metrics for one model."""
	fit, val, test = results["fit"], results["val"], results["test"]
	print(f"\n[{name}]")
	print(f"  Fit  | time={fit['fit_seconds']:.2f}s")
	if "accuracy" in val:
		print(
			f"  Val  | acc={val['accuracy']:.4f}, "
			f"f1_macro={val['f1_macro']:.4f}, time={val['eval_seconds']:.2f}s"
		)
		print(
			f"  Test | acc={test['accuracy']:.4f}, "
			f"f1_macro={test['f1_macro']:.4f}, time={test['eval_seconds']:.2f}s"
		)
	else:
		print(
			f"  Val  | mse={val['mse']:.4f}, "
			f"r2={val['r2']:.4f}, time={val['eval_seconds']:.2f}s"
		)
		print(
			f"  Test | mse={test['mse']:.4f}, "
			f"r2={test['r2']:.4f}, time={test['eval_seconds']:.2f}s"
		)


def _save_results_snapshot(results: dict, results_path: Path) -> None:
	"""Persist a JSON snapshot atomically to reduce data loss risk on crashes."""
	results_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = results_path.with_suffix(results_path.suffix + ".tmp")
	tmp_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
	tmp_path.replace(results_path)


# ===================================================================
# 9. Main experiment loop
# ===================================================================


def run_experiment(
	dataset: str,
	data_dir: Path,
	module_path: Path,
	seed: int,
	n_estimators: int,
	max_train_samples: int | None,
	max_eval_samples: int | None,
	train_sizes: list[int] | None,
	methods: list[str] | None,
	results_path: Path | None,
	feature_mode: str,
	image_reducer: str,
	image_reducer_dim: int,
	feature_suite: bool,
	suite_reducer_dim: int,
	xgb_progress: bool,
	xgb_verbose_every: int,
	device: str,
	tabicl_features_dir: Path | None = None,
) -> None:
	# --- Data loading ------------------------------------------------
	need_images = feature_suite or feature_mode in ("image", "concat")
	data_root, splits, dataset_metadata = _load_data(dataset, data_dir, module_path, need_images)
	task = str(dataset_metadata.get("task", "classification")).lower()
	if task not in ("classification", "regression"):
		raise ValueError(f"Unsupported task '{task}' in dataset metadata")

	X_tab_full = {s: d[0] for s, d in splits.items()}
	X_img_full = {s: d[1] for s, d in splits.items()}
	y_full     = {s: d[2] for s, d in splits.items()}

	# --- Feature experiment configs ----------------------------------
	feature_experiments = _resolve_feature_experiments(
		feature_mode, image_reducer, image_reducer_dim, feature_suite, suite_reducer_dim,
	)

	# --- Sub-sample val / test ---------------------------------------
	val_idx  = _sample_indices(y_full["val"],  max_eval_samples, seed, task=task)
	test_idx = _sample_indices(y_full["test"], max_eval_samples, seed, task=task)

	X_tab_val  = X_tab_full["val"][val_idx]
	X_tab_test = X_tab_full["test"][test_idx]
	y_val  = y_full["val"][val_idx]
	y_test = y_full["test"][test_idx]
	X_img_val  = _maybe_index(X_img_full["val"],  val_idx)
	X_img_test = _maybe_index(X_img_full["test"], test_idx)

	# --- Train sizes -------------------------------------------------
	train_sizes_to_run = (
		[max_train_samples] if not train_sizes
		else [s for s in train_sizes if s > 0]
	)
	if not train_sizes_to_run:
		raise ValueError("No valid train sizes provided. Use positive integers.")

	# --- Select methods ----------------------------------------------
	if methods:
		selected_methods = methods
	elif tabicl_features_dir is not None:
		selected_methods = PROBING_METHODS
	else:
		selected_methods = STANDARD_METHODS

	probing_mode = any(m in PROBING_METHODS for m in selected_methods)
	probing_reps_supported = task == "classification"
	if probing_mode and not probing_reps_supported:
		raise ValueError(
			"Probing methods ('linear_probe', 'mlp') are not supported for regression tasks yet. "
			"This run would require TabICL representations for regression, but the current "
			"TabICLRegressor API does not expose them. Please run with standard methods only "
			"for now, or add the planned regression probing entry point first."
		)
	if tabicl_features_dir is not None:
		print(
			"[info] --tabicl-features-dir is now legacy. "
			"Probing representations are extracted on-the-fly per train subsample."
		)

	# --- Print summary -----------------------------------------------
	print(f"\n{dataset} experiment  seed={seed}  device={device}  probing={probing_mode}")
	print(f"  data_root  : {data_root}")
	if task == "classification":
		print(f"  tabular    : {X_tab_full['train'].shape}  classes={len(np.unique(y_full['train']))}")
	else:
		print(f"  tabular    : {X_tab_full['train'].shape}  task=regression")
	if X_img_full["train"] is not None:
		print(f"  image      : {X_img_full['train'].shape}")
	print(f"  methods    : {selected_methods}")
	print(f"  train sizes: {train_sizes_to_run}\n")

	# --- Results container --------------------------------------------
	results = {
		"metadata": {
			"dataset": dataset, "data_root": str(data_root),
			"task": task,
			"seed": seed, "device": device,
			"probing_mode": probing_mode,
			"feature_suite": feature_suite, "feature_experiments": feature_experiments,
			"train_sizes": train_sizes_to_run, "methods": selected_methods,
			"max_eval_samples": max_eval_samples,
		},
		"experiments": [],
	}

	experiment_results_by_label = {
		cfg["label"]: {**cfg, "runs": []}
		for cfg in feature_experiments
	}

	if results_path is not None:
		_save_results_snapshot(results, results_path)

	# --- Outer loop: train sizes (for fair on-the-fly probing reps) ---
	for train_size in train_sizes_to_run:
		train_idx   = _sample_indices(y_full["train"], train_size, seed, task=task)
		X_tab_train = X_tab_full["train"][train_idx]
		y_train     = y_full["train"][train_idx]
		X_img_train = _maybe_index(X_img_full["train"], train_idx)

		label = "all" if train_size is None else str(train_size)
		print(f"\n=== Train size: {label} | actual: {len(y_train)} ===")

		X_rep_train: np.ndarray | None = None
		X_rep_val: np.ndarray | None = None
		X_rep_test: np.ndarray | None = None
		if probing_mode and probing_reps_supported:
			print("[info] Extracting probing representations on-the-fly for current train subsample...")
			X_rep_train, X_rep_val, X_rep_test = _extract_representations_via_subprocess(
				X_train=X_tab_train,
				y_train=y_train,
				X_val=X_tab_val,
				X_test=X_tab_test,
				n_estimators=n_estimators,
				seed=seed,
			)

		# --- Inner loop: feature configs ---------------------------------
		for cfg in feature_experiments:
			cfg_label   = cfg["label"]
			cfg_mode    = cfg["feature_mode"]
			cfg_reducer = cfg["image_reducer"]
			cfg_dim     = cfg["image_reducer_dim"]

			print(
				f"\n##### {cfg_label}  "
				f"(mode={cfg_mode}  reducer={cfg_reducer}  dim={cfg_dim}) "
				f"train_size={label} #####"
			)

			feature_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, object | None]] = {}

			def _features_for_method(model_name: str):
				cache_key = "probing" if model_name in PROBING_METHODS else "standard"
				if cache_key in feature_cache:
					return feature_cache[cache_key]

				use_reps = (
					cache_key == "probing"
					and probing_reps_supported
					and X_rep_train is not None
				)
				X_train_f, X_val_f, X_test_f, fitted_reducer = _build_features(
					mode=cfg_mode,
					X_tab=(X_tab_train, X_tab_val, X_tab_test),
					X_img=(X_img_train, X_img_val, X_img_test),
					reducer_name=cfg_reducer,
					reducer_dim=cfg_dim,
					seed=seed,
					X_rep=(X_rep_train, X_rep_val, X_rep_test) if use_reps else (None, None, None),
				)
				feature_cache[cache_key] = (X_train_f, X_val_f, X_test_f, fitted_reducer)
				return feature_cache[cache_key]

			std_models = _build_standard_models(task, seed, n_estimators, xgb_progress, xgb_verbose_every)

			run_results = {
				"train_size_requested": None if train_size is None else int(train_size),
				"train_size_actual": int(len(y_train)),
				"models": {},
			}

			for model_name in selected_methods:
				X_train_f, X_val_f, X_test_f, fitted_reducer = _features_for_method(model_name)
				feature_source = (
					"tabicl_representations"
					if model_name in PROBING_METHODS and probing_reps_supported
					else ("assembled_features" if model_name in PROBING_METHODS else "raw_tabular")
				)
				print(f"[info] {model_name} feature_source: {feature_source} | dim={X_train_f.shape[1]}")
				if fitted_reducer is not None:
					print(f"[info] Reducer: {cfg_reducer} (dim={getattr(fitted_reducer, 'n_components', '?')})")

				# -- Probing heads --
				if model_name in PROBING_METHODS:
					print(f"[info] {model_name} device: {_resolve_device(device)}")
					model_results = _run_probing_head(
						model_name, X_train_f, y_train, X_val_f, y_val, X_test_f, y_test, device, task,
					)

				# -- Standard sklearn / TabICL models --
				elif model_name in std_models:
					clf = std_models[model_name]
					print(f"[info] {model_name} device: {_resolve_device(device)}")
					fit_m = _fit_sklearn_model(model_name, clf, X_train_f, y_train, X_val_f, y_val, device)
					model_results = {
						"fit": fit_m,
						"val": _evaluate_model(task, clf, X_val_f, y_val),
						"test": _evaluate_model(task, clf, X_test_f, y_test),
					}
				else:
					print(f"[warning] Unknown method '{model_name}'; skipping.")
					continue

				_print_model_results(model_name, model_results)
				run_results["models"][model_name] = model_results

			experiment_results_by_label[cfg_label]["runs"].append(run_results)

			if results_path is not None:
				results["experiments"] = [
					experiment_results_by_label[c["label"]]
					for c in feature_experiments
				]
				if len(results["experiments"]) == 1:
					results["runs"] = results["experiments"][0]["runs"]
				_save_results_snapshot(results, results_path)

	results["experiments"] = [
		experiment_results_by_label[cfg["label"]]
		for cfg in feature_experiments
	]

	# Convenience shortcut when there is only one feature experiment.
	if len(results["experiments"]) == 1:
		results["runs"] = results["experiments"][0]["runs"]

	if results_path is not None:
		_save_results_snapshot(results, results_path)
		print(f"\nSaved results to: {results_path}")


# ===================================================================
# 10. CLI & entry point
# ===================================================================


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Multi-dataset experiment runner")

	# Dataset selection
	p.add_argument("--dataset", type=str, choices=DATASET_NAMES, default="dvm",
		help="Dataset to evaluate (default: dvm)")

	# Paths (defaults resolved per-dataset after parsing)
	p.add_argument("--data-dir", type=Path, default=None,
		help="Root data directory (default: per-dataset)")
	p.add_argument("--module-path", "--dvm-module-path", dest="module_path",
		type=Path, default=None,
		help="Path to dataset loader module (default: per-dataset)")
	p.add_argument("--results-path", type=Path, default=None,
		help="JSON results file (default: results/<dataset>_experiments_results.json)")
	p.add_argument("--tabicl-features-dir", type=Path, default=None,
		help=(
			"Legacy flag. Pre-extracted features are no longer used; probing "
			"representations are extracted on-the-fly per train subsample. "
			"When set without --methods, defaults to probing methods."
		))

	# Seed & sizing
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--seeds", type=int, nargs="+", default=None,
		help="Optional list of seeds to run (overrides --seed), e.g. 0 1 2 3 4")
	p.add_argument("--n-estimators", type=int, default=1, help="TabICL ensemble size")
	p.add_argument("--max-train-samples", type=int, default=20000, help="Cap on train rows (<=0 for all)")
	p.add_argument("--max-eval-samples", type=int, default=20000, help="Cap on val/test rows (<=0 for all)")
	p.add_argument("--train-sizes", type=int, nargs="+", default=None,
		help="Specific train sizes to sweep (e.g. 500 2000 10000)")

	# Methods
	p.add_argument("--methods", type=str, nargs="+", choices=ALL_METHODS, default=None,
		help="Subset of methods to evaluate")

	# Features
	p.add_argument("--feature-mode", type=str, choices=["tabular", "image", "concat"], default="tabular")
	p.add_argument("--image-reducer", type=str, choices=["none", "pca", "ica", "random_projection"], default="none")
	p.add_argument("--image-reducer-dim", type=int, default=128)
	p.add_argument("--feature-suite", action="store_true",
		help="Run the full 9-config feature comparison suite")
	p.add_argument("--suite-reducer-dim", "--suite-pca-dim", dest="suite_reducer_dim",
		type=int, default=64, help="Reducer dim for --feature-suite (legacy alias: --suite-pca-dim)")

	# XGBoost
	p.add_argument("--xgb-progress", action="store_true")
	p.add_argument("--xgb-verbose-every", type=int, default=10)

	# Device
	p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")

	return p.parse_args()


def main() -> None:
	args = parse_args()
	cfg = DATASET_CONFIGS[args.dataset]

	# Resolve per-dataset defaults for paths not explicitly provided.
	if args.data_dir is None:
		args.data_dir = cfg["data_dir"]
	if args.module_path is None:
		args.module_path = cfg["module_path"]
	if args.results_path is None:
		args.results_path = Path(f"results/{args.dataset}_experiments_results.json")

	seeds_to_run = args.seeds if args.seeds is not None else [args.seed]
	if len(seeds_to_run) != len(set(seeds_to_run)):
		raise ValueError("--seeds contains duplicate values; please provide unique seeds.")

	max_train = None if args.max_train_samples <= 0 else args.max_train_samples
	max_eval  = None if args.max_eval_samples  <= 0 else args.max_eval_samples

	for idx, seed in enumerate(seeds_to_run, start=1):
		if len(seeds_to_run) == 1:
			seed_results_path = args.results_path
		else:
			seed_results_path = args.results_path.with_name(
				f"{args.results_path.stem}_seed{seed}{args.results_path.suffix}"
			)
		print(f"\n[seed-run] {idx}/{len(seeds_to_run)}  seed={seed}  results={seed_results_path}")

		run_experiment(
			dataset=args.dataset,
			data_dir=args.data_dir,
			module_path=args.module_path,
			seed=seed,
			n_estimators=args.n_estimators,
			max_train_samples=max_train,
			max_eval_samples=max_eval,
			train_sizes=args.train_sizes,
			methods=args.methods,
			results_path=seed_results_path,
			feature_mode=args.feature_mode,
			image_reducer=args.image_reducer,
			image_reducer_dim=args.image_reducer_dim,
			feature_suite=args.feature_suite,
			suite_reducer_dim=args.suite_reducer_dim,
			xgb_progress=args.xgb_progress,
			xgb_verbose_every=args.xgb_verbose_every,
			device=args.device,
			tabicl_features_dir=args.tabicl_features_dir,
		)


if __name__ == "__main__":
	main()
