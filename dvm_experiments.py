"""DVM Dataset experiment runner.

Benchmarks classification models on the DVM car dataset using tabular features,
DINOv2 image embeddings, or their concatenation.  Supports optional dimensionality
reduction (PCA / ICA / random projection) on image features and a "probing mode"
that evaluates LinearProbe / MLP heads on pre-extracted TabICL representations.

See DVM_EXPERIMENTS.md for a full walkthrough.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.random_projection import GaussianRandomProjection
from sklearn.tree import DecisionTreeClassifier
from tabicl import TabICLClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from baseline_heads import LinearProbe, MLPHead, predict_head, train_head

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(
	"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset"
)
DEFAULT_DVM_MODULE_PATH = Path(
	"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset/dvm_dataset_with_dinov2.py"
)
DEFAULT_TABICL_FEATURES_DIR = Path(
	"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset/tabiclv2_features"
)

STANDARD_METHODS = ["tabicl", "decision_tree", "random_forest", "xgboost"]
PROBING_METHODS = ["linear_probe", "mlp"]
ALL_METHODS = STANDARD_METHODS + PROBING_METHODS


# ===================================================================
# 1. Data loading
# ===================================================================


def _import_load_dvm_dataset(dvm_module_path: Path):
	"""Dynamically import ``load_dvm_dataset`` from the given module path."""
	if not dvm_module_path.exists():
		raise FileNotFoundError(f"DVM module not found: {dvm_module_path}")

	spec = importlib.util.spec_from_file_location("dvm_dataset", str(dvm_module_path))
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module spec from {dvm_module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	if not hasattr(module, "load_dvm_dataset"):
		raise AttributeError(f"Module {dvm_module_path} does not define load_dvm_dataset")

	return module.load_dvm_dataset


def _extract_modalities_from_dataset(dataset):
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
		targets.append(int(item["target"]))

	X_tab = np.stack(tabular_features)
	X_img = None if any(f is None for f in image_features) else np.stack(image_features)
	y = np.asarray(targets, dtype=np.int64)
	return X_tab, X_img, y


def _load_data(data_dir: Path, dvm_module_path: Path, need_images: bool):
	"""Load the DVM dataset and extract numpy arrays for each split.

	Returns ``(data_root, splits)`` where *splits* maps split name to
	``(X_tab, X_img | None, y)``.
	"""
	data_root = data_dir.parent if data_dir.name == "preprocessed_csvs" else data_dir
	load_fn = _import_load_dvm_dataset(dvm_module_path)
	train_loader, val_loader, test_loader, _metadata = load_fn(
		data_dir=str(data_root), batch_size=2048, num_workers=0, use_images=need_images,
	)

	print("Extracting full train / val / test features from DVM dataset...")
	splits = {
		name: _extract_modalities_from_dataset(loader.dataset)
		for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]
	}
	return data_root, splits


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
	seed: int, n_estimators: int, xgb_progress: bool, xgb_verbose_every: int,
) -> dict[str, object]:
	"""Instantiate the standard (non-probing) classifiers."""
	models: dict[str, object] = {
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


def _evaluate_clf(clf, X, y) -> dict[str, float]:
	"""Predict with *clf* and return accuracy / f1 / timing."""
	start = time.time()
	y_pred = np.asarray(clf.predict(X))
	y_true = np.asarray(y)
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
) -> dict:
	"""Train and evaluate a probing head (``linear_probe`` or ``mlp``).

	Returns ``{fit: {...}, val: {...}, test: {...}}``.
	"""
	# Use all splits to determine num_classes so the output layer covers
	# classes that may be absent from a small training subsample.
	num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
	probe_device = _resolve_device(device)

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

	def _eval(X, y):
		t = time.time()
		preds, _ = predict_head(head, X, device=probe_device)
		return {
			"accuracy": float(accuracy_score(y, preds)),
			"f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
			"eval_seconds": float(time.time() - t),
		}

	return {"fit": fit_metrics, "val": _eval(X_val, y_val), "test": _eval(X_test, y_test)}


def _print_model_results(name: str, results: dict) -> None:
	"""Pretty-print fit / val / test metrics for one model."""
	fit, val, test = results["fit"], results["val"], results["test"]
	print(f"\n[{name}]")
	print(f"  Fit  | time={fit['fit_seconds']:.2f}s")
	print(
		f"  Val  | acc={val['accuracy']:.4f}, "
		f"f1_macro={val['f1_macro']:.4f}, time={val['eval_seconds']:.2f}s"
	)
	print(
		f"  Test | acc={test['accuracy']:.4f}, "
		f"f1_macro={test['f1_macro']:.4f}, time={test['eval_seconds']:.2f}s"
	)


# ===================================================================
# 9. Main experiment loop
# ===================================================================


def run_experiment(
	data_dir: Path,
	dvm_module_path: Path,
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
	data_root, splits = _load_data(data_dir, dvm_module_path, need_images)

	X_tab_full = {s: d[0] for s, d in splits.items()}
	X_img_full = {s: d[1] for s, d in splits.items()}
	y_full     = {s: d[2] for s, d in splits.items()}

	# --- Probing representations (optional) --------------------------
	probing_mode = tabicl_features_dir is not None
	X_rep_full: dict[str, np.ndarray | None] = {"train": None, "val": None, "test": None}
	if probing_mode:
		reps = _load_all_representations(
			tabicl_features_dir,
			len(y_full["train"]), len(y_full["val"]), len(y_full["test"]),
		)
		X_rep_full["train"], X_rep_full["val"], X_rep_full["test"] = reps

	# --- Feature experiment configs ----------------------------------
	feature_experiments = _resolve_feature_experiments(
		feature_mode, image_reducer, image_reducer_dim, feature_suite, suite_reducer_dim,
	)

	# --- Sub-sample val / test ---------------------------------------
	val_idx  = _stratified_indices(y_full["val"],  max_eval_samples, seed)
	test_idx = _stratified_indices(y_full["test"], max_eval_samples, seed)

	X_tab_val  = X_tab_full["val"][val_idx]
	X_tab_test = X_tab_full["test"][test_idx]
	y_val  = y_full["val"][val_idx]
	y_test = y_full["test"][test_idx]
	X_img_val  = _maybe_index(X_img_full["val"],  val_idx)
	X_img_test = _maybe_index(X_img_full["test"], test_idx)
	X_rep_val  = _maybe_index(X_rep_full["val"],  val_idx)
	X_rep_test = _maybe_index(X_rep_full["test"], test_idx)

	# --- Train sizes -------------------------------------------------
	train_sizes_to_run = (
		[max_train_samples] if not train_sizes
		else [s for s in train_sizes if s > 0]
	)
	if not train_sizes_to_run:
		raise ValueError("No valid train sizes provided. Use positive integers.")

	# --- Select methods ----------------------------------------------
	available = PROBING_METHODS if probing_mode else STANDARD_METHODS
	selected_methods = methods if methods else available

	# --- Print summary -----------------------------------------------
	print(f"\nDVM experiment  seed={seed}  device={device}  probing={probing_mode}")
	print(f"  data_root  : {data_root}")
	print(f"  tabular    : {X_tab_full['train'].shape}  classes={len(np.unique(y_full['train']))}")
	if X_img_full["train"] is not None:
		print(f"  image      : {X_img_full['train'].shape}")
	print(f"  methods    : {selected_methods}")
	print(f"  train sizes: {train_sizes_to_run}\n")

	# --- Results container --------------------------------------------
	results = {
		"metadata": {
			"data_root": str(data_root), "seed": seed, "device": device,
			"feature_suite": feature_suite, "feature_experiments": feature_experiments,
			"train_sizes": train_sizes_to_run, "methods": selected_methods,
			"max_eval_samples": max_eval_samples,
		},
		"experiments": [],
	}

	# --- Outer loop: feature configs ---------------------------------
	for cfg in feature_experiments:
		cfg_label   = cfg["label"]
		cfg_mode    = cfg["feature_mode"]
		cfg_reducer = cfg["image_reducer"]
		cfg_dim     = cfg["image_reducer_dim"]

		print(f"\n##### {cfg_label}  (mode={cfg_mode}  reducer={cfg_reducer}  dim={cfg_dim}) #####")
		experiment_results = {**cfg, "runs": []}

		# --- Inner loop: train sizes ---------------------------------
		for train_size in train_sizes_to_run:
			train_idx   = _stratified_indices(y_full["train"], train_size, seed)
			X_tab_train = X_tab_full["train"][train_idx]
			y_train     = y_full["train"][train_idx]
			X_img_train = _maybe_index(X_img_full["train"], train_idx)
			X_rep_train = _maybe_index(X_rep_full["train"], train_idx)

			X_train, X_val_f, X_test_f, fitted_reducer = _build_features(
				mode=cfg_mode,
				X_tab=(X_tab_train, X_tab_val, X_tab_test),
				X_img=(X_img_train, X_img_val, X_img_test),
				reducer_name=cfg_reducer, reducer_dim=cfg_dim, seed=seed,
				X_rep=(X_rep_train, X_rep_val, X_rep_test),
			)

			label = "all" if train_size is None else str(train_size)
			print(f"\n=== Train size: {label} | actual: {len(y_train)} | dim: {X_train.shape[1]} ===")
			if fitted_reducer is not None:
				print(f"[info] Reducer: {cfg_reducer} (dim={getattr(fitted_reducer, 'n_components', '?')})")

			std_models = _build_standard_models(seed, n_estimators, xgb_progress, xgb_verbose_every)

			run_results = {
				"train_size_requested": None if train_size is None else int(train_size),
				"train_size_actual": int(len(y_train)),
				"models": {},
			}

			for model_name in selected_methods:
				# -- Probing heads --
				if model_name in PROBING_METHODS:
					if not probing_mode:
						print(f"[warning] '{model_name}' requires --tabicl-features-dir; skipping.")
						continue
					print(f"[info] {model_name} device: {_resolve_device(device)}")
					model_results = _run_probing_head(
						model_name, X_train, y_train, X_val_f, y_val, X_test_f, y_test, device,
					)

				# -- Standard sklearn / TabICL models --
				elif model_name in std_models:
					clf = std_models[model_name]
					print(f"[info] {model_name} device: {_resolve_device(device)}")
					fit_m = _fit_sklearn_model(model_name, clf, X_train, y_train, X_val_f, y_val, device)
					model_results = {
						"fit": fit_m,
						"val": _evaluate_clf(clf, X_val_f, y_val),
						"test": _evaluate_clf(clf, X_test_f, y_test),
					}
				else:
					print(f"[warning] Unknown method '{model_name}'; skipping.")
					continue

				_print_model_results(model_name, model_results)
				run_results["models"][model_name] = model_results

			experiment_results["runs"].append(run_results)
		results["experiments"].append(experiment_results)

	# Convenience shortcut when there is only one feature experiment.
	if len(results["experiments"]) == 1:
		results["runs"] = results["experiments"][0]["runs"]

	if results_path is not None:
		results_path.parent.mkdir(parents=True, exist_ok=True)
		results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
		print(f"\nSaved results to: {results_path}")


# ===================================================================
# 10. CLI & entry point
# ===================================================================


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="DVM dataset experiment runner")

	# Paths
	p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	p.add_argument("--dvm-module-path", type=Path, default=DEFAULT_DVM_MODULE_PATH)
	p.add_argument("--results-path", type=Path, default=Path("results/dvm_experiments_results.json"))
	p.add_argument("--tabicl-features-dir", type=Path, default=None,
		help="Pre-extracted TabICL representations dir (enables probing mode)")

	# Seed & sizing
	p.add_argument("--seed", type=int, default=0)
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

	max_train = None if args.max_train_samples <= 0 else args.max_train_samples
	max_eval  = None if args.max_eval_samples  <= 0 else args.max_eval_samples

	# Auto-resolve features dir when probing methods are requested.
	if (
		args.tabicl_features_dir is None
		and args.methods
		and set(args.methods) & set(PROBING_METHODS)
	):
		args.tabicl_features_dir = DEFAULT_TABICL_FEATURES_DIR
		print(f"[info] --tabicl-features-dir not set; defaulting to {args.tabicl_features_dir}")

	run_experiment(
		data_dir=args.data_dir,
		dvm_module_path=args.dvm_module_path,
		seed=args.seed,
		n_estimators=args.n_estimators,
		max_train_samples=max_train,
		max_eval_samples=max_eval,
		train_sizes=args.train_sizes,
		methods=args.methods,
		results_path=args.results_path,
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
