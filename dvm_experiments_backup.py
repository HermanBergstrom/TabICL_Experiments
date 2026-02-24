from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import FastICA, PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier

from tqdm import tqdm

from baseline_heads import LinearProbe, MLPHead, train_head, predict_head

try:
	from xgboost import XGBClassifier
except ImportError:
	XGBClassifier = None


DEFAULT_TABICL_FEATURES_DIR = Path(
	"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset/tabiclv2_features"
)


DEFAULT_DATA_DIR = Path(
	"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset"
)

#DEFAULT_DVM_MODULE_PATH = Path(
	#"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset/dvm_dataset.py"
#)

DEFAULT_DVM_MODULE_PATH = Path(
	"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset/dvm_dataset_with_dinov2.py"
)


def _import_load_dvm_dataset(dvm_module_path: Path):
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
	tabular_features = []
	image_features = []
	targets = []

	for index in tqdm(range(len(dataset))):
		item = dataset[index]
		tabular = item.get("tabular")
		image_embedding = item.get("image_embedding")

		if tabular is None:
			raise ValueError("Dataset item is missing 'tabular' features")

		tabular_features.append(np.asarray(tabular, dtype=np.float32))

		if image_embedding is None:
			image_features.append(None)
		else:
			image_features.append(np.asarray(image_embedding, dtype=np.float32))

		targets.append(int(item["target"]))

	X_tab = np.asarray(tabular_features, dtype=np.float32)
	if any(feature is None for feature in image_features):
		X_img = None
	else:
		X_img = np.asarray(image_features, dtype=np.float32)
	y = np.asarray(targets, dtype=np.int64)
	return X_tab, X_img, y


def _sample_indices(y: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
	if max_samples is None or len(y) <= max_samples:
		return np.arange(len(y), dtype=np.int64)

	splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
	selected_idx, _ = next(splitter.split(np.zeros((len(y), 1)), y))
	return selected_idx


def _stratified_train_indices(y_train: np.ndarray, train_size: int | None, seed: int) -> np.ndarray:
	if train_size is None or len(y_train) <= train_size:
		return np.arange(len(y_train), dtype=np.int64)

	n_classes = int(len(np.unique(y_train)))
	if train_size < n_classes:
		raise ValueError(
			f"Requested train_size={train_size} is too small for stratified sampling across {n_classes} classes."
		)

	splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
	selected_idx, _ = next(splitter.split(np.zeros((len(y_train), 1)), y_train))
	return selected_idx


# ---------------------------------------------------------------------------
# TabICL pre-extracted representation helpers
# ---------------------------------------------------------------------------

def _load_tabicl_representations(
	features_dir: Path,
	split: str,
	n_rows: int,
) -> np.ndarray:
	"""Load pre-extracted TabICL representations for *split* and return a 2-D array.

	The .pt file stores ``{row_index: tensor(n_estimators, embed_dim)}``.
	We average across estimators to get a single vector per row and stack them
	in dataset order (indices 0 … n_rows-1).

	Returns
	-------
	np.ndarray of shape (n_rows, embed_dim)
	"""
	pt_path = features_dir / f"{split}_representations.pt"
	if not pt_path.exists():
		raise FileNotFoundError(
			f"Pre-extracted TabICL features not found at {pt_path}.\n"
			"Run extract_tabicl_features.py first."
		)
	rep_dict: dict[int, torch.Tensor] = torch.load(pt_path, map_location="cpu", weights_only=True)

	# Average across estimators → (embed_dim,)
	rows = []
	for idx in range(n_rows):
		if idx not in rep_dict:
			raise KeyError(
				f"Row index {idx} missing from {pt_path}. "
				"Re-run extract_tabicl_features.py to regenerate."
			)
		rows.append(rep_dict[idx].mean(dim=0).numpy())

	return np.stack(rows, axis=0).astype(np.float32)


def _load_tabicl_for_splits(
	features_dir: Path,
	n_train: int,
	n_val: int,
	n_test: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Convenience wrapper: load representations for all three splits."""
	X_rep_train = _load_tabicl_representations(features_dir, "train", n_train)
	X_rep_val = _load_tabicl_representations(features_dir, "val", n_val)
	X_rep_test = _load_tabicl_representations(features_dir, "test", n_test)
	print(
		f"[info] Loaded TabICL representations: "
		f"train={X_rep_train.shape}, val={X_rep_val.shape}, test={X_rep_test.shape}"
	)
	return X_rep_train, X_rep_val, X_rep_test


def _fit_image_reducer(
	X_image_train: np.ndarray,
	reducer_name: str,
	reducer_dim: int,
	seed: int,
):
	if reducer_name == "none":
		return None

	if reducer_dim <= 0:
		raise ValueError("--image-reducer-dim must be > 0 when reducer is enabled")

	if reducer_name == "pca":
		max_dim = min(X_image_train.shape[0], X_image_train.shape[1])
		effective_dim = min(reducer_dim, max_dim)
		if effective_dim != reducer_dim:
			print(
				f"[warning] Reducing PCA components from {reducer_dim} to {effective_dim} due to train shape constraints."
			)
		return PCA(n_components=effective_dim, random_state=seed)

	if reducer_name == "ica":
		max_dim = min(X_image_train.shape[0], X_image_train.shape[1])
		effective_dim = min(reducer_dim, max_dim)
		if effective_dim != reducer_dim:
			print(
				f"[warning] Reducing ICA components from {reducer_dim} to {effective_dim} due to train shape constraints."
			)
		return FastICA(n_components=effective_dim, random_state=seed)

	if reducer_name == "random_projection":
		return GaussianRandomProjection(n_components=reducer_dim, random_state=seed)

	raise ValueError(f"Unknown image reducer: {reducer_name}")


def _build_features_for_mode(
	feature_mode: str,
	X_tab_train: np.ndarray,
	X_tab_val: np.ndarray,
	X_tab_test: np.ndarray,
	X_img_train: np.ndarray | None,
	X_img_val: np.ndarray | None,
	X_img_test: np.ndarray | None,
	image_reducer: str,
	image_reducer_dim: int,
	seed: int,
	X_rep_train: np.ndarray | None = None,
	X_rep_val: np.ndarray | None = None,
	X_rep_test: np.ndarray | None = None,
):
	# When probing representations are provided, use them in place of raw tabular features
	if X_rep_train is not None:
		X_tab_train = X_rep_train
		X_tab_val = X_rep_val
		X_tab_test = X_rep_test

	if feature_mode == "tabular":
		return X_tab_train, X_tab_val, X_tab_test, None

	if X_img_train is None or X_img_val is None or X_img_test is None:
		raise ValueError(
			f"feature_mode='{feature_mode}' requires image embeddings, but dataset does not provide them"
		)

	reducer = _fit_image_reducer(
		X_image_train=X_img_train,
		reducer_name=image_reducer,
		reducer_dim=image_reducer_dim,
		seed=seed,
	)

	if reducer is not None:
		X_img_train_used = reducer.fit_transform(X_img_train)
		X_img_val_used = reducer.transform(X_img_val)
		X_img_test_used = reducer.transform(X_img_test)
	else:
		X_img_train_used = X_img_train
		X_img_val_used = X_img_val
		X_img_test_used = X_img_test

	if feature_mode == "image":
		return X_img_train_used, X_img_val_used, X_img_test_used, reducer

	if feature_mode == "concat":
		X_train = np.concatenate([X_tab_train, X_img_train_used], axis=1)
		X_val = np.concatenate([X_tab_val, X_img_val_used], axis=1)
		X_test = np.concatenate([X_tab_test, X_img_test_used], axis=1)
		return X_train, X_val, X_test, reducer

	raise ValueError(f"Unknown feature_mode: {feature_mode}")


def _resolve_feature_experiments(
	feature_mode: str,
	image_reducer: str,
	image_reducer_dim: int,
	feature_suite: bool,
	suite_reducer_dim: int,
):
	if not feature_suite:
		return [
			{
				"label": f"{feature_mode}_{image_reducer}",
				"feature_mode": feature_mode,
				"image_reducer": image_reducer,
				"image_reducer_dim": image_reducer_dim,
			}
		]

	return [
		{
			"label": "tabular_only",
			"feature_mode": "tabular",
			"image_reducer": "none",
			"image_reducer_dim": suite_reducer_dim,
		},
		{
			"label": "image_only",
			"feature_mode": "image",
			"image_reducer": "none",
			"image_reducer_dim": suite_reducer_dim,
		},
		{
			"label": f"image_pca{suite_reducer_dim}",
			"feature_mode": "image",
			"image_reducer": "pca",
			"image_reducer_dim": suite_reducer_dim,
		},
		{
			"label": f"image_ica{suite_reducer_dim}",
			"feature_mode": "image",
			"image_reducer": "ica",
			"image_reducer_dim": suite_reducer_dim,
		},
		{
			"label": f"image_rp{suite_reducer_dim}",
			"feature_mode": "image",
			"image_reducer": "random_projection",
			"image_reducer_dim": suite_reducer_dim,
		},
		{
			"label": "concat",
			"feature_mode": "concat",
			"image_reducer": "none",
			"image_reducer_dim": suite_reducer_dim,
		},
		{
			"label": f"concat_pca{suite_reducer_dim}",
			"feature_mode": "concat",
			"image_reducer": "pca",
			"image_reducer_dim": suite_reducer_dim,
		},
		{
			"label": f"concat_ica{suite_reducer_dim}",
			"feature_mode": "concat",
			"image_reducer": "ica",
			"image_reducer_dim": suite_reducer_dim,
		},
		{
			"label": f"concat_rp{suite_reducer_dim}",
			"feature_mode": "concat",
			"image_reducer": "random_projection",
			"image_reducer_dim": suite_reducer_dim,
		},
	]


def _maybe_subsample(
	X,
	y: np.ndarray,
	max_samples: int | None,
	seed: int,
) -> tuple[object, np.ndarray]:
	if max_samples is None or len(y) <= max_samples:
		return X, y

	splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
	selected_idx, _ = next(splitter.split(X, y))
	if hasattr(X, "iloc"):
		return X.iloc[selected_idx].reset_index(drop=True), y[selected_idx]
	return X[selected_idx], y[selected_idx]


def _to_numpy(array_like):
	if hasattr(array_like, "detach") and hasattr(array_like, "cpu"):
		return array_like.detach().cpu().numpy()
	if hasattr(array_like, "get"):
		return array_like.get()
	return np.asarray(array_like)


def _maybe_to_xgb_cuda_inputs(clf, X, y):
	if getattr(clf, "_active_device", "cpu") != "cuda":
		return X, y

	try:
		import cupy as cp
	except ImportError:
		print(
			f"[warning] {getattr(clf, '_model_name', 'model')} requested CUDA but cupy is not installed; using CPU arrays."
		)
		return X, y

	try:
		return cp.asarray(X), cp.asarray(y)
	except Exception as exc:
		err_msg = str(exc).splitlines()[0] if str(exc) else repr(exc)
		print(
			f"[warning] Failed to move data to CUDA for {getattr(clf, '_model_name', 'model')} ({err_msg}); using CPU arrays."
		)
		return X, y


def _set_model_device(clf, model_name: str, device: str) -> None:
	clf._model_name = model_name
	clf._active_device = "cpu"
	clf._auto_fallback = False
	clf._device_param_key = None

	if device == "cpu":
		return

	candidates = [
		("device", "cuda", "cpu"),
		("inference_device", "cuda", "cpu"),
		("accelerator", "cuda", "cpu"),
		("use_cuda", True, False),
		("gpu", True, False),
		("gpu_id", 0, -1),
	]

	for param_name, cuda_value, _ in candidates:
		try:
			clf.set_params(**{param_name: cuda_value})
			clf._active_device = "cuda"
			clf._device_param_key = param_name
			if device == "auto":
				clf._auto_fallback = True
			return
		except Exception:
			continue

	if device == "cuda":
		print(f"[warning] {model_name} has no recognized GPU parameter; staying on CPU.")


def _fit_model(name: str, clf, X_train, y_train, X_val, y_val) -> dict[str, float]:
	start = time.time()
	fit_kwargs = dict(getattr(clf, "_fit_kwargs", {}))

	X_train_fit, y_train_fit = X_train, y_train
	X_val_fit, y_val_fit = X_val, y_val
	if name == "xgboost":
		X_train_fit, y_train_fit = _maybe_to_xgb_cuda_inputs(clf, X_train, y_train)
		X_val_fit, y_val_fit = _maybe_to_xgb_cuda_inputs(clf, X_val, y_val)
		if getattr(clf, "_xgb_use_eval_set", False):
			fit_kwargs["eval_set"] = [(X_val_fit, y_val_fit)]

	try:
		clf.fit(X_train_fit, y_train_fit, **fit_kwargs)
	except Exception as exc:
		if getattr(clf, "_auto_fallback", False):
			err_msg = str(exc).splitlines()[0] if str(exc) else repr(exc)
			print(
				f"[warning] {name} CUDA failed in auto mode ({err_msg}). Retrying on CPU."
			)
			fallback_map = {
				"device": "cpu",
				"inference_device": "cpu",
				"accelerator": "cpu",
				"use_cuda": False,
				"gpu": False,
				"gpu_id": -1,
			}
			param_name = getattr(clf, "_device_param_key", None)
			if param_name in fallback_map:
				clf.set_params(**{param_name: fallback_map[param_name]})
			clf._active_device = "cpu"
			clf._auto_fallback = False
			if name == "xgboost" and "eval_set" in fit_kwargs:
				fit_kwargs["eval_set"] = [(X_val, y_val)]
			clf.fit(X_train, y_train, **fit_kwargs)
		else:
			raise
	duration = time.time() - start
	return {"fit_seconds": float(duration)}


def _evaluate_split(name: str, clf, X_eval, y_eval) -> dict[str, float]:
	start = time.time()
	X_eval_fit, y_eval_fit = X_eval, y_eval
	if name == "xgboost":
		X_eval_fit, y_eval_fit = _maybe_to_xgb_cuda_inputs(clf, X_eval, y_eval)
	y_pred = clf.predict(X_eval_fit)
	y_pred = _to_numpy(y_pred)
	y_eval_np = _to_numpy(y_eval_fit)
	duration = time.time() - start

	return {
		"accuracy": float(accuracy_score(y_eval_np, y_pred)),
		"f1_macro": float(f1_score(y_eval_np, y_pred, average="macro", zero_division=0)),
		"eval_seconds": float(duration),
	}


def _build_models(
	seed: int,
	n_estimators: int,
	xgb_progress: bool,
	xgb_verbose_every: int,
):
	models = {
		"tabicl": TabICLClassifier(n_estimators=n_estimators, random_state=seed),
		"decision_tree": DecisionTreeClassifier(random_state=seed),
		"random_forest": RandomForestClassifier(
			n_estimators=100,
			random_state=seed,
			min_samples_split=10,
			n_jobs=-1,
		),
	}

	if XGBClassifier is None:
		print("[warning] xgboost is not installed; skipping xgboost baseline.")
	else:
		xgb_model = XGBClassifier(
			n_estimators=500,
			max_depth=8,
			learning_rate=0.1,
			tree_method="hist",
			device="cuda",
		)
		xgb_model._fit_kwargs = {
			"verbose": xgb_verbose_every if xgb_progress else False,
		}
		xgb_model._xgb_use_eval_set = True
		models["xgboost"] = xgb_model

	return models


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
	data_root = data_dir.parent if data_dir.name == "preprocessed_csvs" else data_dir

	load_dvm_dataset = _import_load_dvm_dataset(dvm_module_path)
	need_images = feature_suite or feature_mode in ("image", "concat")
	train_loader, val_loader, test_loader, metadata = load_dvm_dataset(
		data_dir=str(data_root),
		batch_size=2048,
		num_workers=0,
		#processed_dir="preprocessed_csvs_confirmed_fronts",
		use_images=need_images,
	)

	print("Extracting full train/val/test features from DVM dataset...")
	X_tab_train_full, X_img_train_full, y_train_full = _extract_modalities_from_dataset(train_loader.dataset)
	X_tab_val_full, X_img_val_full, y_val_full = _extract_modalities_from_dataset(val_loader.dataset)
	X_tab_test_full, X_img_test_full, y_test_full = _extract_modalities_from_dataset(test_loader.dataset)

	# Pre-extracted TabICL representations (probing mode)
	probing_mode = tabicl_features_dir is not None
	X_rep_train_full = None
	X_rep_val_full = None
	X_rep_test_full = None
	if probing_mode:
		X_rep_train_full, X_rep_val_full, X_rep_test_full = _load_tabicl_for_splits(
			features_dir=tabicl_features_dir,
			n_train=len(y_train_full),
			n_val=len(y_val_full),
			n_test=len(y_test_full),
		)

	feature_experiments = _resolve_feature_experiments(
		feature_mode=feature_mode,
		image_reducer=image_reducer,
		image_reducer_dim=image_reducer_dim,
		feature_suite=feature_suite,
		suite_reducer_dim=suite_reducer_dim,
	)

	print("Train feature dtype summary:")
	if feature_suite:
		print(f"  feature_suite: enabled (9 configs, reducer dim={suite_reducer_dim})")
	else:
		print(f"  feature_mode: {feature_mode}")
		print(f"  image_reducer: {image_reducer}")
		if image_reducer != "none":
			print(f"  image_reducer_dim: {image_reducer_dim}")
	print(f"  X_tab_train dtype: {X_tab_train_full.dtype}")
	if X_img_train_full is not None:
		print(f"  X_img_train dtype: {X_img_train_full.dtype}")
	print(f"  y_train dtype: {y_train_full.dtype}")
	print()

	val_idx = _sample_indices(y_val_full, max_eval_samples, seed)
	test_idx = _sample_indices(y_test_full, max_eval_samples, seed)

	X_tab_val = X_tab_val_full[val_idx]
	X_tab_test = X_tab_test_full[test_idx]
	y_val = y_val_full[val_idx]
	y_test = y_test_full[test_idx]

	X_img_val = None if X_img_val_full is None else X_img_val_full[val_idx]
	X_img_test = None if X_img_test_full is None else X_img_test_full[test_idx]

	X_rep_val = None if X_rep_val_full is None else X_rep_val_full[val_idx]
	X_rep_test = None if X_rep_test_full is None else X_rep_test_full[test_idx]

	if train_sizes is None or len(train_sizes) == 0:
		train_sizes_to_run = [max_train_samples]
	else:
		train_sizes_to_run = [size for size in train_sizes if size > 0]

	if len(train_sizes_to_run) == 0:
		raise ValueError("No valid train sizes provided. Use positive integers.")

	print("DVM baseline setup")
	print(f"Data dir: {data_root}")
	print(f"Full train tabular shape: {X_tab_train_full.shape}, classes: {len(np.unique(y_train_full))}")
	if X_img_train_full is not None:
		print(f"Full train image shape: {X_img_train_full.shape}")
	print()

	if probing_mode:
		available_model_names = ["linear_probe", "mlp"]
	else:
		available_model_names = ["tabicl", "decision_tree", "random_forest", "xgboost"]
	if methods is None or len(methods) == 0:
		selected_methods = available_model_names
	else:
		selected_methods = methods

	results = {
		"metadata": {
			"data_root": str(data_root),
			"seed": seed,
			"device": device,
			"feature_suite": feature_suite,
			"feature_experiments": feature_experiments,
			"train_sizes": train_sizes_to_run,
			"methods": selected_methods,
			"max_eval_samples": max_eval_samples,
		},
		"experiments": [],
	}

	for feature_cfg in feature_experiments:
		cfg_label = feature_cfg["label"]
		cfg_mode = feature_cfg["feature_mode"]
		cfg_reducer = feature_cfg["image_reducer"]
		cfg_dim = feature_cfg["image_reducer_dim"]

		print(
			f"\n##### Feature experiment: {cfg_label} (mode={cfg_mode}, reducer={cfg_reducer}, dim={cfg_dim}) #####"
		)

		experiment_results = {
			"label": cfg_label,
			"feature_mode": cfg_mode,
			"image_reducer": cfg_reducer,
			"image_reducer_dim": cfg_dim,
			"runs": [],
		}

		for train_size in train_sizes_to_run:
			train_idx = _stratified_train_indices(y_train_full, train_size, seed)
			X_tab_train = X_tab_train_full[train_idx]
			y_train = y_train_full[train_idx]
			X_img_train = None if X_img_train_full is None else X_img_train_full[train_idx]
			X_rep_train = None if X_rep_train_full is None else X_rep_train_full[train_idx]

			X_train, X_val, X_test, fitted_reducer = _build_features_for_mode(
				feature_mode=cfg_mode,
				X_tab_train=X_tab_train,
				X_tab_val=X_tab_val,
				X_tab_test=X_tab_test,
				X_img_train=X_img_train,
				X_img_val=X_img_val,
				X_img_test=X_img_test,
				image_reducer=cfg_reducer,
				image_reducer_dim=cfg_dim,
				seed=seed,
				X_rep_train=X_rep_train,
				X_rep_val=X_rep_val,
				X_rep_test=X_rep_test,
			)

			run_key = "all" if train_size is None else str(train_size)
			print(
				f"\n=== Train size: {run_key} | actual: {len(y_train)} | feature dim: {X_train.shape[1]} ==="
			)
			if fitted_reducer is not None:
				out_dim = getattr(fitted_reducer, "n_components", None)
				print(
					f"[info] Image reducer fitted on train only: {cfg_reducer} (dim={out_dim})"
				)

			models = _build_models(
				seed=seed,
				n_estimators=n_estimators,
				xgb_progress=xgb_progress,
				xgb_verbose_every=xgb_verbose_every,
			)

			run_results = {
				"train_size_requested": None if train_size is None else int(train_size),
				"train_size_actual": int(len(y_train)),
				"models": {},
			}

			for model_name in selected_methods:
				# --- Probing heads (PyTorch baseline_heads) ---
				if probing_mode and model_name in ("linear_probe", "mlp"):
					num_classes = len(np.unique(y_train))
					probe_device = (
						"cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
					)

					if model_name == "linear_probe":
						head = LinearProbe(input_dim=X_train.shape[1], num_classes=num_classes)
						lr = 1e-3
					else:
						head = MLPHead(
							input_dim=X_train.shape[1],
							num_classes=num_classes,
							hidden_dim=256,
						)
						lr = 1e-4

					print(f"[info] {model_name} device: {probe_device}")

					start = time.time()
					head, _ = train_head(
						model=head,
						X_train=X_train,
						y_train=y_train,
						X_val=X_val,
						y_val=y_val,
						num_classes=num_classes,
						learning_rate=lr,
						num_epochs=100,
						batch_size=32,
						early_stopping_patience=10,
						device=probe_device,
						verbose=False,
					)
					fit_metrics = {"fit_seconds": float(time.time() - start)}

					start = time.time()
					val_preds, _ = predict_head(head, X_val, device=probe_device)
					val_metrics = {
						"accuracy": float(accuracy_score(y_val, val_preds)),
						"f1_macro": float(f1_score(y_val, val_preds, average="macro", zero_division=0)),
						"eval_seconds": float(time.time() - start),
					}

					start = time.time()
					test_preds, _ = predict_head(head, X_test, device=probe_device)
					test_metrics = {
						"accuracy": float(accuracy_score(y_test, test_preds)),
						"f1_macro": float(f1_score(y_test, test_preds, average="macro", zero_division=0)),
						"eval_seconds": float(time.time() - start),
					}

					print(f"\n[{model_name}]")
					print(f"  Fit  | time={fit_metrics['fit_seconds']:.2f}s")
					print(
						"  Val  | "
						f"acc={val_metrics['accuracy']:.4f}, "
						f"f1_macro={val_metrics['f1_macro']:.4f}, "
						f"time={val_metrics['eval_seconds']:.2f}s"
					)
					print(
						"  Test | "
						f"acc={test_metrics['accuracy']:.4f}, "
						f"f1_macro={test_metrics['f1_macro']:.4f}, "
						f"time={test_metrics['eval_seconds']:.2f}s"
					)

					run_results["models"][model_name] = {
						"fit": fit_metrics,
						"val": val_metrics,
						"test": test_metrics,
					}
					continue

				# --- Standard sklearn / TabICL models ---
				if model_name not in models:
					print(f"[warning] Requested method '{model_name}' unavailable; skipping.")
					continue

				clf = models[model_name]
				_set_model_device(clf, model_name=model_name, device=device)

				print(f"[info] {model_name} device: {getattr(clf, '_active_device', 'cpu')}")
				fit_metrics = _fit_model(model_name, clf, X_train, y_train, X_val, y_val)
				val_metrics = _evaluate_split(model_name, clf, X_val, y_val)
				test_metrics = _evaluate_split(model_name, clf, X_test, y_test)

				print(f"\n[{model_name}]")
				print(f"  Fit  | time={fit_metrics['fit_seconds']:.2f}s")
				print(
					"  Val  | "
					f"acc={val_metrics['accuracy']:.4f}, "
					f"f1_macro={val_metrics['f1_macro']:.4f}, "
					f"time={val_metrics['eval_seconds']:.2f}s"
				)
				print(
					"  Test | "
					f"acc={test_metrics['accuracy']:.4f}, "
					f"f1_macro={test_metrics['f1_macro']:.4f}, "
					f"time={test_metrics['eval_seconds']:.2f}s"
				)

				run_results["models"][model_name] = {
					"fit": fit_metrics,
					"val": val_metrics,
					"test": test_metrics,
				}

			experiment_results["runs"].append(run_results)

		results["experiments"].append(experiment_results)

	if len(results["experiments"]) == 1:
		results["runs"] = results["experiments"][0]["runs"]

	if results_path is not None:
		results_path.parent.mkdir(parents=True, exist_ok=True)
		with results_path.open("w", encoding="utf-8") as handle:
			json.dump(results, handle, indent=2)
		print(f"\nSaved results to: {results_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Simple DVM tabular baseline experiments")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=DEFAULT_DATA_DIR,
		help="DVM dataset root directory (or preprocessed_csvs directory)",
	)
	parser.add_argument(
		"--dvm-module-path",
		type=Path,
		default=DEFAULT_DVM_MODULE_PATH,
		help="Path to dvm_dataset.py containing load_dvm_dataset",
	)
	parser.add_argument("--seed", type=int, default=0, help="Random seed")
	parser.add_argument(
		"--n-estimators",
		type=int,
		default=1,
		help="Number of estimators for TabICL",
	)
	parser.add_argument(
		"--max-train-samples",
		type=int,
		default=20000,
		help="Stratified cap on training rows (use <=0 for all)",
	)
	parser.add_argument(
		"--train-sizes",
		type=int,
		nargs="+",
		default=None,
		help="One or more training sizes to run (e.g., --train-sizes 500 2000 10000)",
	)
	parser.add_argument(
		"--max-eval-samples",
		type=int,
		default=20000,
		help="Stratified cap on val/test rows (use <=0 for all)",
	)
	parser.add_argument(
		"--methods",
		type=str,
		nargs="+",
		choices=["tabicl", "decision_tree", "random_forest", "xgboost", "linear_probe", "mlp"],
		default=None,
		help="Subset of methods to run (linear_probe and mlp require --tabicl-features-dir)",
	)
	parser.add_argument(
		"--results-path",
		type=Path,
		default=Path("results/dvm_experiments_results.json"),
		help="Path to save JSON results",
	)
	parser.add_argument(
		"--feature-mode",
		type=str,
		choices=["tabular", "image", "concat"],
		default="tabular",
		help="Features used for prediction: tabular only, image only, or concatenated",
	)
	parser.add_argument(
		"--image-reducer",
		type=str,
		choices=["none", "pca", "ica", "random_projection"],
		default="none",
		help="Optional dimensionality reduction applied to image features before image/concat mode",
	)
	parser.add_argument(
		"--image-reducer-dim",
		type=int,
		default=128,
		help="Target dimensionality for image reducer (used when --image-reducer is not none)",
	)
	parser.add_argument(
		"--feature-suite",
		action="store_true",
		help="Run fixed 9-feature comparison suite: tabular, image, image+[pca|ica|random_projection], concat, concat+[pca|ica|random_projection]",
	)
	parser.add_argument(
		"--suite-reducer-dim",
		"--suite-pca-dim",
		dest="suite_reducer_dim",
		type=int,
		default=64,
		help="Reducer dimension used in --feature-suite for image/concat PCA, ICA, and random projection variants (legacy alias: --suite-pca-dim)",
	)
	parser.add_argument(
		"--xgb-progress",
		action="store_true",
		help="Show XGBoost training progress during fitting",
	)
	parser.add_argument(
		"--xgb-verbose-every",
		type=int,
		default=10,
		help="If --xgb-progress is set, print eval metric every N boosting rounds",
	)
	parser.add_argument(
		"--device",
		type=str,
		choices=["auto", "cpu", "cuda"],
		default="auto",
		help="Global device policy for all classifiers: auto tries CUDA then falls back to CPU",
	)
	parser.add_argument(
		"--tabicl-features-dir",
		type=Path,
		default=None,
		help="Path to pre-extracted TabICL representations (enables probing mode with linear_probe/mlp)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	max_train_samples = None if args.max_train_samples <= 0 else args.max_train_samples
	max_eval_samples = None if args.max_eval_samples <= 0 else args.max_eval_samples
	train_sizes = args.train_sizes

	# Auto-resolve features dir when probing methods are requested but no dir given
	probing_methods = {"linear_probe", "mlp"}
	if args.tabicl_features_dir is None and args.methods and probing_methods & set(args.methods):
		args.tabicl_features_dir = DEFAULT_TABICL_FEATURES_DIR
		print(f"[info] --tabicl-features-dir not set; defaulting to {args.tabicl_features_dir}")

	run_experiment(
		data_dir=args.data_dir,
		dvm_module_path=args.dvm_module_path,
		seed=args.seed,
		n_estimators=args.n_estimators,
		max_train_samples=max_train_samples,
		max_eval_samples=max_eval_samples,
		train_sizes=train_sizes,
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
