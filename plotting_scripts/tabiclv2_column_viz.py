from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from tqdm import tqdm

from tabicl import TabICLClassifier


DEFAULT_DATA_DIR = Path(
	"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset"
)
DEFAULT_DVM_MODULE_PATH = Path(
	"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset/dvm_dataset_with_dinov2.py"
)
DEFAULT_OUTPUT_PATH = Path(
	"/home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/plots/tabiclv2_column_embeddings.png"
)
N_CLS_COLUMNS = 4


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


def _extract_modalities_from_loader(loader, max_samples: int | None, split_name: str):
	tabular_batches = []
	image_batches = []
	target_batches = []
	missing_any_image = False

	n_loaded = 0
	for batch in tqdm(loader, desc=f"Loading {split_name} batches"):
		if not isinstance(batch, dict):
			raise ValueError(
				f"Expected dataloader to yield dict batches for {split_name}, got {type(batch)}"
			)

		tabular = batch.get("tabular")
		image_embedding = batch.get("image_embedding")
		target = batch.get("target")

		if tabular is None:
			raise ValueError(f"{split_name} batch is missing 'tabular' features")
		if target is None:
			raise ValueError(f"{split_name} batch is missing 'target'")

		tabular_np = _to_numpy(tabular).astype(np.float32)
		target_np = _to_numpy(target).astype(np.int64)

		if tabular_np.ndim == 1:
			tabular_np = tabular_np[None, :]
		if target_np.ndim == 0:
			target_np = target_np[None]

		batch_size = tabular_np.shape[0]
		if max_samples is None:
			take = batch_size
		else:
			remaining = max_samples - n_loaded
			if remaining <= 0:
				break
			take = min(batch_size, remaining)

		tabular_batches.append(tabular_np[:take])
		target_batches.append(target_np[:take])

		if image_embedding is None:
			missing_any_image = True
		else:
			image_np = _to_numpy(image_embedding).astype(np.float32)
			if image_np.ndim == 1:
				image_np = image_np[None, :]
			image_batches.append(image_np[:take])

		n_loaded += take
		if max_samples is not None and n_loaded >= max_samples:
			break

	if len(tabular_batches) == 0:
		raise ValueError(f"No samples were loaded for split '{split_name}'")

	X_tab = np.concatenate(tabular_batches, axis=0)
	y = np.concatenate(target_batches, axis=0)

	if missing_any_image or len(image_batches) == 0:
		X_img = None
	else:
		X_img = np.concatenate(image_batches, axis=0)

	print(f"Loaded {split_name} samples: {len(y)}")
	return X_tab, X_img, y


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
):
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


def _to_numpy(value) -> np.ndarray:
	if isinstance(value, np.ndarray):
		return value
	if hasattr(value, "detach") and hasattr(value, "cpu"):
		return value.detach().cpu().numpy()
	return np.asarray(value)


def _resolve_tabular_column_names(dataset, metadata: dict | None, n_tabular_features: int) -> list[str]:
	candidate_keys = [
		"tabular_feature_names",
		"feature_names",
		"tabular_columns",
		"column_names",
	]

	if isinstance(metadata, dict):
		for key in candidate_keys:
			value = metadata.get(key)
			if isinstance(value, (list, tuple)) and len(value) == n_tabular_features:
				return [str(name) for name in value]

	if hasattr(dataset, "data") and hasattr(dataset.data, "columns"):
		columns = [str(col) for col in dataset.data.columns]
		exclude_names = {
			"target",
			"target_encoded",
			"label",
			"class",
			"image_path",
			"split",
		}
		candidate_cols = [col for col in columns if col.lower() not in exclude_names]
		if len(candidate_cols) >= n_tabular_features:
			return candidate_cols[:n_tabular_features]

	return [f"tab_{index}" for index in range(n_tabular_features)]


def _build_concat_column_names(
	tabular_column_names: list[str],
	n_image_columns: int,
	image_reducer: str,
) -> list[str]:
	_ = image_reducer
	image_column_names = [f"img_{index}" for index in range(n_image_columns)]
	return tabular_column_names + image_column_names


def _build_grouped_column_labels(base_column_names: list[str]) -> list[str]:
	n_columns = len(base_column_names)
	if n_columns == 0:
		return []

	grouped_labels = []
	for idx in range(n_columns):
		n0 = base_column_names[idx % n_columns]
		n1 = base_column_names[(idx + 1) % n_columns]
		n3 = base_column_names[(idx + 3) % n_columns]
		grouped_labels.append(f"{n0} | {n1} | {n3}")

	return grouped_labels


def _subset_image_columns_for_projection(
	col_embeddings: np.ndarray,
	column_names: list[str],
	n_image_columns: int,
	image_column_fraction: float,
	seed: int,
):
	if image_column_fraction <= 0 or image_column_fraction > 1:
		raise ValueError("--image-column-fraction must be in the interval (0, 1].")

	n_columns = col_embeddings.shape[0]
	n_image_columns = min(max(0, n_image_columns), n_columns)
	n_tabular_columns = n_columns - n_image_columns

	if n_image_columns == 0 or image_column_fraction >= 1.0:
		return col_embeddings, column_names, n_image_columns

	n_image_keep = max(1, int(np.ceil(n_image_columns * image_column_fraction)))
	if n_image_keep >= n_image_columns:
		return col_embeddings, column_names, n_image_columns

	rng = np.random.default_rng(seed)
	image_local_idx = np.sort(rng.choice(n_image_columns, size=n_image_keep, replace=False))
	selected_idx = np.concatenate(
		[
			np.arange(n_tabular_columns, dtype=np.int64),
			n_tabular_columns + image_local_idx,
		]
	)

	filtered_embeddings = col_embeddings[selected_idx]
	filtered_names = [column_names[int(index)] for index in selected_idx]
	return filtered_embeddings, filtered_names, n_image_keep


def _compute_2d_projections(col_embeddings: np.ndarray, seed: int):
	pca_2d = PCA(n_components=2, random_state=seed).fit_transform(col_embeddings)

	n_columns = col_embeddings.shape[0]
	if n_columns < 3:
		raise ValueError("Need at least 3 columns to run t-SNE.")
	perplexity = min(30, max(2, n_columns // 4))
	if perplexity >= n_columns:
		perplexity = n_columns - 1

	tsne_2d = TSNE(
		n_components=2,
		perplexity=perplexity,
		learning_rate="auto",
		init="pca",
		random_state=seed,
	).fit_transform(col_embeddings)

	return pca_2d, tsne_2d


def _plot_column_embeddings(
	pca_2d: np.ndarray,
	tsne_2d: np.ndarray,
	n_image_columns: int,
	column_names: list[str],
	output_path: Path,
):
	n_columns = pca_2d.shape[0]
	if len(column_names) != n_columns:
		raise ValueError(
			f"column_names length ({len(column_names)}) must match number of columns ({n_columns})"
		)

	n_image_columns = min(max(0, n_image_columns), n_columns)
	n_tabular_columns = n_columns - n_image_columns

	tab_indices = np.arange(0, n_tabular_columns)
	img_indices = np.arange(n_tabular_columns, n_columns)

	fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)

	for ax, projection, title in [
		(axes[0], pca_2d, "PCA (2D)"),
		(axes[1], tsne_2d, "t-SNE (2D)"),
	]:
		if n_image_columns > 0:
			ax.scatter(
				projection[img_indices, 0],
				projection[img_indices, 1],
				s=18,
				alpha=0.55,
				label=f"Image columns (last {n_image_columns})",
				color="tab:orange",
				zorder=2,
			)

		if n_tabular_columns > 0:
			ax.scatter(
				projection[tab_indices, 0],
				projection[tab_indices, 1],
				s=48,
				alpha=0.95,
				label=f"Tabular columns (first {n_tabular_columns})",
				color="tab:blue",
				edgecolors="black",
				linewidths=0.4,
				zorder=3,
			)

		ax.set_title(title)
		ax.set_xlabel("Dim 1")
		ax.set_ylabel("Dim 2")
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best")

		for idx, name in enumerate(column_names):
			x_coord = projection[idx, 0]
			y_coord = projection[idx, 1]
			is_image = idx >= n_tabular_columns
			ax.text(
				x_coord,
				y_coord,
				name,
				fontsize=6 if is_image else 7,
				alpha=0.72 if is_image else 0.9,
				zorder=4,
			)

	fig.suptitle("TabICL column embeddings: tabular vs image columns", fontsize=13)
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=220)
	plt.close(fig)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Visualize TabICL column embeddings on DVM concat features (tabular + image)."
	)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--dvm-module-path", type=Path, default=DEFAULT_DVM_MODULE_PATH)
	parser.add_argument("--batch-size", type=int, default=2048)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--n-estimators", type=int, default=1)
	parser.add_argument(
		"--max-train-samples",
		type=int,
		default=None,
		help="Optional cap on number of training samples loaded from dataloader.",
	)
	parser.add_argument(
		"--max-val-samples",
		type=int,
		default=None,
		help="Optional cap on number of validation samples loaded from dataloader.",
	)
	parser.add_argument(
		"--image-reducer",
		type=str,
		choices=["none", "pca", "random_projection"],
		default="none",
	)
	parser.add_argument("--image-reducer-dim", type=int, default=768)
	parser.add_argument(
		"--image-column-fraction",
		type=float,
		default=1.0,
		help=(
			"Fraction of image columns to keep for PCA/t-SNE plotting after embeddings are computed. "
			"Tabular columns are always kept."
		),
	)
	parser.add_argument(
		"--image-columns",
		type=int,
		default=None,
		help=(
			"Optional manual override for number of trailing image columns in the plot. "
			"If omitted, inferred from concat feature dims after --image-reducer."
		),
	)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	load_dvm_dataset = _import_load_dvm_dataset(args.dvm_module_path)

	data_root = args.data_dir.parent if args.data_dir.name == "preprocessed_csvs" else args.data_dir
	train_loader, val_loader, _, metadata = load_dvm_dataset(
		data_dir=str(data_root),
		batch_size=args.batch_size,
		num_workers=0,
		use_images=True,
	)

	print("Extracting train split features...")
	X_tab_train, X_img_train, y_train = _extract_modalities_from_loader(
		train_loader,
		max_samples=args.max_train_samples,
		split_name="train",
	)
	print("Extracting val split features...")
	X_tab_val, X_img_val, y_val = _extract_modalities_from_loader(
		val_loader,
		max_samples=args.max_val_samples,
		split_name="val",
	)

	X_train, X_val, _, _ = _build_features_for_mode(
		feature_mode="concat",
		X_tab_train=X_tab_train,
		X_tab_val=X_tab_val,
		X_tab_test=X_tab_val,
		X_img_train=X_img_train,
		X_img_val=X_img_val,
		X_img_test=X_img_val,
		image_reducer=args.image_reducer,
		image_reducer_dim=args.image_reducer_dim,
		seed=args.seed,
	)
	n_image_columns_used = X_val.shape[1] - X_tab_val.shape[1]
	if n_image_columns_used < 0:
		raise ValueError(
			"Inferred negative image feature dimension. Check feature construction pipeline."
		)
	if args.image_columns is not None:
		n_image_columns_for_plot = args.image_columns
		print(
			f"Using manual image-columns override for plotting: {n_image_columns_for_plot}"
		)
	else:
		n_image_columns_for_plot = n_image_columns_used
		print(
			"Using inferred image-columns from reducer output for plotting: "
			f"{n_image_columns_for_plot}"
		)

	tabular_column_names = _resolve_tabular_column_names(
		dataset=train_loader.dataset,
		metadata=metadata if isinstance(metadata, dict) else None,
		n_tabular_features=X_tab_train.shape[1],
	)
	column_names = _build_concat_column_names(
		tabular_column_names=tabular_column_names,
		n_image_columns=n_image_columns_for_plot,
		image_reducer=args.image_reducer,
	)

	clf = TabICLClassifier(n_estimators=args.n_estimators, random_state=args.seed, feat_shuffle_method="none")

	

	print(f"Fitting TabICLClassifier on train set with shape={X_train.shape}...")
	clf.fit(X_train, y_train)

	#clf.model_.col_feature_group = False
	print(f"Collecting column embeddings from val set with shape={X_val.shape}...")
	_, col_embeddings = clf.predict_proba(X_val, return_col_embedding_sample=True)
	print("Got the output")
	raw_n_columns = col_embeddings.shape[0]
	if raw_n_columns <= N_CLS_COLUMNS:
		raise ValueError(
			f"Expected more than {N_CLS_COLUMNS} columns, got {raw_n_columns}."
		)
	col_embeddings = col_embeddings[N_CLS_COLUMNS:]
	print(
		f"Dropped first {N_CLS_COLUMNS} CLS columns: {raw_n_columns} -> {col_embeddings.shape[0]} columns"
	)
	if len(column_names) != col_embeddings.shape[0]:
		print(
			"[warning] Column-name count does not match embedding column count after CLS removal. "
			"Falling back to generic names."
		)
		n_tab = col_embeddings.shape[0] - min(max(0, n_image_columns_for_plot), col_embeddings.shape[0])
		column_names = [f"tab_{index}" for index in range(n_tab)] + [
			f"img_{index}" for index in range(col_embeddings.shape[0] - n_tab)
		]

	column_names = _build_grouped_column_labels(column_names)

	col_embeddings, column_names, n_image_columns_for_plot = _subset_image_columns_for_projection(
		col_embeddings=col_embeddings,
		column_names=column_names,
		n_image_columns=n_image_columns_for_plot,
		image_column_fraction=args.image_column_fraction,
		seed=args.seed,
	)
	print(
		"Using columns for projection: "
		f"total={col_embeddings.shape[0]}, image={n_image_columns_for_plot}, "
		f"image_fraction={args.image_column_fraction}"
	)

	if raw_n_columns != X_train.shape[1]:
		print(
			"[warning] Number of returned column embeddings does not match feature dimension: "
			f"{raw_n_columns} vs {X_train.shape[1]}"
		)

	print(f"Column embedding tensor shape: {col_embeddings.shape}")
	pca_2d, tsne_2d = _compute_2d_projections(col_embeddings, seed=args.seed)

	_plot_column_embeddings(
		pca_2d=pca_2d,
		tsne_2d=tsne_2d,
		n_image_columns=n_image_columns_for_plot,
		column_names=column_names,
		output_path=args.output,
	)

	print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
	main()