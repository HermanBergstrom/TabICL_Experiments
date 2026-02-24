"""
Extract TabICL row representations.
====================================

This script uses TabICLClassifier.predict_proba(return_test_icl_representations=True)
to extract learned row-level feature embeddings and stores them as .pt files.
Supports multiple datasets (DVM, PetFinder, etc.) via a ``--dataset`` flag.

**Val / Test extraction:**
    - The full training set is used to fit TabICL.
    - Val and test rows are processed in chunks (default 10 000) to limit memory.

**Training set extraction (10-fold CV):**
    - The training set is split into 10 stratified folds.
    - For each fold, TabICL is fit on the other 9 folds and representations
      are extracted for the held-out fold.

All representations are saved to:
    <output-dir>/<split>_representations.pt   — dict of {row_index: tensor}
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from tabicl import TabICLClassifier
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset registry  (mirrors experiments.py)
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
        "output_dir": Path(
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
        "output_dir": Path(
            "/home/hermanb/projects/aip-rahulgk/image_icl_project/petfinder/tabiclv2_features"
        ),
    },
}

DATASET_NAMES = list(DATASET_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Helpers copied/adapted from experiments.py
# ---------------------------------------------------------------------------

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


def _extract_tabular_and_targets(dataset):
    """Return (X_tab, y) as numpy arrays from a DVMDataset."""
    tabular_features = []
    targets = []
    for index in tqdm(range(len(dataset)), desc="  reading dataset"):
        item = dataset[index]
        tabular_features.append(np.asarray(item["tabular"], dtype=np.float32))
        targets.append(int(item["target"]))

    X = np.asarray(tabular_features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _extract_representations_in_chunks(
    clf: TabICLClassifier,
    X_test: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Run predict_proba with return_test_icl_representations in chunks.

    Returns
    -------
    representations : np.ndarray
        Shape (n_estimators, n_test, embed_dim) — concatenated across chunks
        along the n_test axis.
    """
    n_test = X_test.shape[0]
    all_reps: list[np.ndarray] = []
    for start in tqdm(range(0, n_test, chunk_size), desc="  extracting chunks"):
        end = min(start + chunk_size, n_test)
        X_chunk = X_test[start:end]
        _proba, reps = clf.predict_proba(X_chunk, return_test_icl_representations = True) # return_test_representations=True)
        all_reps.append(reps)  # (n_estimators, chunk_size, embed_dim)

    return np.concatenate(all_reps, axis=1)


def _save_representations(
    representations: np.ndarray,
    global_indices: np.ndarray,
    output_dir: Path,
    split: str,
) -> Path:
    """Save a dict  {global_row_index: tensor(n_estimators, embed_dim)}  as a single .pt file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}_representations.pt"

    rep_dict: dict[int, torch.Tensor] = {}
    # representations shape: (n_estimators, n_rows, embed_dim)
    for local_idx, global_idx in enumerate(global_indices):
        rep_dict[int(global_idx)] = torch.from_numpy(
            representations[:, local_idx, :].copy()
        )

    torch.save(rep_dict, out_path)
    print(f"  Saved {len(rep_dict)} rows to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main routines
# ---------------------------------------------------------------------------

def extract_val_test(
    clf: TabICLClassifier,
    X_val: np.ndarray,
    X_test: np.ndarray,
    output_dir: Path,
    chunk_size: int,
) -> None:
    """Extract representations for val and test splits (model already fitted)."""
    for split_name, X_split in [("val", X_val), ("test", X_test)]:
        print(f"\n[extract] {split_name}  ({X_split.shape[0]} rows)")
        t0 = time.time()
        reps = _extract_representations_in_chunks(clf, X_split, chunk_size)
        indices = np.arange(X_split.shape[0])
        _save_representations(reps, indices, output_dir, split_name)
        print(f"  done in {time.time() - t0:.1f}s")


def extract_train_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_dir: Path,
    n_folds: int,
    chunk_size: int,
    n_estimators: int,
    seed: int,
) -> None:
    """Extract training-set representations via stratified K-fold CV."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # We'll accumulate all fold results and save once
    all_reps: dict[int, torch.Tensor] = {}

    for fold_idx, (fit_idx, held_out_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n[train fold {fold_idx + 1}/{n_folds}]  "
              f"fit={len(fit_idx)}  held_out={len(held_out_idx)}")

        X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
        X_held = X_train[held_out_idx]

        clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
        t0 = time.time()
        clf.fit(X_fit, y_fit)
        print(f"  fit in {time.time() - t0:.1f}s")

        reps = _extract_representations_in_chunks(clf, X_held, chunk_size)
        # reps shape: (n_estimators, held_out_size, embed_dim)
        for local_idx, global_idx in enumerate(held_out_idx):
            all_reps[int(global_idx)] = torch.from_numpy(
                reps[:, local_idx, :].copy()
            )

    out_path = output_dir / "train_representations.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(all_reps, out_path)
    print(f"\n  Saved {len(all_reps)} train rows to {out_path}")


def _effective_n_folds(y_train: np.ndarray, requested_n_folds: int) -> int:
    """Return a valid stratified K-fold count for *y_train*.

    Caps folds by the minimum per-class count and enforces >= 2.
    """
    if requested_n_folds < 2:
        raise ValueError("--n-folds must be >= 2")

    _classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min()) if len(counts) > 0 else 0
    effective = min(requested_n_folds, min_count)
    if effective < 2:
        raise ValueError(
            "Not enough samples per class for stratified CV train extraction. "
            f"Minimum class count is {min_count}."
        )
    if effective != requested_n_folds:
        print(f"[warning] Capping n_folds: {requested_n_folds} -> {effective}")
    return effective


def _extract_train_cv_array(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int,
    chunk_size: int,
    n_estimators: int,
    seed: int,
) -> np.ndarray:
    """Return train representations as ``(n_rows, embed_dim)`` via stratified CV."""
    effective_n_folds = _effective_n_folds(y_train, n_folds)
    skf = StratifiedKFold(n_splits=effective_n_folds, shuffle=True, random_state=seed)

    # row_idx -> (n_estimators, embed_dim)
    all_reps: dict[int, np.ndarray] = {}

    for fold_idx, (fit_idx, held_out_idx) in enumerate(skf.split(X_train, y_train)):
        print(
            f"\n[train fold {fold_idx + 1}/{effective_n_folds}]  "
            f"fit={len(fit_idx)}  held_out={len(held_out_idx)}"
        )

        X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
        X_held = X_train[held_out_idx]

        clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
        t0 = time.time()
        clf.fit(X_fit, y_fit)
        print(f"  fit in {time.time() - t0:.1f}s")

        reps = _extract_representations_in_chunks(clf, X_held, chunk_size)
        for local_idx, global_idx in enumerate(held_out_idx):
            all_reps[int(global_idx)] = reps[:, local_idx, :].copy()

    # Restore original row order and mean-pool across estimators.
    rows = []
    for idx in range(X_train.shape[0]):
        if idx not in all_reps:
            raise KeyError(f"Missing train representation for row {idx}")
        rows.append(all_reps[idx].mean(axis=0))
    return np.asarray(rows, dtype=np.float32)


def _extract_from_arrays_npz(
    arrays_path: Path,
    output_npz: Path,
    chunk_size: int,
    n_estimators: int,
    n_folds: int,
    seed: int,
) -> None:
    """Extract train/val/test representations from arrays stored in an ``.npz`` file.

    Expected keys:
        - ``X_train`` (N_train, D), ``y_train`` (N_train,)
        - ``X_val``   (N_val, D)
        - ``X_test``  (N_test, D)

    Output keys:
        - ``train_representations`` (N_train, E)
        - ``val_representations``   (N_val, E)
        - ``test_representations``  (N_test, E)
    where E is the representation dimension after mean-pooling across estimators.
    """
    data = np.load(arrays_path)
    required_keys = ["X_train", "y_train", "X_val", "X_test"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {arrays_path}: {missing}")

    X_train = np.asarray(data["X_train"], dtype=np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    X_val = np.asarray(data["X_val"], dtype=np.float32)
    X_test = np.asarray(data["X_test"], dtype=np.float32)

    print("[arrays-mode] Loaded arrays")
    print(f"  train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

    train_reps = _extract_train_cv_array(
        X_train=X_train,
        y_train=y_train,
        n_folds=n_folds,
        chunk_size=chunk_size,
        n_estimators=n_estimators,
        seed=seed,
    )

    print("\n[arrays-mode] Fitting TabICL on full provided train set for val/test reps …")
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"  fit in {time.time() - t0:.1f}s")

    val_reps = _extract_representations_in_chunks(clf, X_val, chunk_size).mean(axis=0).astype(np.float32)
    test_reps = _extract_representations_in_chunks(clf, X_test, chunk_size).mean(axis=0).astype(np.float32)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        train_representations=train_reps,
        val_representations=val_reps,
        test_representations=test_reps,
    )
    print(f"\n[arrays-mode] Saved representations to {output_npz}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TabICL row representations for a supported dataset"
    )
    parser.add_argument(
        "--dataset", type=str, choices=DATASET_NAMES, default="dvm",
        help="Dataset to extract representations for (default: dvm)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Root data directory (default: per-dataset)",
    )
    parser.add_argument(
        "--module-path", "--dvm-module-path", dest="module_path",
        type=Path, default=None,
        help="Path to dataset loader module (default: per-dataset)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write .pt representation files (default: per-dataset)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10000,
        help="Max test rows per predict_proba call (memory control)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=1,
        help="Number of TabICL ensemble estimators",
    )
    parser.add_argument(
        "--n-folds", type=int, default=10,
        help="Number of CV folds for training-set extraction",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+",
        choices=["val", "test", "train"],
        default=["val", "test", "train"],
        help="Which splits to extract (default: all three)",
    )
    parser.add_argument(
        "--from-arrays-npz", type=Path, default=None,
        help=(
            "Optional arrays-mode input .npz. When set, skips dataset loading and "
            "extracts representations from arrays with keys X_train,y_train,X_val,X_test."
        ),
    )
    parser.add_argument(
        "--save-arrays-npz", type=Path, default=None,
        help=(
            "Required with --from-arrays-npz. Output .npz for extracted representations "
            "(train_representations, val_representations, test_representations)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- arrays mode (subprocess-friendly) ----
    if args.from_arrays_npz is not None:
        if args.save_arrays_npz is None:
            raise ValueError("--save-arrays-npz is required when --from-arrays-npz is used")
        _extract_from_arrays_npz(
            arrays_path=args.from_arrays_npz,
            output_npz=args.save_arrays_npz,
            chunk_size=args.chunk_size,
            n_estimators=args.n_estimators,
            n_folds=args.n_folds,
            seed=args.seed,
        )
        print("\nAll done.")
        return

    cfg = DATASET_CONFIGS[args.dataset]

    # Resolve per-dataset defaults for paths not explicitly provided.
    if args.data_dir is None:
        args.data_dir = cfg["data_dir"]
    if args.module_path is None:
        args.module_path = cfg["module_path"]
    if args.output_dir is None:
        args.output_dir = cfg["output_dir"]

    # ---- load dataset ----
    load_fn = _import_dataset_loader(args.module_path, cfg["loader_fn"])

    if args.dataset == "dvm":
        data_root = (
            args.data_dir.parent
            if args.data_dir.name == "preprocessed_csvs"
            else args.data_dir
        )
        train_loader, val_loader, test_loader, _metadata = load_fn(
            data_dir=str(data_root),
            batch_size=2048,
            num_workers=0,
            use_images=False,
        )
    elif args.dataset == "petfinder":
        data_root = args.data_dir
        train_loader, val_loader, test_loader, _metadata = load_fn(
            processed_dir=args.data_dir / "preprocessed",
            pt_path=args.data_dir / "petfinder_dinov3_features.pt",
            batch_size=2048,
            num_workers=0,
            use_images=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print("Extracting tabular features …")
    X_train, y_train = _extract_tabular_and_targets(train_loader.dataset)
    X_val, y_val = _extract_tabular_and_targets(val_loader.dataset)
    X_test, y_test = _extract_tabular_and_targets(test_loader.dataset)

    print(f"  train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

    # ---- val / test extraction (fit once on full train) ----
    if "val" in args.splits or "test" in args.splits:
        print("\nFitting TabICL on full training set …")
        clf = TabICLClassifier(
            n_estimators=args.n_estimators, random_state=args.seed,
        )
        t0 = time.time()
        clf.fit(X_train, y_train)
        print(f"  fit in {time.time() - t0:.1f}s")

        splits_to_extract = []
        if "val" in args.splits:
            splits_to_extract.append(("val", X_val))
        if "test" in args.splits:
            splits_to_extract.append(("test", X_test))

        for split_name, X_split in splits_to_extract:
            print(f"\n[extract] {split_name}  ({X_split.shape[0]} rows)")
            t1 = time.time()
            reps = _extract_representations_in_chunks(clf, X_split, args.chunk_size)
            indices = np.arange(X_split.shape[0])
            _save_representations(reps, indices, args.output_dir, split_name)
            print(f"  done in {time.time() - t1:.1f}s")

    # ---- train extraction (10-fold CV) ----
    if "train" in args.splits:
        print("\nExtracting training-set representations via 10-fold CV …")
        extract_train_cv(
            X_train=X_train,
            y_train=y_train,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            chunk_size=args.chunk_size,
            n_estimators=args.n_estimators,
            seed=args.seed,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
