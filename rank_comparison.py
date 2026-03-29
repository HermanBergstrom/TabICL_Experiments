"""Compare the normalized effective rank of tabular vs. image features.

For each dataset registered in experiments.py, loads the full dataset,
computes the normalized effective rank of:
  - tabular features alone
  - image features alone
  - tabular + image concatenated

Uses the same effective_rank() definition as feature_quality_tab_arena.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Reuse helpers from the existing experiment files.
sys.path.insert(0, str(Path(__file__).parent))

from experiments import (
    DATASET_CONFIGS,
    _import_dataset_loader,
    _extract_modalities_from_dataset,
    _vectorize_tabular_splits,
)
from feature_quality_tab_arena import effective_rank, normalized_effective_rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank_stats(X: np.ndarray, label: str) -> dict:
    """Return a dict with shape, full matrix rank, and normalized effective rank."""
    n, d = X.shape
    eff = effective_rank(X)
    norm_eff = eff / min(n, d)
    mat_rank = int(np.linalg.matrix_rank(X))
    return {
        "modality": label,
        "n_samples": n,
        "n_features": d,
        "matrix_rank": mat_rank,
        "effective_rank": round(eff, 4),
        "normalized_effective_rank": round(norm_eff, 4),
    }


def _collect_all_splits(splits: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Concatenate train/val/test splits into a single (X_tab, X_img) pair.
    Returns raw (pre-vectorized) tabular and image arrays.
    """
    tabs, imgs = [], []
    for _split_name, (X_tab, X_img, _X_text, _y) in splits.items():
        tabs.append(X_tab)
        if X_img is not None:
            imgs.append(X_img)

    X_tab_all = np.concatenate(tabs, axis=0)
    X_img_all = np.concatenate(imgs, axis=0) if imgs else None
    return X_tab_all, X_img_all


def _subsample_aligned(
    X_tab: np.ndarray,
    X_img: np.ndarray | None,
    max_samples: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Randomly subsample the same row indices from both X_tab and X_img so they
    stay aligned.  Only applied when X_tab.shape[0] > max_samples.
    """
    n = X_tab.shape[0]
    if n <= max_samples:
        return X_tab, X_img
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    idx.sort()
    print(f"  Subsampled to {max_samples} rows (from {n}).")
    return X_tab[idx], (X_img[idx] if X_img is not None else None)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def compute_rank_comparison(
    datasets: list[str],
    need_images: bool = True,
    feature_source: str = "dinov3",
    mimic_task: str = "los_classification",
    max_samples: int | None = None,
    img_pca_components: int = 128,
    random_state: int = 42,
    output_csv: Path | None = None,
) -> list[dict]:
    """
    Load each dataset, compute effective rank for tabular, image, and combined
    feature matrices, and return a list of result rows.
    """
    all_rows = []

    for dataset_name in tqdm(datasets, desc="Datasets"):
        cfg = DATASET_CONFIGS[dataset_name]
        print(f"\n=== {dataset_name} ===")

        try:
            load_fn = _import_dataset_loader(cfg["module_path"], cfg["loader_fn"])
        except (FileNotFoundError, ImportError, AttributeError) as e:
            print(f"  Skipping {dataset_name}: could not load module — {e}")
            continue

        # Load using the same dispatch logic as experiments.py.
        try:
            if dataset_name == "dvm":
                data_dir = cfg["data_dir"]
                data_root = data_dir.parent if data_dir.name == "preprocessed_csvs" else data_dir
                train_loader, val_loader, test_loader, _meta = load_fn(
                    data_dir=str(data_root),
                    batch_size=2048,
                    num_workers=0,
                    use_images=need_images,
                )
            elif dataset_name == "petfinder":
                train_loader, val_loader, test_loader, _meta = load_fn(
                    feature_source=feature_source,
                    batch_size=2048,
                    num_workers=0,
                    use_images=need_images,
                    use_text=False,
                )
            elif dataset_name == "paintings":
                data_dir = cfg["data_dir"]
                train_loader, val_loader, test_loader, _meta = load_fn(
                    processed_dir=data_dir / "preprocessed",
                    pt_path=data_dir / "paintings_dinov3_features.pt",
                    batch_size=2048,
                    num_workers=0,
                    use_images=need_images,
                )
            elif dataset_name == "mimic":
                train_loader, val_loader, test_loader, _meta = load_fn(
                    processed_dir=None,
                    csv_path=cfg["data_dir"] / "final_dataframe.csv",
                    use_one_subject_csv=True,
                    task=mimic_task,
                    batch_size=2048,
                    num_workers=0,
                    use_images=need_images,
                    #image_features_root="/project/aip-rahulgk/hermanb/datasets/mimic-cxr-jpg-features-medclip",
                    image_features_root="/project/aip-rahulgk/hermanb/datasets/mimic-cxr-jpg-features-rad-dino-mean",
                    missing_image_features="drop",
                )
            else:
                print(f"  Skipping {dataset_name}: unknown dataset.")
                continue
        except Exception as e:
            print(f"  Skipping {dataset_name}: error during load — {e}")
            continue

        # Extract numpy arrays from each split.
        splits = {}
        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            try:
                X_tab, X_img, X_text, y = _extract_modalities_from_dataset(loader.dataset, task="classification")
                splits[split_name] = (X_tab, X_img, X_text, y)
            except Exception as e:
                print(f"  Warning: could not extract {split_name} split — {e}")

        if not splits:
            print(f"  Skipping {dataset_name}: no splits extracted.")
            continue

        X_tab_raw, X_img_raw = _collect_all_splits(splits)

        # Vectorize tabular features (fit on train, transform all).
        train_tab, val_tab, test_tab = (
            splits["train"][0] if "train" in splits else X_tab_raw[:1],
            splits.get("val", (X_tab_raw[:1], None, None, None))[0],
            splits.get("test", (X_tab_raw[:1], None, None, None))[0],
        )
        try:
            X_train_vec, X_val_vec, X_test_vec = _vectorize_tabular_splits(
                train_tab, val_tab, test_tab
            )
            tab_parts = [X_train_vec]
            if "val" in splits:
                tab_parts.append(X_val_vec)
            if "test" in splits:
                tab_parts.append(X_test_vec)
            X_tab_all = np.concatenate(tab_parts, axis=0).astype(np.float32)
        except Exception as e:
            print(f"  Warning: vectorization failed, using raw tabular features — {e}")
            X_tab_all = X_tab_raw

        # Subsample after vectorization so both modalities use the same rows.
        if max_samples is not None:
            X_tab_all, X_img_raw = _subsample_aligned(
                X_tab_all, X_img_raw, max_samples=max_samples, random_state=random_state
            )

        # Tabular rank.
        print(f"  Tabular: {X_tab_all.shape}")
        row_tab = _rank_stats(X_tab_all, "tabular")
        row_tab["dataset"] = dataset_name
        all_rows.append(row_tab)
        print(
            f"    matrix_rank={row_tab['matrix_rank']}  "
            f"eff_rank={row_tab['effective_rank']}  "
            f"norm_eff_rank={row_tab['normalized_effective_rank']}"
        )

        # Image rank (only if available).
        if X_img_raw is not None:
            print(f"  Image:    {X_img_raw.shape}")
            row_img = _rank_stats(X_img_raw, "image")
            row_img["dataset"] = dataset_name
            all_rows.append(row_img)
            print(
                f"    matrix_rank={row_img['matrix_rank']}  "
                f"eff_rank={row_img['effective_rank']}  "
                f"norm_eff_rank={row_img['normalized_effective_rank']}"
            )

            # Combined rank.
            X_combined = np.concatenate([X_tab_all, X_img_raw], axis=1)
            print(f"  Combined: {X_combined.shape}")
            row_combined = _rank_stats(X_combined, "combined")
            row_combined["dataset"] = dataset_name
            all_rows.append(row_combined)
            print(
                f"    matrix_rank={row_combined['matrix_rank']}  "
                f"eff_rank={row_combined['effective_rank']}  "
                f"norm_eff_rank={row_combined['normalized_effective_rank']}"
            )

            # PCA-reduced image rank (and combined with tabular).
            pca_dim = min(img_pca_components, X_img_raw.shape[0], X_img_raw.shape[1])
            pca = PCA(n_components=pca_dim, random_state=random_state)
            X_img_pca = pca.fit_transform(X_img_raw)

            print(f"  Image (PCA-{pca_dim}): {X_img_pca.shape}")
            row_img_pca = _rank_stats(X_img_pca, f"image_pca{pca_dim}")
            row_img_pca["dataset"] = dataset_name
            all_rows.append(row_img_pca)
            print(
                f"    matrix_rank={row_img_pca['matrix_rank']}  "
                f"eff_rank={row_img_pca['effective_rank']}  "
                f"norm_eff_rank={row_img_pca['normalized_effective_rank']}"
            )

            X_combined_pca = np.concatenate([X_tab_all, X_img_pca], axis=1)
            print(f"  Combined (PCA-{pca_dim}): {X_combined_pca.shape}")
            row_combined_pca = _rank_stats(X_combined_pca, f"combined_pca{pca_dim}")
            row_combined_pca["dataset"] = dataset_name
            all_rows.append(row_combined_pca)
            print(
                f"    matrix_rank={row_combined_pca['matrix_rank']}  "
                f"eff_rank={row_combined_pca['effective_rank']}  "
                f"norm_eff_rank={row_combined_pca['normalized_effective_rank']}"
            )

            # Joint PCA: z-normalize both modalities first, then concatenate and PCA.
            # Z-normalization is essential here so neither modality's raw variance
            # dominates the decomposition.
            X_tab_z = StandardScaler().fit_transform(X_tab_all)
            X_img_z = StandardScaler().fit_transform(X_img_raw)
            X_joint = np.concatenate([X_tab_z, X_img_z], axis=1)
            joint_pca_dim = min(img_pca_components, X_joint.shape[0], X_joint.shape[1])
            X_joint_pca = PCA(n_components=joint_pca_dim, random_state=random_state).fit_transform(X_joint)

            print(f"  Joint PCA-{joint_pca_dim} (tab_z + img_z → PCA): {X_joint_pca.shape}")
            row_joint_pca = _rank_stats(X_joint_pca, f"joint_pca{joint_pca_dim}")
            row_joint_pca["dataset"] = dataset_name
            all_rows.append(row_joint_pca)
            print(
                f"    matrix_rank={row_joint_pca['matrix_rank']}  "
                f"eff_rank={row_joint_pca['effective_rank']}  "
                f"norm_eff_rank={row_joint_pca['normalized_effective_rank']}"
            )
        else:
            print(f"  Image features not available for {dataset_name}; skipping image/combined ranks.")

    if output_csv is not None and all_rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "dataset", "modality", "n_samples", "n_features",
            "matrix_rank", "effective_rank", "normalized_effective_rank",
        ]
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved rank comparison results to: {output_csv}")

    return all_rows


def print_rank_table(rows: list[dict]) -> None:
    header = f"{'dataset':<14} {'modality':<10} {'n_samples':>10} {'n_features':>12} {'mat_rank':>10} {'eff_rank':>10} {'norm_eff_rank':>14}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['dataset']:<14} {row['modality']:<10} "
            f"{row['n_samples']:>10} {row['n_features']:>12} "
            f"{row['matrix_rank']:>10} {row['effective_rank']:>10.4f} "
            f"{row['normalized_effective_rank']:>14.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare normalized effective rank of tabular vs. image features."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
        help="Which datasets to evaluate (default: all).",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip loading image embeddings (tabular rank only).",
    )
    parser.add_argument(
        "--feature-source",
        default="dinov3",
        help="Image feature source for petfinder (default: dinov3).",
    )
    parser.add_argument(
        "--mimic-task",
        default="los_classification",
        help="MIMIC task identifier (default: los_classification).",
    )
    parser.add_argument(
        "--img-pca-components",
        type=int,
        default=128,
        help="Number of PCA components to reduce image embeddings to (default: 128).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Cap the total number of rows used for rank computation. "
            "Rows are randomly subsampled after concatenating all splits. "
            "Useful for large datasets where SVD is slow. Default: no limit."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for subsampling (default: 42).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/rank_comparison.csv"),
        help="Where to save the CSV results.",
    )
    args = parser.parse_args()

    rows = compute_rank_comparison(
        datasets=args.datasets,
        need_images=not args.no_images,
        feature_source=args.feature_source,
        mimic_task=args.mimic_task,
        max_samples=args.max_samples,
        img_pca_components=args.img_pca_components,
        random_state=args.random_state,
        output_csv=args.output_csv,
    )

    print()
    print_rank_table(rows)
