#!/usr/bin/env python
"""Compute SVD/PCA spectrum statistics and plots for various datasets.

Example:
  python plotting_scripts/svd_analysis.py --dataset imagenet --num-datasets 5
  python plotting_scripts/svd_analysis.py --dataset butterfly --split all
  python plotting_scripts/svd_analysis.py --dataset petfinder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from imagenet_projection.sampling import ShardedEmbeddingSampler, sample_one_dataset
from experiments import DATASET_CONFIGS, _load_data


def compute_spectrum(X: np.ndarray) -> dict:
    X = X.astype(np.float64)
    X_centered = X - X.mean(axis=0, keepdims=True)
    n = X_centered.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to compute covariance eigenvalues.")

    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    eigvals = (s ** 2) / (n - 1)
    total = eigvals.sum()
    explained_ratio = eigvals / total if total > 0 else np.zeros_like(eigvals)
    cumulative = np.cumsum(explained_ratio)

    return {
        "singular_values": s,
        "eigenvalues": eigvals,
        "explained_ratio": explained_ratio,
        "cumulative_ratio": cumulative,
        "n_samples": int(n),
        "n_features": int(X_centered.shape[1]),
    }


def compute_pairwise_centroid_similarities(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all pairs of class centroids."""
    classes = np.unique(y)
    n_classes = len(classes)
    
    if n_classes < 2:
        return np.array([])
        
    centroids = np.zeros((n_classes, X.shape[1]), dtype=np.float64)
    
    for i, cls in enumerate(classes):
        mask = (y == cls)
        centroids[i] = X[mask].mean(axis=0)
        
    # L2 normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    centroids_norm = centroids / norms
    
    similarity_matrix = centroids_norm @ centroids_norm.T
    triu_indices = np.triu_indices(n_classes, k=1)
    return similarity_matrix[triu_indices]


def _load_butterfly_split(features_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = features_dir / f"butterfly_{split}_dinov3_features.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")

    data = torch.load(path, map_location="cpu")
    if "features" not in data or "labels" not in data:
        raise KeyError(f"Missing 'features' or 'labels' in: {path}")

    return data["features"].numpy().astype(np.float64), data["labels"].numpy()


def get_imagenet_data(args) -> list[tuple[str, np.ndarray, np.ndarray]]:
    print(f"Initializing sampler from {args.features_dir} ({args.split} split)...")
    sampler = ShardedEmbeddingSampler(
        output_dir=Path(PROJECT_ROOT / args.features_dir),
        split=args.split,
        seed=args.seed,
    )

    datasets = []
    for i in range(args.num_datasets):
        print(f"Sampling ImageNet dataset {i+1}/{args.num_datasets}...")
        X, y, _ = sample_one_dataset(
            sampler=sampler,
            min_dataset_size=args.min_dataset_size,
            max_dataset_size=args.max_dataset_size,
            min_classes=args.min_classes,
            max_classes=args.max_classes,
            min_per_class=args.min_per_class,
            remap_labels=True,
            allow_replacement=False,
            sampling_distribution="log-uniform",
        )
        datasets.append((f"Sample {i+1}", X, y))
    return datasets


def get_butterfly_data(args) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if args.split == "all":
        X_tr, y_tr = _load_butterfly_split(Path(args.features_dir), "train")
        X_te, y_te = _load_butterfly_split(Path(args.features_dir), "test")
        X = np.concatenate([X_tr, X_te], axis=0)
        y = np.concatenate([y_tr, y_te], axis=0)
    else:
        X, y = _load_butterfly_split(Path(args.features_dir), args.split)
    return [(f"Butterfly ({args.split})", X, y)]


def get_petfinder_data(args) -> list[tuple[str, np.ndarray, np.ndarray]]:
    print("Loading petfinder data...")
    ds_cfg = DATASET_CONFIGS["petfinder"]
    
    _, splits, _ = _load_data(
        dataset="petfinder",
        data_dir=ds_cfg["data_dir"],
        module_path=ds_cfg["module_path"],
        need_images=True,
    )
    
    X_img_train, y_train = splits["train"][1], splits["train"][3]
    X_img_val, y_val = splits["val"][1], splits["val"][3]
    X_img_test, y_test = splits["test"][1], splits["test"][3]
    
    if args.split == "all":
        X = np.concatenate([X_img_train, X_img_val, X_img_test], axis=0)
        y = np.concatenate([y_train, y_val, y_test], axis=0)
    elif args.split == "train":
        X, y = X_img_train, y_train
    elif args.split == "test":
        X, y = X_img_test, y_test
    else:
        X, y = X_img_val, y_val
        
    return [(f"Petfinder ({args.split})", X, y)]


def main():
    parser = argparse.ArgumentParser(description="Compute SVD/PCA spectrum and centroid similarities")
    parser.add_argument("--dataset", type=str, required=True, choices=["imagenet", "butterfly", "petfinder"])
    parser.add_argument("--features-dir", type=str, default="extracted_features")
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/svd_analysis"))
    parser.add_argument("--top-k", type=int, default=768)
    # ImageNet only args
    parser.add_argument("--num-datasets", type=int, default=5)
    parser.add_argument("--min-dataset-size", type=int, default=2000)
    parser.add_argument("--max-dataset-size", type=int, default=6000)
    parser.add_argument("--min-classes", type=int, default=50)
    parser.add_argument("--max-classes", type=int, default=200)
    parser.add_argument("--min-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == "imagenet":
        if args.features_dir == "extracted_features":
            args.features_dir = "extracted_features/imagenet21k_dinov3"
        datasets = get_imagenet_data(args)
    elif args.dataset == "butterfly":
        datasets = get_butterfly_data(args)
    elif args.dataset == "petfinder":
        datasets = get_petfinder_data(args)
        
    spectra = []
    for name, X, y in datasets:
        n_classes = len(np.unique(y))
        print(f"{name} -> Shape: {X.shape}, Classes: {n_classes}")
        
        spec = compute_spectrum(X)
        spec["name"] = name
        spec["n_classes"] = n_classes
        spec["pairwise_similarities"] = compute_pairwise_centroid_similarities(X, y)
        spectra.append(spec)

    k = args.top_k
    prefix = f"{args.dataset}_{args.split}"
    
    # 1. Singular Values (Log)
    fig, ax = plt.subplots(figsize=(8, 5))
    for spec in spectra:
        s = spec["singular_values"]
        safe_s = np.maximum(s[:k], 1e-16)
        n_feat = min(k, len(safe_s))
        label = f"{spec['name']} (N={spec['n_samples']}, C={spec['n_classes']})"
        ax.plot(np.arange(1, n_feat + 1), safe_s, marker="o" if len(spectra)==1 else None, linewidth=1.5, alpha=0.8, markersize=3, label=label)
    
    ax.set_yscale("log")
    ax.set_title(f"Ordered Singular Values - {args.dataset.capitalize()} (Log Scale)")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Singular Value (log)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{prefix}_singular_values_log.png", dpi=200)
    plt.close(fig)

    # 2. Eigenvalues (Scree)
    fig, ax = plt.subplots(figsize=(8, 5))
    for spec in spectra:
        e = spec["eigenvalues"]
        n_feat = min(k, len(e))
        label = f"{spec['name']} (N={spec['n_samples']}, C={spec['n_classes']})"
        ax.plot(np.arange(1, n_feat + 1), e[:k], marker="o" if len(spectra)==1 else None, linewidth=1.5, alpha=0.8, markersize=3, label=label)
    
    ax.set_title(f"Scree Plot - {args.dataset.capitalize()}")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Eigenvalue")
    ax.grid(alpha=0.3)
    ax.legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{prefix}_scree.png", dpi=200)
    plt.close(fig)

    # 2b. Eigenvalues (Scree) - Log Scale
    fig, ax = plt.subplots(figsize=(8, 5))
    for spec in spectra:
        e = spec["eigenvalues"]
        safe_e = np.maximum(e[:k], 1e-16)
        n_feat = min(k, len(safe_e))
        label = f"{spec['name']} (N={spec['n_samples']}, C={spec['n_classes']})"
        ax.plot(np.arange(1, n_feat + 1), safe_e[:n_feat], marker="o" if len(spectra)==1 else None, linewidth=1.5, alpha=0.8, markersize=3, label=label)
    
    ax.set_yscale("log")
    ax.set_title(f"Scree Plot - {args.dataset.capitalize()} (Log Scale)")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Eigenvalue (log)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{prefix}_scree_log.png", dpi=200)
    plt.close(fig)

    # 3. Cumulative Explained Variance
    fig, ax = plt.subplots(figsize=(8, 5))
    for spec in spectra:
        c = spec["cumulative_ratio"]
        label = f"{spec['name']} (N={spec['n_samples']}, C={spec['n_classes']})"
        ax.plot(np.arange(1, len(c) + 1), c, linewidth=1.5, alpha=0.8, label=label)
        
    ax.set_title(f"Cumulative Explained Variance - {args.dataset.capitalize()}")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Ratio")
    ax.set_ylim(0.0, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{prefix}_cumulative_variance.png", dpi=200)
    plt.close(fig)

    # 4. Pairwise Centroid Similarity Distributions
    num_to_plot = len(spectra)
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(8, max(3, 3 * num_to_plot)), squeeze=False)
    for i, spec in enumerate(spectra):
        ax = axes[i, 0]
        sims = spec["pairwise_similarities"]
        label = f"{spec['name']} (N={spec['n_samples']}, C={spec['n_classes']})"
        if len(sims) > 0:
            ax.hist(sims, bins=50, alpha=0.7, color='C%d' % (i % 10))
        ax.set_title(f"Pairwise Centroid Similarity: {label}")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Frequency")
        ax.set_xlim(-1, 1)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{prefix}_centroid_similarity.png", dpi=200)
    plt.close(fig)

    print("\nDataset Summary:")
    for spec in spectra:
        print(f"  {spec['name']}: {spec['n_samples']} samples, {spec['n_classes']} classes")

    print(f"\nSaved plots to {args.output_dir}:")
    print(f"  - {prefix}_singular_values_log.png")
    print(f"  - {prefix}_scree.png")
    print(f"  - {prefix}_scree_log.png")
    print(f"  - {prefix}_cumulative_variance.png")
    print(f"  - {prefix}_centroid_similarity.png")


if __name__ == "__main__":
    main()
