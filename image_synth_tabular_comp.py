#!/usr/bin/env python
"""PCA-based comparison of synthetic tabular vs. image feature dimensionality.

This script:
    1) generates synthetic tabular features of dimension ``d`` and size ``n``,
    2) loads image features and takes ``n`` rows,
    3) computes how many principal components are needed to explain 99% variance
         for each modality.

Usage:
        python image_synth_tabular_comp.py --num-samples 5000 --feature-dim 768
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tabicl.prior.dataset import PriorDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_DIM = 768          # Default synthetic tabular dimensionality
IMAGE_FEAT_PATH = Path("extracted_features/imagenet22k_dinov3_features.pt")


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def load_image_features(path: Path) -> torch.Tensor:
    """Load image features from a saved tensor.

    Parameters
    ----------
    path : Path
        Path to the saved image features tensor of shape ``(N, D)``.

    Returns
    -------
    torch.Tensor
        Features of shape ``(N, D)``.
    """
    raw: torch.Tensor = torch.load(path, map_location="cpu", weights_only=True)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2-D tensor, got shape {raw.shape}")
    print(f"Loaded image features: {raw.shape}")
    return raw

def pca_components_for_variance_torch(X_np, threshold=0.99, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(X_np).to(device)

    # Center data (important — sklearn does this automatically)
    X = X - X.mean(dim=0, keepdim=True)

    # Compute low-rank PCA (randomized SVD)
    # q can be at most min(n, d)
    #q = min(X.shape)  
    #U, S, V = torch.pca_lowrank(X, q=q)
    
    # 2. Compute SVD
    _U, S, _Vh = torch.linalg.svd(X, full_matrices=False)

    # Compute explained variance ratio
    eigenvalues = (S ** 2) / (X.shape[0] - 1)
    explained = eigenvalues / eigenvalues.sum()
    cumulative = torch.cumsum(explained, dim=0)

    n_components = int((cumulative >= threshold).nonzero()[0].item() + 1)
    achieved = float(cumulative[n_components - 1].item())

    return n_components, achieved

#def pca_components_for_variance(X: np.ndarray, threshold: float = 0.99)  -> tuple[int, float]:
#    """Return number of PCs required to reach *threshold* explained variance."""
#    pca = PCA(n_components=threshold, svd_solver="randomized")
#    pca.fit(X)
#    return pca.n_components_, float(np.sum(pca.explained_variance_ratio_))

def sample_tabular_features(
    num_samples: int,
    feature_dim: int,
    batch_size: int,
) -> torch.Tensor:
    """Generate synthetic tabular feature vectors from PriorDataset.

    Generates exactly one synthetic dataset with sequence length equal to
    ``num_samples`` and returns all rows.

    Parameters
    ----------
    num_samples : int
        Total number of feature vectors to collect.
    feature_dim : int
        Dimensionality of features (sets min/max_features in PriorDataset).
    Returns
    -------
    torch.Tensor
        Collected tabular features of shape ``(batch_size, num_samples, feature_dim)``.
    """
    dataset = PriorDataset(
        min_features=feature_dim,
        max_features=feature_dim,
        batch_size=batch_size,
        #min_seq_len=num_samples,
        max_seq_len=num_samples,
        prior_type="mix_scm",
        device="cpu",
        n_jobs=1,
    )

    print(
        f"Generating synthetic tabular batch with batch_size={batch_size}, "
        f"seq_len={num_samples} …"
    )
    breakpoint()
    X, _y, _d, _seq_lens, _train_sizes = next(iter(dataset))
    # X shape: (batch_size, num_samples, feature_dim)
    return X.cpu()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCA-99 comparison: synthetic tabular vs image features")
    p.add_argument("--num-samples", type=int, default=5000,
                   help="Number of rows per modality (default: 5000)")
    p.add_argument("--image-feat-path", type=Path, default=IMAGE_FEAT_PATH,
                   help="Path to image features .pt file")
    p.add_argument("--feature-dim", type=int, default=768,
                   help="Synthetic tabular dimensionality d (default: 768)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="PriorDataset batch size (number of synthetic datasets to sample)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variance-threshold", type=float, default=0.90,
                   help="Explained variance target for PCA (default: 0.90)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    if args.feature_dim <= 0:
        raise ValueError("--feature-dim must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    # 1) Load image features and keep n rows
    image_feats = load_image_features(args.image_feat_path)

    print(image_feats.shape)
    if image_feats.shape[0] < args.num_samples:
        raise ValueError(
            f"Requested --num-samples={args.num_samples}, but only {image_feats.shape[0]} image rows are available"
        )
    image_feats = image_feats[: args.num_samples]
    image_np = image_feats.numpy().astype(np.float32)

    # 2) Generate synthetic tabular features with dimension d and same n
    tabular_feats = sample_tabular_features(
        num_samples=args.num_samples,
        feature_dim=args.feature_dim,
        batch_size=args.batch_size,
    )

    #breakpoint()

    tabular_np = tabular_feats.numpy().astype(np.float32)

    print("Normalizing features with standard scaling …")
    image_std = image_np.std(0, keepdims=True)
    image_std = np.where(image_std < 1e-8, 1.0, image_std)
    image_np = (image_np - image_np.mean(0, keepdims=True)) / image_std
    
    #tabular_np = normalize_with_skrub(tabular_np)
    #image_np = normalize_with_skrub(image_np)

    # 3) PCA 99%-variance component counts
    print("\nComputing PCA explained variance for tabular features (per batch) …")
    tabular_results = []
    for batch_idx in range(tabular_np.shape[0]):
        batch_X = tabular_np[batch_idx]
        #batch_std = batch_X.std(0, keepdims=True)
        #batch_std = np.where(batch_std < 1e-8, 1.0, batch_std)
        #batch_X = (batch_X - batch_X.mean(0, keepdims=True)) / batch_std
        tab_n, tab_var = pca_components_for_variance_torch(
            batch_X, threshold=args.variance_threshold
        )
        tabular_results.append((tab_n, tab_var))

    print("Computing PCA explained variance for image features …")
    img_n, img_var = pca_components_for_variance_torch(image_np, threshold=args.variance_threshold)

    print("\n=== PCA explained-variance summary ===")
    print(f"Samples per modality: n={args.num_samples}")
    print(f"Tabular batch size: b={args.batch_size}")
    print(f"Synthetic tabular dim: d={args.feature_dim}")
    print(f"Image feature dim: d={image_np.shape[1]}")
    for batch_idx, (tab_n, tab_var) in enumerate(tabular_results):
        print(
            f"Tabular batch {batch_idx}: PCs for {args.variance_threshold:.2%} variance = {tab_n} "
            f"(achieved {tab_var:.4f})"
        )
    print(
        f"Image  : PCs for {args.variance_threshold:.2%} variance = {img_n} "
        f"(achieved {img_var:.4f})"
    )


if __name__ == "__main__":
    main()
