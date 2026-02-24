#!/usr/bin/env python
"""Compare geometric structure of tabular vs image feature datasets.

Metrics:
  - Covariance eigenvalue spectrum
  - Intrinsic dimensionality (participation ratio + MLE estimate)

Example:
  python geometry_compare.py --num-samples 5000 --feature-dim 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from tabicl.prior.dataset import PriorDataset
from tqdm import tqdm


DEFAULT_IMAGE_PATH = Path("extracted_features/imagenet22k_dinov3_features.pt")
DEFAULT_OUT_PATH = Path("outputs/geometry_compare_results.pt")


def load_image_features(path: Path, target_dim: int, seed: int) -> torch.Tensor:
    raw: torch.Tensor = torch.load(path, map_location="cpu", weights_only=True)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2-D tensor, got shape {raw.shape}")

    if raw.shape[1] == target_dim:
        return raw

    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(raw.shape[1], target_dim, generator=gen) / (target_dim ** 0.5)
    return raw @ proj


def load_tabular_features(
    path: Path,
    target_dim: int,
    num_samples: int,
    seed: int,
) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(data, torch.Tensor):
        feats = data
    elif isinstance(data, dict):
        if "train_X" in data:
            feats = data["train_X"]
            if "val_X" in data:
                feats = torch.cat([feats, data["val_X"]], dim=0)
        elif "X" in data:
            feats = data["X"]
        elif "features" in data:
            feats = data["features"]
        else:
            raise ValueError(f"Unsupported tabular dataset keys: {list(data.keys())}")
    else:
        raise ValueError("Unsupported tabular dataset format")

    if feats.ndim == 3:
        bs, seq_len, _ = feats.shape
        rand_idx = torch.randint(0, seq_len, (bs,))
        feats = feats[torch.arange(bs), rand_idx]
    elif feats.ndim != 2:
        raise ValueError(f"Expected 2-D or 3-D tensor, got shape {feats.shape}")

    if feats.shape[1] != target_dim:
        gen = torch.Generator().manual_seed(seed)
        proj = torch.randn(feats.shape[1], target_dim, generator=gen) / (target_dim ** 0.5)
        feats = feats @ proj

    if feats.shape[0] > num_samples:
        feats = feats[:num_samples]

    return feats


def sample_tabular_features(
    num_samples: int,
    feature_dim: int,
    prior_batch_size: int,
    max_seq_len: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    dataset = PriorDataset(
        min_features=feature_dim,
        max_features=feature_dim,
        batch_size=prior_batch_size,
        max_seq_len=max_seq_len,
        prior_type="mix_scm",
        device="cpu",
        n_jobs=1,
    )

    collected: list[torch.Tensor] = []
    remaining = num_samples

    pbar = tqdm(total=num_samples, desc="Tabular samples")
    for X, _y, _d, _seq_lens, _train_sizes in dataset:
        bs = X.shape[0]
        rand_idx = torch.randint(0, X.shape[1], (bs,))
        rows = X[torch.arange(bs), rand_idx]  # (bs, feature_dim)

        take = min(bs, remaining)
        collected.append(rows[:take].cpu())
        remaining -= take
        pbar.update(take)
        if remaining <= 0:
            break

    pbar.close()
    return torch.cat(collected, dim=0)[:num_samples]


def scale_and_normalize(
    X: torch.Tensor,
    standardize: bool = True,
    l2_normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    if standardize:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        X = (X - mean) / std

    if l2_normalize:
        norms = X.norm(dim=1, keepdim=True).clamp_min(eps)
        X = X / norms

    return X


def covariance_eigenspectrum(X: torch.Tensor) -> torch.Tensor:
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.T @ X) / (X.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).flip(0)  # descending
    return eigvals


def participation_ratio(eigvals: torch.Tensor, eps: float = 1e-12) -> float:
    s1 = eigvals.sum().item()
    s2 = (eigvals ** 2).sum().item()
    if s2 < eps:
        return 0.0
    return (s1 * s1) / s2


def intrinsic_dim_mle(X: torch.Tensor, k: int = 10, eps: float = 1e-12) -> float:
    if X.shape[0] <= k:
        return 0.0

    # Pairwise distances (squared) and convert to euclidean
    d2 = torch.cdist(X, X, p=2)
    d2.fill_diagonal_(float("inf"))
    knn, _ = torch.topk(d2, k=k, largest=False)

    rk = knn[:, -1].clamp_min(eps)
    logs = torch.log(rk.unsqueeze(1) / knn.clamp_min(eps))
    id_est = (k - 1) / logs.sum(dim=1)
    return id_est.mean().item()


def summarize_spectrum(eigvals: torch.Tensor) -> Tuple[float, float, float]:
    total = eigvals.sum().item()
    top1 = (eigvals[0].item() / total) if total > 0 else 0.0
    top5 = (eigvals[:5].sum().item() / total) if eigvals.numel() >= 5 and total > 0 else top1
    top10 = (eigvals[:10].sum().item() / total) if eigvals.numel() >= 10 and total > 0 else top5
    return top1, top5, top10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare geometry of image vs tabular features")
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--feature-dim", type=int, default=100)
    p.add_argument("--image-feat-path", type=Path, default=DEFAULT_IMAGE_PATH)
    p.add_argument("--tabular-feat-path", type=Path, default=None,
                   help="Optional path to stored tabular feature dataset")
    p.add_argument("--prior-batch-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--no-l2-norm", action="store_true")
    p.add_argument("--mle-k", type=int, default=10)
    p.add_argument("--save", action="store_true")
    p.add_argument("--save-path", type=Path, default=DEFAULT_OUT_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    image_feats = load_image_features(args.image_feat_path, args.feature_dim, seed=args.seed)
    if image_feats.shape[0] > args.num_samples:
        image_feats = image_feats[: args.num_samples]

    if args.tabular_feat_path is not None:
        tabular_feats = load_tabular_features(
            args.tabular_feat_path,
            target_dim=args.feature_dim,
            num_samples=image_feats.shape[0],
            seed=args.seed,
        )
    else:
        tabular_feats = sample_tabular_features(
            num_samples=image_feats.shape[0],
            feature_dim=args.feature_dim,
            prior_batch_size=args.prior_batch_size,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
        )

    image_feats = scale_and_normalize(
        image_feats,
        standardize=not args.no_standardize,
        l2_normalize=not args.no_l2_norm,
    )
    tabular_feats = scale_and_normalize(
        tabular_feats,
        standardize=not args.no_standardize,
        l2_normalize=not args.no_l2_norm,
    )

    print(f"Image features: {image_feats.shape}")
    print(f"Tabular features: {tabular_feats.shape}")

    img_eigs = covariance_eigenspectrum(image_feats)
    tab_eigs = covariance_eigenspectrum(tabular_feats)

    img_pr = participation_ratio(img_eigs)
    tab_pr = participation_ratio(tab_eigs)

    img_id = intrinsic_dim_mle(image_feats, k=args.mle_k)
    tab_id = intrinsic_dim_mle(tabular_feats, k=args.mle_k)

    img_top1, img_top5, img_top10 = summarize_spectrum(img_eigs)
    tab_top1, tab_top5, tab_top10 = summarize_spectrum(tab_eigs)

    print("\nEigen spectrum summary (fraction of variance):")
    print(f"  Image  top1={img_top1:.4f}  top5={img_top5:.4f}  top10={img_top10:.4f}")
    print(f"  Tab    top1={tab_top1:.4f}  top5={tab_top5:.4f}  top10={tab_top10:.4f}")

    print("\nIntrinsic dimensionality:")
    print(f"  Image  PR={img_pr:.2f}  MLE(k={args.mle_k})={img_id:.2f}")
    print(f"  Tab    PR={tab_pr:.2f}  MLE(k={args.mle_k})={tab_id:.2f}")

    if args.save:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "image_eigvals": img_eigs,
                "tabular_eigvals": tab_eigs,
                "image_participation_ratio": img_pr,
                "tabular_participation_ratio": tab_pr,
                "image_id_mle": img_id,
                "tabular_id_mle": tab_id,
                "image_top1": img_top1,
                "image_top5": img_top5,
                "image_top10": img_top10,
                "tab_top1": tab_top1,
                "tab_top5": tab_top5,
                "tab_top10": tab_top10,
                "feature_dim": args.feature_dim,
                "num_samples": image_feats.shape[0],
                "tabular_feat_path": str(args.tabular_feat_path) if args.tabular_feat_path else None,
                "standardized": not args.no_standardize,
                "l2_normalized": not args.no_l2_norm,
                "mle_k": args.mle_k,
                "seed": args.seed,
            },
            args.save_path,
        )
        print(f"Saved results to: {args.save_path}")


if __name__ == "__main__":
    main()
