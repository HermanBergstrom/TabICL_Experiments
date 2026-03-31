"""Offline cluster preparation for cluster-based class sampling.

Computes per-class centroids from pre-extracted shard files, then runs multiple
independent K-means clusterings via faiss to produce a cluster file used by
ClusterBasedClassSampler during training.  Run this once before training.

Usage:
    python -m imagenet_projection.prepare_clusters \\
        --data-dir /path/to/imagenet21k_dinov3 \\
        --split all \\
        --output /path/to/imagenet21k_dinov3/clusters.pt \\
        --k 500 \\
        --num-runs 50 \\
        --seed 42

The output file is a torch-saved dict with keys:
    centroids   : FloatTensor (num_classes, feature_dim)
    cluster_runs: list of dicts {cluster_id -> [class_idx, ...]} (len == num_runs)
    k, num_runs, seed, num_classes, feature_dim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm


def compute_class_centroids(data_dir: Path, split: str) -> tuple[np.ndarray, int]:
    """Load all shards and compute the mean feature vector per class.

    Args:
        data_dir: Directory containing shard files and the manifest.
        split   : Manifest split name (e.g. "all", "train").

    Returns:
        centroids  : np.ndarray float32, shape (num_classes, feature_dim)
        num_classes: total number of classes found in the manifest / shards
    """
    manifest_path = data_dir / f"{split}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open() as f:
        manifest = json.load(f)

    shards: list[dict] = manifest["shards"]
    if not shards:
        raise ValueError("Manifest contains no shards")

    # Determine num_classes from class_to_idx if present
    class_to_idx: dict[str, int] = manifest.get("class_to_idx") or {}
    if class_to_idx:
        num_classes = max(class_to_idx.values()) + 1
    else:
        num_classes = None  # will be determined from data

    # Peek at the first shard to get feature_dim
    first_path = data_dir / shards[0]["file"]
    first_shard = torch.load(first_path, map_location="cpu", weights_only=True)
    feature_dim: int = first_shard["features"].shape[1]
    running_max: int = int(first_shard["targets"].max().item())
    del first_shard

    if num_classes is None:
        print("[prepare] Scanning shards to determine num_classes ...", flush=True)
        for shard_meta in tqdm(shards, desc="scan"):
            shard = torch.load(data_dir / shard_meta["file"], map_location="cpu", weights_only=True)
            running_max = max(running_max, int(shard["targets"].max().item()))
        num_classes = running_max + 1

    print(f"[prepare] num_classes={num_classes}  feature_dim={feature_dim}", flush=True)

    class_sum = np.zeros((num_classes, feature_dim), dtype=np.float64)
    class_count = np.zeros(num_classes, dtype=np.int64)

    for shard_meta in tqdm(shards, desc="compute centroids"):
        shard = torch.load(data_dir / shard_meta["file"], map_location="cpu", weights_only=True)
        feats = shard["features"].float().numpy()   # (N, D)
        targets = shard["targets"].long().numpy()   # (N,)

        np.add.at(class_sum, targets, feats)
        np.add.at(class_count, targets, 1)

    valid = class_count > 0
    centroids = np.zeros((num_classes, feature_dim), dtype=np.float32)
    centroids[valid] = (class_sum[valid] / class_count[valid, np.newaxis]).astype(np.float32)

    n_valid = int(valid.sum())
    n_empty = int(num_classes) - n_valid
    print(
        f"[prepare] Centroids computed: {n_valid}/{num_classes} classes present"
        + (f"  ({n_empty} empty — zero centroid)" if n_empty else ""),
        flush=True,
    )
    return centroids, int(num_classes)


def run_kmeans_clusterings(
    centroids: np.ndarray,
    k: int,
    num_runs: int,
    seed: int = 42,
    faiss_verbose: bool = False,
) -> list[dict[int, list[int]]]:
    """Run K-means *num_runs* times and return one cluster map per run.

    Args:
        centroids   : float32 array (num_classes, feature_dim)
        k           : number of clusters per run
        num_runs    : number of independent clusterings
        seed        : base seed; run i uses seed+i
        faiss_verbose: pass to faiss.Kmeans

    Returns:
        List of dicts mapping cluster_id (int) -> list of class indices (int).
    """
    num_classes, feature_dim = centroids.shape
    cluster_runs: list[dict[int, list[int]]] = []

    for run_idx in tqdm(range(num_runs), desc="K-means runs"):
        km = faiss.Kmeans(feature_dim, k, niter=20, verbose=faiss_verbose, seed=seed + run_idx)
        km.train(centroids)

        _, labels = km.index.search(centroids, 1)
        labels_flat = labels.ravel().tolist()

        cluster_map: dict[int, list[int]] = {}
        for class_idx, cluster_id in enumerate(labels_flat):
            cluster_map.setdefault(int(cluster_id), []).append(int(class_idx))

        cluster_runs.append(cluster_map)

        n_clusters = len(cluster_map)
        avg_size = num_classes / n_clusters if n_clusters else 0.0
        print(
            f"  run {run_idx:02d}: {n_clusters} non-empty clusters, "
            f"avg {avg_size:.1f} classes/cluster",
            flush=True,
        )

    return cluster_runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare offline cluster files for cluster-based class sampling.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing shard files and the split manifest.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Manifest split name (e.g. 'all', 'train'). Default: all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the cluster .pt file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=500,
        help="Number of K-means clusters per run. Default: 500.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Number of independent K-means clusterings. Default: 50.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (run i uses seed+i). Default: 42.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--faiss-verbose",
        action="store_true",
        help="Pass verbose=True to faiss.Kmeans (prints per-iteration loss).",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(
            f"[prepare] Output already exists: {args.output}\n"
            "Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(0)

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be > 0")

    centroids, num_classes = compute_class_centroids(args.data_dir, args.split)

    cluster_runs = run_kmeans_clusterings(
        centroids,
        k=args.k,
        num_runs=args.num_runs,
        seed=args.seed,
        faiss_verbose=args.faiss_verbose,
    )

    print(f"[prepare] Saving to {args.output} ...", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "centroids": torch.from_numpy(centroids),
            "cluster_runs": cluster_runs,
            "k": args.k,
            "num_runs": args.num_runs,
            "seed": args.seed,
            "num_classes": num_classes,
            "feature_dim": int(centroids.shape[1]),
        },
        args.output,
    )
    total_mb = args.output.stat().st_size / 1_048_576
    print(
        f"[prepare] Done. {args.num_runs} cluster runs, k={args.k}, "
        f"file size {total_mb:.1f} MB → {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
