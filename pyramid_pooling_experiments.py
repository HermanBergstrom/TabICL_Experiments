"""Hierarchical (pyramid) pooling experiments on the butterfly dataset.

Two strategies are supported, selected via ``--pca-first``:

pool-first (default)
    Pool patches spatially at each level, then fit a separate PCA per level
    treating all regions as independent samples (N × level² vectors).

pca-first
    Fit one PCA on all individual training patches (N × P samples of dim D),
    project every patch into the lower-dimensional space, then pool spatially
    at each level.  No per-scale PCA is needed since the patches are already
    projected.

Usage
-----
Pre-extract patch tokens first (see local_embedding_experiments.py), then:

    python pyramid_pooling_experiments.py [--depth 2] [--pooling mean] \\
        [--pca-dim 128] [--pca-first] [--n-train 512]

depth controls the number of pyramid levels:
    depth=1  →  1×1 (global only)
    depth=2  →  1×1, 2×2
    depth=3  →  1×1, 2×2, 4×4
    …
Levels that do not evenly divide the patch grid side-length are skipped with
a warning.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

from local_embedding_experiments import ButterflyPatchDataset, FEATURES_DIR


# ---------------------------------------------------------------------------
# Spatial pooling
# ---------------------------------------------------------------------------

def _spatial_pool(patches: torch.Tensor, level: int, pooling: str) -> torch.Tensor:
    """Pool patch tokens into a level×level grid of spatial regions.

    Args:
        patches: float tensor [N, P, D] where P = H*H (H = sqrt(P)).
        level:   number of divisions per spatial side (1 = global, 2 = 2×2, …).
        pooling: ``"mean"`` or ``"max"``.

    Returns:
        float tensor [N, level², D] – one pooled vector per region per image.

    Raises:
        ValueError: if H is not divisible by ``level``.
    """
    N, P, D = patches.shape
    H = int(P ** 0.5)
    if H * H != P:
        raise ValueError(f"P={P} is not a perfect square; cannot form a spatial grid.")
    if H % level != 0:
        raise ValueError(
            f"Patch grid side H={H} is not divisible by level={level}. "
            "Choose levels that evenly divide H."
        )

    b = H // level                                        # block side length in patches
    spatial = patches.reshape(N, H, H, D)                # [N, H, H, D]
    blocks  = spatial.reshape(N, level, b, level, b, D)  # split rows and cols
    blocks  = blocks.permute(0, 1, 3, 2, 4, 5)           # [N, level, level, b, b, D]
    blocks  = blocks.reshape(N, level * level, b * b, D) # [N, level², b², D]

    if pooling == "mean":
        return blocks.mean(dim=2)   # [N, level², D]
    else:
        return blocks.max(dim=2).values


# ---------------------------------------------------------------------------
# Pyramid pooler (fits + applies PCA at each scale)
# ---------------------------------------------------------------------------

class PyramidPooler:
    """Pyramid spatial pooling with PCA, supporting two orderings.

    pool-first (``pca_first=False``, default)
        Pool at each scale → fit a separate PCA per scale on the pooled
        region vectors (N × level² independent samples).

    pca-first (``pca_first=True``)
        Fit one PCA on all individual training patches (N × P samples).
        Project every patch, then pool spatially at each level.
        No per-scale PCA is needed since patches are already projected.

    Args:
        depth:     Number of pyramid levels (depth=2 → scales [1×1, 2×2]).
        pca_dim:   Target PCA dimensionality (applied per-scale or to patches).
        pooling:   ``"mean"`` or ``"max"`` within spatial regions.
        pca_first: Use the pca-first strategy when ``True``.
        seed:      Random state for PCA.
    """

    def __init__(
        self,
        depth: int = 2,
        pca_dim: int = 128,
        pooling: str = "mean",
        pca_first: bool = False,
        seed: int = 42,
    ) -> None:
        self.requested_levels = [2 ** i for i in range(depth)]
        self.pca_dim = pca_dim
        self.pooling = pooling
        self.pca_first = pca_first
        self.seed = seed
        self.levels: list[int] = []
        self.pcas:   dict[int, PCA] = {}   # keyed by level (pool-first) or "patch" (pca-first)

    # ------------------------------------------------------------------
    def _resolve_levels(self, H: int) -> None:
        self.levels = []
        for lvl in self.requested_levels:
            if H % lvl != 0:
                warnings.warn(
                    f"Skipping level={lvl}: patch grid H={H} not divisible by {lvl}.",
                    stacklevel=3,
                )
            else:
                self.levels.append(lvl)

    # ------------------------------------------------------------------
    def fit_transform(self, features: torch.Tensor) -> np.ndarray:
        """Fit PCA(s) on training features and return projected concatenation.

        Args:
            features: float tensor [N, P, D].

        Returns:
            np.ndarray [N, total_dim].
        """
        N, P, D = features.shape
        H = int(P ** 0.5)
        self._resolve_levels(H)

        if self.pca_first:
            return self._fit_transform_pca_first(features, N, P, D)
        else:
            return self._fit_transform_pool_first(features, N, D)

    def _fit_transform_pool_first(
        self, features: torch.Tensor, N: int, D: int
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        for lvl in self.levels:
            pooled = _spatial_pool(features, lvl, self.pooling)   # [N, lvl², D]
            flat   = pooled.reshape(N * lvl * lvl, D).numpy()
            n_comp = min(self.pca_dim, flat.shape[0], flat.shape[1])
            pca    = PCA(n_components=n_comp, random_state=self.seed)
            proj   = pca.fit_transform(flat)                       # [N×lvl², n_comp]
            self.pcas[lvl] = pca
            parts.append(proj.reshape(N, lvl * lvl * n_comp))
            print(
                f"  [pool-first] level={lvl}×{lvl}: regions/image={lvl**2}  "
                f"PCA({D}→{n_comp})  contrib_dim={lvl**2 * n_comp}"
            )
        return np.concatenate(parts, axis=1)

    def _fit_transform_pca_first(
        self, features: torch.Tensor, N: int, P: int, D: int
    ) -> np.ndarray:
        # Fit one PCA on all individual patches
        flat   = features.reshape(N * P, D).numpy()               # [N×P, D]
        n_comp = min(self.pca_dim, flat.shape[0], flat.shape[1])
        pca    = PCA(n_components=n_comp, random_state=self.seed)
        proj   = pca.fit_transform(flat)                           # [N×P, n_comp]
        self.pcas["patch"] = pca
        print(f"  [pca-first] patch PCA({D}→{n_comp}) fitted on {N*P} patches")

        # Pool in projected space
        proj_t = torch.from_numpy(proj.reshape(N, P, n_comp))     # [N, P, n_comp]
        parts: list[np.ndarray] = []
        for lvl in self.levels:
            pooled = _spatial_pool(proj_t, lvl, self.pooling)     # [N, lvl², n_comp]
            parts.append(pooled.reshape(N, lvl * lvl * n_comp).numpy())
            print(
                f"  [pca-first] level={lvl}×{lvl}: regions/image={lvl**2}  "
                f"contrib_dim={lvl**2 * n_comp}"
            )
        return np.concatenate(parts, axis=1)

    # ------------------------------------------------------------------
    def transform(self, features: torch.Tensor) -> np.ndarray:
        """Project features using the already-fitted PCA(s).

        Args:
            features: float tensor [N, P, D].

        Returns:
            np.ndarray [N, total_dim].
        """
        if not self.pcas:
            raise RuntimeError("Call fit_transform before transform.")
        N, P, _ = features.shape

        if self.pca_first:
            pca    = self.pcas["patch"]
            flat   = features.reshape(N * P, -1).numpy()
            proj   = pca.transform(flat)
            n_comp = proj.shape[1]
            proj_t = torch.from_numpy(proj.reshape(N, P, n_comp))
            parts  = [
                _spatial_pool(proj_t, lvl, self.pooling)
                .reshape(N, lvl * lvl * n_comp).numpy()
                for lvl in self.levels
            ]
        else:
            parts = []
            for lvl in self.levels:
                pooled = _spatial_pool(features, lvl, self.pooling)
                flat   = pooled.reshape(N * lvl * lvl, -1).numpy()
                proj   = self.pcas[lvl].transform(flat)
                n_comp = proj.shape[1]
                parts.append(proj.reshape(N, lvl * lvl * n_comp))

        return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pyramid_baseline(
    features_dir: Path = FEATURES_DIR,
    depth: int = 2,
    pca_dim: int = 128,
    pooling: str = "mean",
    pca_first: bool = False,
    n_estimators: int = 1,
    seed: int = 42,
    n_train: Optional[int] = None,
) -> dict[str, float]:
    """Pyramid pooling → PCA → TabICL (or PCA → pyramid pooling when pca_first=True).

    Args:
        depth:     Number of pyramid levels (1=global only, 2=+2×2, 3=+4×4, …).
        pca_dim:   PCA output dimension (per scale for pool-first; patch-level for pca-first).
        pooling:   ``"mean"`` or ``"max"``.
        pca_first: If True, project patches with a single PCA before pooling.
        n_train:   If set, randomly subsample this many training examples.

    Returns:
        dict with key ``"test_acc"``.
    """
    from tabicl import TabICLClassifier

    train_ds = ButterflyPatchDataset(features_dir, split="train")
    test_ds  = ButterflyPatchDataset(features_dir, split="test")

    train_features = train_ds.features   # [N_train, P, D]
    y_train = train_ds.labels.numpy()

    if n_train is not None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(train_features), size=min(n_train, len(train_features)), replace=False)
        train_features = train_features[idx]
        y_train = y_train[idx]

    strategy = "pca-first" if pca_first else "pool-first"
    print(
        f"\n[pyramid/{strategy}] pooling={pooling}  depth={depth}  "
        f"pca_dim={pca_dim}  n_train={len(train_features)}"
    )

    pooler = PyramidPooler(
        depth=depth, pca_dim=pca_dim, pooling=pooling, pca_first=pca_first, seed=seed
    )

    print(f"[pyramid/{strategy}] Fitting on training set...")
    X_train = pooler.fit_transform(train_features)
    print(f"[pyramid/{strategy}] Final train feature dim: {X_train.shape[1]}")

    print(f"[pyramid/{strategy}] Transforming test set...")
    X_test = pooler.transform(test_ds.features)

    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X_train, y_train)

    test_acc = float(np.mean(clf.predict(X_test) == test_ds.labels.numpy()))
    print(f"[pyramid/{strategy}] test_acc={test_acc:.4f}")
    return {"test_acc": test_acc}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pyramid pooling baseline for butterfly dataset")
    parser.add_argument("--features-dir",  type=Path,  default=FEATURES_DIR)
    parser.add_argument("--depth",         type=int,   default=2,
                        help="Number of pyramid levels (1=global, 2=+2×2, 3=+4×4, …)")
    parser.add_argument("--pooling",       type=str,   default="mean", choices=["mean", "max"])
    parser.add_argument("--pca-dim",       type=int,   default=128,
                        help="PCA output dimension (per-scale for pool-first; patch-level for pca-first)")
    parser.add_argument("--pca-first",     action="store_true",
                        help="Project patches with one PCA before spatial pooling")
    parser.add_argument("--n-estimators",  type=int,   default=1)
    parser.add_argument("--n-train",       type=int,   default=None,
                        help="Subsample training set to this many examples")
    parser.add_argument("--seed",          type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pyramid_baseline(
        features_dir=args.features_dir,
        depth=args.depth,
        pca_dim=args.pca_dim,
        pooling=args.pooling,
        pca_first=args.pca_first,
        n_estimators=args.n_estimators,
        seed=args.seed,
        n_train=args.n_train,
    )
