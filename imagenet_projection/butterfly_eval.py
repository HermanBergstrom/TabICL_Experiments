"""Butterfly dataset loading, feature extraction, and training-time validation.

Loads butterfly DINOv3 features once at startup, then evaluates the current
projection head at each validation step by projecting the features and running
TabICLClassifier.  Mirrors the evaluation done in projection_head_eval.py.

Feature extraction (run once before training)::

    python -m imagenet_projection.butterfly_eval \\
        --data-dir /path/to/butterfly-image-classification \\
        --output-dir /path/to/butterfly-image-classification \\
        --repo-dir /path/to/dinov3_repo \\
        --weights /path/to/dinov3_weights.pth
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from tabicl import TabICLClassifier
    _TABICL_AVAILABLE = True
except ImportError:
    _TABICL_AVAILABLE = False


def load_butterfly_data(features_dir: Path) -> dict[str, np.ndarray]:
    """Load butterfly DINOv3 features from pre-extracted .pt files.

    Args:
        features_dir: Directory containing ``butterfly_train_dinov3_features.pt``
            and ``butterfly_test_dinov3_features.pt``.

    Returns:
        Dict with keys ``X_train``, ``y_train``, ``X_test``, ``y_test``
        (all ``np.ndarray``).
    """
    features_dir = Path(features_dir)
    train_pt = features_dir / "butterfly_train_dinov3_features.pt"
    test_pt  = features_dir / "butterfly_test_dinov3_features.pt"

    for p in (train_pt, test_pt):
        if not p.exists():
            raise FileNotFoundError(f"Butterfly features file not found: {p}")

    train_ckpt = torch.load(train_pt, map_location="cpu", weights_only=True)
    test_ckpt  = torch.load(test_pt,  map_location="cpu", weights_only=True)

    return {
        "X_train": train_ckpt["features"].float().numpy(),
        "y_train": train_ckpt["labels"].numpy(),
        "X_test":  test_ckpt["features"].float().numpy(),
        "y_test":  test_ckpt["labels"].numpy(),
    }


class ButterflyEvaluator:
    """Pre-loads butterfly data, computes PCA/RP baselines once, and evaluates
    the projection head's test accuracy at each val step.

    Usage::

        evaluator = ButterflyEvaluator(features_dir="...", projection_dim=128)
        # baselines available immediately as evaluator.pca_test_acc, .rp_test_acc
        metrics = evaluator.evaluate(head, projection_method="spectral_hypernetwork", device="cuda")
        # metrics: {"test_acc": 0.82}
    """

    def __init__(
        self,
        features_dir: Path,
        projection_dim: int = 128,
        seed: int = 42,
        n_estimators: int = 1,
    ) -> None:
        if not _TABICL_AVAILABLE:
            raise ImportError(
                "tabicl is required for ButterflyEvaluator. "
                "Install it or disable butterfly validation."
            )
        from sklearn.decomposition import PCA
        from sklearn.random_projection import GaussianRandomProjection

        data = load_butterfly_data(features_dir)
        self.X_train: np.ndarray = data["X_train"]
        self.y_train: np.ndarray = data["y_train"]
        self.X_test:  np.ndarray = data["X_test"]
        self.y_test:  np.ndarray = data["y_test"]
        self.n_estimators = n_estimators
        self.seed = seed

        n_train   = len(self.X_train)
        n_test    = len(self.X_test)
        n_classes = int(np.unique(self.y_train).size)
        print(
            f"[info] ButterflyEvaluator: train={n_train}  test={n_test}  "
            f"classes={n_classes}  projection_dim={projection_dim}",
            flush=True,
        )

        # PCA baseline — fit on train, evaluate on test
        n_components = min(projection_dim, self.X_train.shape[0], self.X_train.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(self.X_train)
        X_train_pca = pca.transform(self.X_train)
        X_test_pca  = pca.transform(self.X_test)
        self.pca_test_acc: float = self._accuracy(X_train_pca, self.y_train, X_test_pca, self.y_test)

        # RP baseline — fit on train, evaluate on test
        rp = GaussianRandomProjection(n_components=projection_dim, random_state=seed)
        rp.fit(self.X_train)
        X_train_rp = rp.transform(self.X_train)
        X_test_rp  = rp.transform(self.X_test)
        self.rp_test_acc: float = self._accuracy(X_train_rp, self.y_train, X_test_rp, self.y_test)

        print(
            f"[info] ButterflyEvaluator baselines: "
            f"pca_test_acc={self.pca_test_acc:.4f}  rp_test_acc={self.rp_test_acc:.4f}",
            flush=True,
        )

    def _accuracy(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_ev: np.ndarray,
        y_ev: np.ndarray,
    ) -> float:
        clf = TabICLClassifier(n_estimators=self.n_estimators, random_state=self.seed)
        clf.fit(X_tr, y_tr)
        return float(np.mean(clf.predict(X_ev) == y_ev))

    @torch.no_grad()
    def evaluate(
        self,
        head: torch.nn.Module,
        projection_method: str,
        device: torch.device | str,
    ) -> dict[str, float]:
        """Project butterfly features through *head* and return test accuracy.

        Train and test features are projected together in a single forward pass
        with train as support, matching the protocol in projection_head_eval.py.

        Returns:
            Dict with ``test_acc`` (float, in [0, 1]).
        """
        head.eval()

        X_all = np.concatenate([self.X_train, self.X_test], axis=0)
        all_t = torch.tensor(X_all, dtype=torch.float32, device=device)

        n_train = len(self.X_train)
        support_indices = torch.arange(n_train, device=device)

        if projection_method == "projection_head":
            projected_t = head(features=all_t)
        else:
            support_labels_t = torch.tensor(
                np.concatenate([self.y_train, self.y_test], axis=0),
                dtype=torch.long,
                device=device,
            )
            projected_t = head(
                features=all_t,
                support_indices=support_indices,
                support_labels=support_labels_t,
            )

        if isinstance(projected_t, tuple):
            projected_t = projected_t[0]

        projected = projected_t.cpu().numpy()
        proj_train = projected[:n_train]
        proj_test  = projected[n_train:]

        return {"test_acc": self._accuracy(proj_train, self.y_train, proj_test, self.y_test)}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class _ButterflyDataset(Dataset):
    """Flat-directory butterfly dataset driven by a CSV label file."""

    def __init__(self, img_dir: Path, csv_path: Path, label_to_idx: dict[str, int], transform):
        self.img_dir = img_dir
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_int = label_to_idx[row["label"]]
                self.samples.append((img_dir / row["filename"], label_int))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return {"image": self.transform(img), "label": label, "idx": idx}


def extract_butterfly_features(
    data_dir: Path,
    output_dir: Path,
    repo_dir: Path,
    weights_path: Path,
    test_fraction: float = 0.2,
    seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "auto",
    img_size: int = 224,
    resize_size: int = 256,
) -> None:
    """Extract DINOv3 features from the butterfly training images and save to .pt files.

    Since the Kaggle test set has no labels, the labeled training split is divided
    into a *train* portion (1 - test_fraction) and a *test* portion (test_fraction),
    saved as ``butterfly_train_dinov3_features.pt`` and
    ``butterfly_test_dinov3_features.pt`` respectively.

    Args:
        data_dir     : Root butterfly directory (contains Training_set.csv and train/).
        output_dir   : Where to write the .pt files.
        repo_dir     : Local torch.hub DINOv3 repo directory.
        weights_path : DINOv3 checkpoint (.pth).
        test_fraction: Fraction of labeled data reserved for the test split.
        seed         : Shuffle seed for the train/test split.
        batch_size   : Inference batch size.
        num_workers  : DataLoader workers.
        device       : "auto", "cuda", or "cpu".
        img_size     : Final crop size (224 for ViT-B/16).
        resize_size  : Resize before center-crop (256 is standard).
    """
    from torchvision import transforms
    from .extraction import load_dinov3, extract_model_features, make_transform, resolve_device

    device_t = resolve_device(device)
    transform = make_transform(resize_size, img_size)

    csv_path = data_dir / "Training_set.csv"
    img_dir  = data_dir / "train"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Training image directory not found: {img_dir}")

    # Build deterministic label→int mapping (sorted for reproducibility)
    all_labels: list[str] = []
    rows: list[tuple[str, str]] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["filename"], row["label"]))
            all_labels.append(row["label"])
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted(set(all_labels)))}

    n_total = len(rows)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)
    n_test = int(n_total * test_fraction)
    test_rows  = [rows[i] for i in idx[:n_test]]
    train_rows = [rows[i] for i in idx[n_test:]]
    print(
        f"[butterfly] {n_total} labeled samples → "
        f"train={len(train_rows)}  test={len(test_rows)}  classes={len(label_to_idx)}",
        flush=True,
    )

    model = load_dinov3(repo_dir=repo_dir, weights_path=weights_path, device=device_t)

    def _extract(split_rows: list[tuple[str, str]], desc: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Build a minimal in-memory dataset from the row list
        class _RowDataset(Dataset):
            def __init__(self, rows, img_dir, label_to_idx, transform):
                self.rows = rows
                self.img_dir = img_dir
                self.label_to_idx = label_to_idx
                self.transform = transform
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, i):
                filename, label_str = self.rows[i]
                from PIL import Image
                img = Image.open(self.img_dir / filename).convert("RGB")
                return self.transform(img), self.label_to_idx[label_str]

        loader = DataLoader(
            _RowDataset(split_rows, img_dir, label_to_idx, transform),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device_t.type == "cuda"),
        )
        all_feats: list[torch.Tensor] = []
        all_labels_out: list[torch.Tensor] = []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=desc):
                images = images.to(device_t)
                feats = extract_model_features(model(images))
                all_feats.append(feats.cpu().to(torch.float16))
                all_labels_out.append(labels)
        return torch.cat(all_feats, dim=0), torch.cat(all_labels_out, dim=0)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_feats, train_labels = _extract(train_rows, "extract train")
    torch.save(
        {"features": train_feats, "labels": train_labels},
        output_dir / "butterfly_train_dinov3_features.pt",
    )
    print(f"[butterfly] Saved train features: {train_feats.shape} → {output_dir / 'butterfly_train_dinov3_features.pt'}")

    test_feats, test_labels = _extract(test_rows, "extract test")
    torch.save(
        {"features": test_feats, "labels": test_labels},
        output_dir / "butterfly_test_dinov3_features.pt",
    )
    print(f"[butterfly] Saved test features:  {test_feats.shape} → {output_dir / 'butterfly_test_dinov3_features.pt'}")


def _extraction_main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 features from the butterfly dataset (run once before training)."
    )
    parser.add_argument("--data-dir",   type=Path, required=True,
                        help="Root butterfly directory (contains Training_set.csv and train/).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write .pt files. Defaults to --data-dir.")
    parser.add_argument("--repo-dir",   type=Path, default=Path("../dinov3"),
                        help="Local torch.hub DINOv3 repo directory. Default: ../dinov3")
    parser.add_argument("--weights",    type=Path,
                        default=Path("model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
                        help="DINOv3 checkpoint (.pth).")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction of labeled data for the test split. Default: 0.2.")
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--batch-size", type=int,  default=128)
    parser.add_argument("--num-workers",type=int,  default=4)
    parser.add_argument("--device",     type=str,  default="auto")
    parser.add_argument("--img-size",   type=int,  default=224)
    parser.add_argument("--resize-size",type=int,  default=256)
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    extract_butterfly_features(
        data_dir=args.data_dir,
        output_dir=output_dir,
        repo_dir=args.repo_dir,
        weights_path=args.weights,
        test_fraction=args.test_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        img_size=args.img_size,
        resize_size=args.resize_size,
    )


if __name__ == "__main__":
    _extraction_main()
