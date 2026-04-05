"""Local embedding (patch-level) experiments with DINOv3 on the butterfly dataset.

Pre-extract patch tokens once (run from the project root):

    python local_embedding_experiments.py extract \\
        [--image-size 224] [--batch-size 32] [--device cuda]

Then run the mean-pool + PCA baseline:

    python local_embedding_experiments.py baseline [--pca-dim 128]

Saved files (in extracted_features/):
    butterfly_train_dinov3_patch_features.pt  – {"features": [N, P, D], "labels": [N]}
    butterfly_test_dinov3_patch_features.pt   – {"features": [N, P, D], "labels": [N]}

where P = (image_size // patch_size)^2  (e.g. 196 for 224px / 16px patches).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_PATH = Path("/project/aip-rahulgk/hermanb/datasets/butterfly-image-classification")
FEATURES_DIR = Path("extracted_features")
DINOV3_REPO   = Path("../dinov3")
DINOV3_WEIGHTS = Path("model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")


# ---------------------------------------------------------------------------
# Image transform
# ---------------------------------------------------------------------------

def make_transform(image_size: int = 224) -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# ---------------------------------------------------------------------------
# Raw image dataset (used only during extraction)
# ---------------------------------------------------------------------------

class _ButterflyImageDataset(Dataset):
    """Reads butterfly images from Training_set.csv, applying a train/test split."""

    def __init__(
        self,
        dataset_path: Path,
        split: str,
        transform: Callable,
        seed: int = 42,
        test_fraction: float = 0.2,
    ) -> None:
        dataset_path = Path(dataset_path)
        csv_path = dataset_path / "Training_set.csv"
        img_dir  = dataset_path / "train"

        rows: list[tuple[str, str]] = []
        all_labels: list[str] = []
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                rows.append((row["filename"], row["label"]))
                all_labels.append(row["label"])

        self.class_to_idx: dict[str, int] = {
            lbl: i for i, lbl in enumerate(sorted(set(all_labels)))
        }

        n_total = len(rows)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n_total)
        n_test = int(n_total * test_fraction)

        if split == "train":
            selected = [rows[i] for i in idx[n_test:]]
        else:
            selected = [rows[i] for i in idx[:n_test]]

        self.samples = [(img_dir / fname, self.class_to_idx[lbl]) for fname, lbl in selected]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_butterfly_patch_features(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = FEATURES_DIR,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "auto",
    seed: int = 42,
    test_fraction: float = 0.2,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Extract DINOv3 patch tokens for the butterfly dataset and save to disk.

    Saves:
        {output_dir}/butterfly_train_dinov3_patch_features.pt
        {output_dir}/butterfly_test_dinov3_patch_features.pt

    Each .pt file contains a dict with:
        "features": float16 tensor of shape [N, P, D]
        "labels":   int64  tensor of shape [N]
    """
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    print(f"[info] Loading DINOv3 on {device_t}...")
    model = torch.hub.load(
        str(DINOV3_REPO), "dinov3_vitb16", source="local", weights=str(DINOV3_WEIGHTS)
    )
    model.eval().to(device_t)

    transform = make_transform(image_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        out_path = output_dir / f"butterfly_{split}_dinov3_patch_features.pt"
        if out_path.exists():
            print(f"[info] {out_path} already exists, skipping.")
            continue

        dataset = _ButterflyImageDataset(
            dataset_path, split, transform, seed=seed, test_fraction=test_fraction
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device_t.type == "cuda"),
        )
        print(f"[info] Extracting {split} ({len(dataset)} images)...")

        all_features: list[torch.Tensor] = []
        all_labels:   list[torch.Tensor] = []

        for images, labels in tqdm(loader, desc=split):
            images = images.to(device_t)
            out    = model.forward_features(images)
            # x_norm_patchtokens: [B, num_patches, embed_dim]
            patches = out["x_norm_patchtokens"].to(dtype=dtype).cpu()
            all_features.append(patches)
            all_labels.append(labels)

        features = torch.cat(all_features, dim=0)   # [N, P, D]
        labels   = torch.cat(all_labels,   dim=0)   # [N]

        torch.save({"features": features, "labels": labels}, out_path)
        print(
            f"[done] Saved {split}: features={tuple(features.shape)}  "
            f"labels={tuple(labels.shape)}  → {out_path}"
        )


# ---------------------------------------------------------------------------
# Dataset over pre-extracted patch features
# ---------------------------------------------------------------------------

class ButterflyPatchDataset(Dataset):
    """Loads pre-extracted DINOv3 patch features for the butterfly dataset.

    Each sample is ``(patches, label)`` where:
        patches : float tensor of shape [num_patches, embed_dim]  (e.g. [196, 768])
        label   : int class index

    Args:
        features_dir: Directory containing the .pt files produced by
            ``extract_butterfly_patch_features``.
        split: ``"train"`` or ``"test"``.
        dtype: Cast features to this dtype on load (default: float32 for model compat).
    """

    def __init__(
        self,
        features_dir: Path = FEATURES_DIR,
        split: str = "train",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        features_dir = Path(features_dir)
        pt_path = features_dir / f"butterfly_{split}_dinov3_patch_features.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Patch features not found at {pt_path}. "
                "Run: python local_embedding_experiments.py extract"
            )

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.features: torch.Tensor = ckpt["features"].to(dtype)   # [N, P, D]
        self.labels:   torch.Tensor = ckpt["labels"].long()        # [N]

        N, P, D = self.features.shape
        print(
            f"[info] ButterflyPatchDataset ({split}): "
            f"N={N}  num_patches={P}  embed_dim={D}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Convenience dataloader factory
# ---------------------------------------------------------------------------

def make_butterfly_patch_dataloaders(
    features_dir: Path = FEATURES_DIR,
    batch_size: int = 64,
    num_workers: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) over pre-extracted patch features.

    Each batch yields ``(patches, labels)`` with shapes:
        patches : [B, num_patches, embed_dim]
        labels  : [B]
    """
    train_ds = ButterflyPatchDataset(features_dir, split="train", dtype=dtype)
    test_ds  = ButterflyPatchDataset(features_dir, split="test",  dtype=dtype)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Baseline: mean-pool → PCA → TabICL
# ---------------------------------------------------------------------------

def _pool(features: torch.Tensor, pooling: str) -> np.ndarray:
    if pooling == "mean":
        return features.mean(dim=1).numpy()
    else:
        return features.max(dim=1).values.numpy()


def run_pool_pca_baseline(
    features_dir: Path = FEATURES_DIR,
    pooling: str = "mean",
    pca_dim: Optional[int] = None,
    pca_first: bool = False,
    n_estimators: int = 1,
    seed: int = 42,
    n_train: Optional[int] = None,
) -> dict[str, float]:
    """Pool patch tokens, optionally apply PCA, evaluate with TabICLClassifier.

    Args:
        pooling:   ``"mean"`` or ``"max"``.
        pca_dim:   PCA output dimension.  ``None`` skips PCA entirely.
        pca_first: When ``True`` (and ``pca_dim`` is set), fit PCA on individual
                   patches [N×P, D] first, then pool in the projected space.
                   Ignored when ``pca_dim`` is ``None``.
        n_train:   If set, randomly subsample this many training examples.

    Returns:
        dict with key "test_acc".
    """
    from sklearn.decomposition import PCA
    from tabicl import TabICLClassifier

    train_ds = ButterflyPatchDataset(features_dir, split="train")
    test_ds  = ButterflyPatchDataset(features_dir, split="test")

    train_features = train_ds.features   # [N, P, D]
    y_train = train_ds.labels.numpy()

    if n_train is not None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(train_features), size=min(n_train, len(train_features)), replace=False)
        train_features = train_features[idx]
        y_train = y_train[idx]

    if pca_dim is None:
        label = f"{pooling}/no-pca"
        X_train = _pool(train_features, pooling)
        X_test  = _pool(test_ds.features, pooling)
        print(f"[{label}] {pooling}-pool (no PCA) → {X_train.shape}")
    elif pca_first:
        label = f"{pooling}/pca-first"
        N, P, D = train_features.shape
        n_comp = min(pca_dim, N * P, D)
        pca = PCA(n_components=n_comp, random_state=seed)
        train_proj = pca.fit_transform(train_features.reshape(N * P, D))   # [N×P, n_comp]
        X_train = _pool(torch.from_numpy(train_proj.reshape(N, P, n_comp)), pooling)
        N_test, _, _ = test_ds.features.shape
        test_proj = pca.transform(test_ds.features.reshape(N_test * P, D))
        X_test = _pool(torch.from_numpy(test_proj.reshape(N_test, P, n_comp)), pooling)
        print(f"[{label}] patch PCA({D}→{n_comp}), then {pooling}-pool → {X_train.shape}")
    else:
        label = f"{pooling}/pool-first"
        X_train = _pool(train_features, pooling)
        X_test  = _pool(test_ds.features, pooling)
        n_comp  = min(pca_dim, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
        print(f"[{label}] {pooling}-pool, then PCA({n_comp}) → {X_train.shape}")

    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X_train, y_train)

    test_acc = float(np.mean(clf.predict(X_test) == test_ds.labels.numpy()))
    print(f"[{label}] test_acc={test_acc:.4f}")
    return {"test_acc": test_acc}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local embedding experiments on butterfly dataset")
    sub = parser.add_subparsers(dest="command")

    ext = sub.add_parser("extract", help="Extract patch features from DINOv3")
    ext.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    ext.add_argument("--output-dir",   type=Path, default=FEATURES_DIR)
    ext.add_argument("--image-size",   type=int,  default=224)
    ext.add_argument("--batch-size",   type=int,  default=5000)
    ext.add_argument("--num-workers",  type=int,  default=4)
    ext.add_argument("--device",       type=str,  default="auto")
    ext.add_argument("--seed",         type=int,  default=42)
    ext.add_argument("--test-fraction",type=float,default=0.2)

    bl = sub.add_parser("baseline", help="Run pool + PCA + TabICL baseline")
    bl.add_argument("--features-dir",  type=Path, default=FEATURES_DIR)
    bl.add_argument("--pooling",       type=str,  default="both",
                    choices=["mean", "max", "both", "pyramid", "attn"])
    bl.add_argument("--depth",         type=int,  default=2,
                    help="Pyramid depth (only used when --pooling pyramid)")
    bl.add_argument("--pca-first",     action="store_true",
                    help="Project patches with one PCA before pooling (only used when --pooling pyramid)")
    bl.add_argument("--pca-dim",       type=int,  default=128)
    bl.add_argument("--n-estimators",  type=int,  default=1)
    bl.add_argument("--n-train",       type=int,  default=None,
                    help="Subsample training set to this many examples")
    bl.add_argument("--seed",          type=int,  default=42)
    # Attention pooling options (only used when --pooling attn)
    bl.add_argument("--num-queries",      type=int,   default=1)
    bl.add_argument("--num-heads",        type=int,   default=8)
    bl.add_argument("--num-steps",        type=int,   default=500)
    bl.add_argument("--lr",               type=float, default=1e-3)
    bl.add_argument("--max-step-samples", type=int,   default=512)
    bl.add_argument("--device",           type=str,   default="auto")
    bl.add_argument("--attn-checkpoint",  type=Path,  default=None,
                    help="Save trained attention head to this .pt file")

    ap = sub.add_parser("attn-pool", help="Train & evaluate learnable attention pooling head")
    ap.add_argument("--features-dir",     type=Path,  default=FEATURES_DIR)
    ap.add_argument("--out-dim",          type=int,   default=None,
                    help="Output dim fed to TabICL (default: embed_dim, no projection)")
    ap.add_argument("--num-queries",      type=int,   default=1,
                    help="Learnable query vectors: 1=CLS-like, >1=multi-view")
    ap.add_argument("--num-heads",        type=int,   default=8)
    ap.add_argument("--dropout",          type=float, default=0.1)
    ap.add_argument("--num-steps",        type=int,   default=500)
    ap.add_argument("--lr",               type=float, default=1e-3)
    ap.add_argument("--max-step-samples", type=int,   default=512)
    ap.add_argument("--n-estimators",     type=int,   default=1)
    ap.add_argument("--n-train",          type=int,   default=None)
    ap.add_argument("--device",           type=str,   default="auto")
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--checkpoint",       type=Path,  default=None,
                    help="Save trained head to this .pt file")

    ae = sub.add_parser("attn-pool-eval", help="Evaluate a saved attention pooling checkpoint")
    ae.add_argument("--features-dir",  type=Path,  default=FEATURES_DIR)
    ae.add_argument("--checkpoint",    type=Path,  required=True)
    ae.add_argument("--n-estimators",  type=int,   default=1)
    ae.add_argument("--n-train",       type=int,   default=None)
    ae.add_argument("--device",        type=str,   default="auto")
    ae.add_argument("--seed",          type=int,   default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.command == "extract":
        extract_butterfly_patch_features(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            test_fraction=args.test_fraction,
        )
    elif args.command == "baseline":
        from pyramid_pooling_experiments import run_pyramid_baseline

        results = {}
        if args.pooling == "pyramid":
            results["pyramid"] = run_pyramid_baseline(
                features_dir=args.features_dir,
                depth=args.depth,
                pca_dim=args.pca_dim,
                pooling="mean",
                pca_first=args.pca_first,
                n_estimators=args.n_estimators,
                seed=args.seed,
                n_train=args.n_train,
            )
        elif args.pooling == "attn":
            from attention_pooling_experiments import run_attn_pool_baseline

            attn_kwargs = dict(
                features_dir=args.features_dir,
                num_queries=args.num_queries,
                num_heads=args.num_heads,
                num_steps=args.num_steps,
                learning_rate=args.lr,
                max_step_samples=args.max_step_samples,
                n_estimators=args.n_estimators,
                n_train=args.n_train,
                device=args.device,
                seed=args.seed,
            )
            results["attn/no-pca"] = run_attn_pool_baseline(
                **attn_kwargs,
                pca_dim=None,
                checkpoint_path=args.attn_checkpoint,
            )
            results["attn/pca"] = run_attn_pool_baseline(
                **attn_kwargs,
                pca_dim=args.pca_dim,
            )
        else:
            poolings = ["mean", "max"] if args.pooling == "both" else [args.pooling]
            strategy = "pca-first" if args.pca_first else "pool-first"
            for p in poolings:
                results[f"{p}/{strategy}"] = run_pool_pca_baseline(
                    features_dir=args.features_dir,
                    pooling=p,
                    pca_dim=args.pca_dim,
                    pca_first=args.pca_first,
                    n_estimators=args.n_estimators,
                    seed=args.seed,
                    n_train=args.n_train,
                )

        print("\n--- summary ---")
        for name, r in results.items():
            print(f"  {name}: test_acc={r['test_acc']:.4f}")
    elif args.command == "attn-pool":
        from attention_pooling_experiments import run_attention_pooling_experiment

        result = run_attention_pooling_experiment(
            features_dir=args.features_dir,
            out_dim=args.out_dim,
            num_queries=args.num_queries,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_steps=args.num_steps,
            learning_rate=args.lr,
            max_step_samples=args.max_step_samples,
            n_estimators=args.n_estimators,
            n_train=args.n_train,
            device=args.device,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
        )
        print(f"\n--- result ---")
        print(f"  attn-pool: test_acc={result['test_acc']:.4f}")
    elif args.command == "attn-pool-eval":
        from attention_pooling_experiments import load_attention_pooling_head, _pool_with_head
        from tabicl import TabICLClassifier

        device_t = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if args.device == "auto" else args.device
        )

        head = load_attention_pooling_head(Path(args.checkpoint), device_t)

        train_ds = ButterflyPatchDataset(args.features_dir, split="train")
        test_ds  = ButterflyPatchDataset(args.features_dir, split="test")

        train_patches = train_ds.features
        y_train = train_ds.labels.numpy()

        if args.n_train is not None:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(train_patches), size=min(args.n_train, len(train_patches)), replace=False)
            train_patches = train_patches[idx]
            y_train = y_train[idx]

        X_train = _pool_with_head(head, train_patches, device_t)
        X_test  = _pool_with_head(head, test_ds.features, device_t)

        clf = TabICLClassifier(n_estimators=args.n_estimators, random_state=args.seed)
        clf.fit(X_train, y_train)
        test_acc = float(np.mean(clf.predict(X_test) == test_ds.labels.numpy()))
        print(f"\n--- result ---")
        print(f"  attn-pool-eval: test_acc={test_acc:.4f}")
    else:
        print("Usage: python local_embedding_experiments.py {extract,baseline,attn-pool,attn-pool-eval} [options]")
