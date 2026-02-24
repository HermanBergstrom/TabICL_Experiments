#!/usr/bin/env python
"""Compare synthetic tabular features vs. image features using a shallow MLP.

This script constructs a binary classification dataset:
  - Label 0: synthetic tabular features from TabICL's PriorDataset
  - Label 1: DINOv3 image features (randomly projected to 100 dims)

A shallow MLP is trained to distinguish between the two modalities and
evaluated on held-out validation data.

Usage:
    python image_synth_tabular_comp.py [--num-samples 5000] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tabicl.prior.dataset import PriorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_DIM = 100          # Target dimensionality for both modalities
IMAGE_FEAT_PATH = Path("extracted_features/imagenet22k_dinov3_features.pt")


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def load_image_features(path: Path, target_dim: int, seed: int = 42) -> torch.Tensor:
    """Load DINOv3 image features and reduce to *target_dim* via random projection.

    Parameters
    ----------
    path : Path
        Path to the saved image features tensor of shape ``(N, D)``.
    target_dim : int
        Desired output dimensionality.
    seed : int
        Random seed for the projection matrix.

    Returns
    -------
    torch.Tensor
        Projected features of shape ``(N, target_dim)``.
    """
    raw: torch.Tensor = torch.load(path, map_location="cpu", weights_only=True)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2-D tensor, got shape {raw.shape}")

    orig_dim = raw.shape[1]
    print(f"Loaded image features: {raw.shape}  (will project {orig_dim} -> {target_dim})")

    # Gaussian random projection (preserves distances up to a constant)
    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(orig_dim, target_dim, generator=gen) / (target_dim ** 0.5)
    projected = raw @ proj
    return projected


def sample_tabular_features(
    num_samples: int,
    feature_dim: int,
    prior_batch_size: int = 8,
    max_seq_len: int = 128,
) -> torch.Tensor:
    """Generate synthetic tabular feature vectors from PriorDataset.

    Since PriorDataset returns multiple rows per dataset we only keep one
    randomly chosen row from each generated dataset.

    Parameters
    ----------
    num_samples : int
        Total number of feature vectors to collect.
    feature_dim : int
        Dimensionality of features (sets min/max_features in PriorDataset).
    prior_batch_size : int
        Batch size passed to PriorDataset (datasets generated per call).
    max_seq_len : int
        Maximum sequence length per generated dataset.

    Returns
    -------
    torch.Tensor
        Collected tabular features of shape ``(num_samples, feature_dim)``.
    """
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

    print(f"Generating {num_samples} synthetic tabular feature vectors …")
    pbar = tqdm(total=num_samples, desc="Tabular samples")
    for X, _y, _d, _seq_lens, _train_sizes in dataset:
        # X shape: (prior_batch_size, seq_len, feature_dim)
        bs = X.shape[0]

        # Pick one random row per dataset
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


def build_dataset(
    image_features: torch.Tensor,
    tabular_features: torch.Tensor,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine both modalities into train / val TensorDatasets.

    Label 0 = tabular, label 1 = image.
    """
    n_img = image_features.shape[0]
    n_tab = tabular_features.shape[0]

    features = torch.cat([tabular_features, image_features], dim=0)  # (N, D)
    labels = torch.cat([
        torch.zeros(n_tab, dtype=torch.long),
        torch.ones(n_img, dtype=torch.long),
    ])

    # Shuffle
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(features.shape[0], generator=gen)
    features = features[perm]
    labels = labels[perm]

    # Split
    val_size = int(features.shape[0] * val_frac)
    train_X, val_X = features[val_size:], features[:val_size]
    train_y, val_y = labels[val_size:], labels[:val_size]

    print(f"Train size: {len(train_X)} | Val size: {len(val_X)}")
    print(f"  Train label distribution: tab={int((train_y == 0).sum())}, img={int((train_y == 1).sum())}")
    print(f"  Val   label distribution: tab={int((val_y == 0).sum())}, img={int((val_y == 1).sum())}")

    return train_X, train_y, val_X, val_y


def _scale_and_normalize_block(
    train_X: torch.Tensor,
    val_X: torch.Tensor,
    standardize: bool = True,
    l2_normalize: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply feature scaling and normalization based on train statistics."""
    if standardize:
        mean = train_X.mean(dim=0, keepdim=True)
        std = train_X.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        train_X = (train_X - mean) / std
        val_X = (val_X - mean) / std

    if l2_normalize:
        train_norm = train_X.norm(dim=1, keepdim=True).clamp_min(eps)
        val_norm = val_X.norm(dim=1, keepdim=True).clamp_min(eps)
        train_X = train_X / train_norm
        val_X = val_X / val_norm

    return train_X, val_X


def scale_and_normalize_modalities(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    standardize: bool = True,
    l2_normalize: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply scaling separately for tabular and image features."""
    train_tab = train_X[train_y == 0]
    train_img = train_X[train_y == 1]
    val_tab = val_X[val_y == 0]
    val_img = val_X[val_y == 1]

    train_tab, val_tab = _scale_and_normalize_block(
        train_tab, val_tab, standardize=standardize, l2_normalize=l2_normalize, eps=eps
    )
    train_img, val_img = _scale_and_normalize_block(
        train_img, val_img, standardize=standardize, l2_normalize=l2_normalize, eps=eps
    )

    train_X = train_X.clone()
    val_X = val_X.clone()
    train_X[train_y == 0] = train_tab
    train_X[train_y == 1] = train_img
    val_X[val_y == 0] = val_tab
    val_X[val_y == 1] = val_img

    return train_X, val_X


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ShallowMLP(nn.Module):
    """Two-hidden-layer MLP for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearClassifier(nn.Module):
    """Linear baseline classifier."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    """Return (loss, accuracy) on the given loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
    label: str = "model",
) -> None:
    """Train the MLP and report metrics each epoch."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"[{label}] Epoch {epoch:3d}/{epochs} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}"
        )


def majority_baseline_accuracy(labels: torch.Tensor) -> float:
    """Return accuracy of the majority-class baseline."""
    if labels.numel() == 0:
        return 0.0
    majority = int((labels == 1).sum() >= (labels == 0).sum())
    return (labels == majority).float().mean().item()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image vs. synthetic tabular feature comparison")
    p.add_argument("--num-samples", type=int, default=5000,
                   help="Number of feature vectors per modality (default: 5000)")
    p.add_argument("--image-feat-path", type=Path, default=IMAGE_FEAT_PATH,
                   help="Path to image features .pt file")
    p.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                   help="Target feature dimensionality (default: 100)")
    p.add_argument("--hidden-dim", type=int, default=128,
                   help="MLP hidden layer width (default: 128)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--linear-lr", type=float, default=1e-2,
                   help="Learning rate for the linear classifier")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-standardize", action="store_true",
                   help="Disable feature standardization")
    p.add_argument("--no-l2-norm", action="store_true",
                   help="Disable per-sample L2 normalization")
    p.add_argument("--save-dataset", action="store_true",
                   help="Save the generated train/val tensors to disk")
    p.add_argument("--save-path", type=Path, default=Path("outputs/image_tabular_dataset.pt"),
                   help="Path to save the dataset when --save-dataset is set")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1. Load & project image features
    image_feats = load_image_features(args.image_feat_path, args.feature_dim, seed=args.seed)

    # Use at most num_samples image features
    if image_feats.shape[0] > args.num_samples:
        image_feats = image_feats[: args.num_samples]
    actual_img = image_feats.shape[0]

    # 2. Generate synthetic tabular features (match count to available images)
    tabular_feats = sample_tabular_features(
        num_samples=actual_img,
        feature_dim=args.feature_dim,
    )

    # 3. Build train / val split
    train_X, train_y, val_X, val_y = build_dataset(
        image_feats, tabular_feats, val_frac=args.val_frac, seed=args.seed,
    )

    train_X, val_X = scale_and_normalize_modalities(
        train_X,
        train_y,
        val_X,
        val_y,
        standardize=not args.no_standardize,
        l2_normalize=not args.no_l2_norm,
    )

    if args.save_dataset:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_X": train_X,
                "train_y": train_y,
                "val_X": val_X,
                "val_y": val_y,
                "feature_dim": args.feature_dim,
                "standardized": not args.no_standardize,
                "l2_normalized": not args.no_l2_norm,
                "seed": args.seed,
            },
            args.save_path,
        )
        print(f"Saved dataset to: {args.save_path}")

    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # 4. Baseline
    baseline_acc = majority_baseline_accuracy(val_y)
    print(f"\nMajority baseline val accuracy: {baseline_acc:.4f}")

    # 5. Train linear classifier
    linear_model = LinearClassifier(input_dim=args.feature_dim)
    print(f"\nLinear classifier:\n{linear_model}\n")
    train(
        linear_model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.linear_lr,
        device=args.device,
        label="linear",
    )
    linear_val_loss, linear_val_acc = evaluate(linear_model, val_loader, args.device)
    print(f"*** Linear final val loss: {linear_val_loss:.4f}  |  val accuracy: {linear_val_acc:.4f} ***")

    # 6. Train MLP
    mlp_model = ShallowMLP(input_dim=args.feature_dim, hidden_dim=args.hidden_dim)
    print(f"\nMLP:\n{mlp_model}\n")
    train(
        mlp_model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        label="mlp",
    )

    # 7. Final evaluation
    val_loss, val_acc = evaluate(mlp_model, val_loader, args.device)
    print(f"*** MLP final val loss: {val_loss:.4f}  |  val accuracy: {val_acc:.4f} ***")


if __name__ == "__main__":
    main()
