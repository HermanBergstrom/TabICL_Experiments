#!/usr/bin/env python
"""Extract DINOv3 features for the butterfly image classification dataset.

Example:
  python extract_butterfly_features.py --split train --batch-size 64
  python extract_butterfly_features.py --split test --batch-size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

# Allow imports from repository root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_transform(resize_size: int = 768) -> v2.Compose:
    """Create image transformation pipeline for DINOv3."""
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


class ButterflyDataset(Dataset):
    """Dataset for butterfly classification images."""

    def __init__(
        self,
        dataset_path: Path,
        split: str = "train",
        transform=None,
        seed: int = 42,
        test_fraction: float = 0.2,
    ):
        """Initialize butterfly dataset.
        
        Args:
            dataset_path: Path to butterfly dataset root
            split: "train" or "test" - split is created from the Training_set.csv
            transform: Image transformation pipeline
            seed: Random seed for train/test split
            test_fraction: Fraction of training data to use for test split
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform

        # Load training CSV (the only one with labels)
        csv_path = self.dataset_path / "Training_set.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        full_df = pd.read_csv(csv_path)
        self.image_dir = self.dataset_path / "train"

        # Build class to index mapping
        self.classes = sorted(full_df["label"].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Split training data into train/test
        n_total = len(full_df)
        n_test = int(n_total * test_fraction)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_total)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        if split == "train":
            self.df = full_df.iloc[train_indices].reset_index(drop=True)
        else:  # split == "test"
            self.df = full_df.iloc[test_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["filename"]
        image_path = self.image_dir / image_name
        label = self.class_to_idx[row["label"]]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (768, 768))

        if self.transform:
            image = self.transform(image)

        return image, label, str(image_path)


def load_dinov3_model(device: torch.device) -> torch.nn.Module:
    """Load DINOv3 model via torch.hub."""
    print("Loading DINOv3 model...")
    model = torch.hub.load("../dinov3", "dinov3_vitb16", source="local", weights_path="model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Extract features from all images in the dataloader."""
    all_features = []
    all_labels = []
    all_paths = []

    print("Extracting features...")
    for images, labels, paths in tqdm(dataloader, desc="Batches"):
        images = images.to(device)

        # Get feature dict from DINOv3
        feature_dict = model.forward_features(images)
        features = feature_dict["x_norm_clstoken"]  # [B, 384] for ViT-S/14

        all_features.append(features.cpu())
        all_labels.append(labels)
        all_paths.extend(paths)

    features = torch.cat(all_features, dim=0)  # [N, 384]
    labels = torch.cat(all_labels, dim=0)  # [N]

    return features, labels, all_paths


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv3 features for butterfly dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/project/aip-rahulgk/hermanb/datasets/butterfly-image-classification",
        help="Path to butterfly dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Dataset split to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/extracted_features",
        help="Output directory for .pt files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of training data to use for test split",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset and dataloader
    transform = make_transform()
    dataset = ButterflyDataset(
        dataset_path=args.dataset_path,
        split=args.split,
        transform=transform,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    # Load model and extract features
    model = load_dinov3_model(device)
    features, labels, paths = extract_features(model, dataloader, device)

    # Save to disk
    output_file = output_dir / f"butterfly_{args.split}_dinov3_features.pt"
    torch.save(
        {
            "features": features,
            "labels": labels,
            "paths": paths,
            "classes": dataset.classes,
            "class_to_idx": dataset.class_to_idx,
        },
        output_file,
    )

    print(f"\nSaved features to {output_file}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
