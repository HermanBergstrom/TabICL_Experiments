#!/usr/bin/env python
"""Extract DINOv3 features from ImageNet-22k dataset.

This script fetches random images from the timm/imagenet-22k-wds dataset
on Hugging Face and extracts DINOv3 features for use in MLP training.

Example:
  python extract_imagenet22k_features.py --num-samples 5000 --batch-size 64
  python extract_imagenet22k_features.py --num-samples 1000 --device cuda
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import v2
from datasets import load_dataset
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def make_transform(resize_size: int | List[int] = 224) -> v2.Compose:
    """Create image transformation pipeline."""
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


class ImageNet22kDataset(IterableDataset):
    """Wrapper for ImageNet-22k WebDataset from Hugging Face."""
    
    def __init__(
        self,
        num_samples: int,
        transform: v2.Compose,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.transform = transform
        self.seed = seed
        
        # Load the dataset
        print("Loading ImageNet-22k dataset from Hugging Face...")
        self.dataset = load_dataset(
            "timm/imagenet-22k-wds",
            split="train",
            streaming=True,
        )
        
        # Set seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
    
    def __iter__(self):
        """Iterate through random samples."""
        count = 0
        for sample in self.dataset:
            if count >= self.num_samples:
                break
            
            try:
                # Extract image from sample
                image = sample.get("jpg") or sample.get("image")
                if image is None:
                    continue
                
                # Handle different image formats
                if isinstance(image, bytes):
                    image = Image.open(BytesIO(image))
                elif not isinstance(image, Image.Image):
                    continue
                
                # Convert to RGB if necessary
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Apply transforms
                if self.transform:
                    transformed = self.transform(image)
                else:
                    transformed = torch.tensor(image)
                
                yield {"image": transformed, "sample_id": count}
                count += 1
                
            except Exception as e:
                # Skip samples that fail to load
                continue


def extract_features(
    num_samples: int,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    img_size: int,
    repo_dir: Path,
    weights_path: Path,
    device: str,
    seed: int,
) -> None:
    """Extract DINOv3 features from ImageNet-22k dataset."""
    
    transform = make_transform(img_size)
    dataset = ImageNet22kDataset(
        num_samples=num_samples,
        transform=transform,
        seed=seed,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    print(f"Loading DINOv3 model from {repo_dir}...")
    model = torch.hub.load(
        str(repo_dir),
        "dinov3_vitb16",
        source="local",
        weights=str(weights_path),
    )
    model.eval()
    model.to(device)

    image_features: List[torch.Tensor] = []
    processed_count = 0

    use_amp = device.startswith("cuda")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    print(f"Extracting features from {num_samples} images...")
    with torch.inference_mode():
        for batch in tqdm(loader, total=(num_samples + batch_size - 1) // batch_size):
            images = batch["image"].to(device, non_blocking=True)

            with autocast_ctx:
                feats = model(images)

            image_features.append(feats.detach().float().cpu())
            processed_count += len(images)

    # Concatenate all features
    all_features = torch.cat(image_features, dim=0)
    
    # Trim to exact number of samples (in case there's rounding)
    all_features = all_features[:num_samples]

    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_features, output_path)

    print(f"\nFeature extraction complete!")
    print(f"Saved {len(all_features)} feature vectors to: {output_path}")
    print(f"Features shape: {all_features.shape}")
    print(f"Feature dimension: {all_features.shape[1]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 features from ImageNet-22k dataset."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of random samples to extract (default: 5000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extracted_features/imagenet22k_dinov3_features.pt"),
        help="Output path for features",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("../dinov3"),
        help="Path to DINOv3 repository",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
        help="Path to DINOv3 weights",
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
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_features(
        num_samples=args.num_samples,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        repo_dir=args.repo_dir,
        weights_path=args.weights,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
