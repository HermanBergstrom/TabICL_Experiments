#!/usr/bin/env python
"""Extract Dinov3 image features for multimodal datasets and save for TabICL.

Example:
  python extract_features.py --dataset petfinder --split train --batch-size 64
  python extract_features.py --dataset covid19 --batch-size 64
  python extract_features.py --dataset paintings --batch-size 64
  python extract_features.py --dataset skin-cancer --batch-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from multimodal_datasets import (
    PetFinderDataset,
    COVID19ChestXrayDataset,
    PaintingsPricePredictionDataset,
    SkinCancerDataset,
)
from tqdm import tqdm


def make_transform(resize_size: int | List[int] = 768) -> v2.Compose:
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


class IndexedDataset(Dataset):
    """Wraps any base dataset and adds index and ID information."""
    
    def __init__(self, base: Dataset, dataset_name: str):
        self.base = base
        self.dataset_name = dataset_name

    def __len__(self) -> int:  # noqa: D401
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        item["index"] = idx
        
        # Extract ID based on dataset type
        if self.dataset_name == "petfinder":
            item["sample_id"] = self.base.df.iloc[idx]["PetID"]
        elif self.dataset_name == "covid19":
            item["sample_id"] = self.base.df.iloc[idx]["filename"]
        elif self.dataset_name == "paintings":
            item["sample_id"] = self.base.df.iloc[idx]["image_url"].split('/')[-1]
        elif self.dataset_name == "skin-cancer":
            item["sample_id"] = self.base.df.iloc[idx]["img_id"]
        else:
            item["sample_id"] = f"sample_{idx}"
        
        return item


def collate_multimodal(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """General collate function for multimodal datasets."""
    batch = [b for b in batch if b.get("image") is not None]
    if not batch:
        return None

    images = torch.stack([b["image"] for b in batch])
    
    # Handle tabular features if present
    tab_features = None
    if "features" in batch[0]:
        tab_features = torch.stack([b["features"] for b in batch])
    
    # Handle targets
    targets = None
    if "target" in batch[0]:
        targets = torch.stack([b["target"] for b in batch])

    indices = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    sample_ids = [b["sample_id"] for b in batch]

    result = {
        "image": images,
        "indices": indices,
        "sample_ids": sample_ids,
    }
    
    if tab_features is not None:
        result["features"] = tab_features
    if targets is not None:
        result["targets"] = targets
    
    return result


def get_dataset(
    dataset_name: str,
    data_dir: Path,
    split: Optional[str],
    transform: v2.Compose,
) -> Dataset:
    """Get the appropriate dataset based on name."""
    if dataset_name == "petfinder":
        if split is None:
            split = "train"
        return PetFinderDataset(
            data_dir=str(data_dir),
            split=split,
            image_transform=transform,
            return_image=True,
        )
    elif dataset_name == "covid19":
        return COVID19ChestXrayDataset(
            data_dir=str(data_dir),
            image_transform=transform,
        )
    elif dataset_name == "paintings":
        return PaintingsPricePredictionDataset(
            data_dir=str(data_dir),
            image_transform=transform,
            return_image=True,
        )
    elif dataset_name == "skin-cancer":
        return SkinCancerDataset(
            data_dir=str(data_dir),
            image_transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def extract_features(
    dataset_name: str,
    data_dir: Path,
    split: Optional[str],
    output_path: Path,
    batch_size: int,
    num_workers: int,
    img_size: int,
    repo_dir: Path,
    weights_path: Path,
    device: str,
) -> None:
    transform = make_transform(img_size)
    base_dataset = get_dataset(dataset_name, data_dir, split, transform)
    dataset = IndexedDataset(base_dataset, dataset_name)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
        collate_fn=collate_multimodal,
    )

    model = torch.hub.load(str(repo_dir), "dinov3_vitb16", source="local", weights=str(weights_path))
    model.eval()
    model.to(device)

    image_features: List[torch.Tensor] = []
    tabular_features: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    indices_list: List[torch.Tensor] = []
    sample_ids_list: List[str] = []

    use_amp = device.startswith("cuda")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    with torch.inference_mode():
        for batch in tqdm(loader):
            if batch is None:
                continue
            images = batch["image"].to(device, non_blocking=True)

            with autocast_ctx:
                feats = model(images)

            image_features.append(feats.detach().float().cpu())
            indices_list.append(batch["indices"].detach().cpu())
            sample_ids_list.extend(batch["sample_ids"])
            
            # Tabular features are optional
            if "features" in batch:
                tabular_features.append(batch["features"].detach().float().cpu())
            
            # Targets may not exist for test splits
            if "targets" in batch and batch["targets"] is not None:
                targets_list.append(batch["targets"].detach().cpu())

    output = {
        "image_features": torch.cat(image_features, dim=0),
        "indices": torch.cat(indices_list, dim=0),
        "sample_ids": sample_ids_list,
    }
    
    if tabular_features:
        output["tabular_features"] = torch.cat(tabular_features, dim=0)
    
    if targets_list:
        output["targets"] = torch.cat(targets_list, dim=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    print(f"Saved features to: {output_path}")
    print(f"Image features shape: {output['image_features'].shape}")
    if "tabular_features" in output:
        print(f"Tabular features shape: {output['tabular_features'].shape}")
    if "targets" in output:
        print(f"Targets shape: {output['targets'].shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Dinov3 features for multimodal datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["petfinder", "covid19", "paintings", "skin-cancer"],
        required=True,
        help="Dataset to extract features from",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to dataset directory (defaults to datasets/<dataset-name>)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default=None,
        help="Dataset split (only applicable for petfinder)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for features (defaults to extracted_features/<dataset>_dinov3_features.pt)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--repo-dir", type=Path, default=Path("../dinov3"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Set default data_dir if not provided
    if args.data_dir is None:
        dataset_dir_map = {
            "petfinder": "petfinder-adoption-prediction",
            "covid19": "covid19-chest-xray",
            "paintings": "paintings-price-prediction",
            "skin-cancer": "skin-cancer",
        }
        args.data_dir = Path("datasets") / dataset_dir_map[args.dataset]
    
    # Set default output path if not provided
    if args.output is None:
        split_suffix = f"_{args.split}" if args.split else ""
        args.output = Path(f"extracted_features/{args.dataset}{split_suffix}_dinov3_features.pt")
    
    return args


def main() -> None:
    args = parse_args()
    extract_features(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        repo_dir=args.repo_dir,
        weights_path=args.weights,
        device=args.device,
    )



if __name__ == "__main__":
    main()
