#!/usr/bin/env python
"""Extract patch features for the RSNA Pneumonia Detection dataset.

Supports two backbones:
  - rad-dino  (default): microsoft/rad-dino via HuggingFace — medical-domain ViT
  - dinov3:              local DINOv3 ViT-B/16 via torch.hub

Saves both patch tokens and the CLS token for each image.

Example:
  python extract_rsna_features.py --split train --batch-size 16
  python extract_rsna_features.py --split train --backbone dinov3 --batch-size 16
  python extract_rsna_features.py --split test  --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import math

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def dicom_to_pil(dcm: pydicom.Dataset) -> Image.Image:
    """Convert a pydicom Dataset to an 8-bit grayscale PIL Image and return as RGB."""
    pixels = dcm.pixel_array.astype(np.float32)

    # Invert if MONOCHROME1 (bright = air, dark = tissue — opposite of normal)
    pi = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    if pi.strip() == "MONOCHROME1":
        pixels = pixels.max() - pixels

    # Normalize to [0, 255]
    pmin, pmax = pixels.min(), pixels.max()
    if pmax > pmin:
        pixels = (pixels - pmin) / (pmax - pmin) * 255.0
    pixels = pixels.astype(np.uint8)

    return Image.fromarray(pixels).convert("RGB")


class RSNADataset(Dataset):
    """RSNA Pneumonia binary classification dataset (DICOM images)."""

    def __init__(
        self,
        dataset_path: Path,
        split: str = "train",
        processor=None,
        transform=None,
        seed: int = 42,
        test_fraction: float = 0.2,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.processor = processor
        self.transform = transform

        labels_csv = self.dataset_path / "stage_2_train_labels.csv"
        if not labels_csv.exists():
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

        df = pd.read_csv(labels_csv)
        # Deduplicate: one row per patient, label = 1 if any bounding box annotation exists
        df = df.groupby("patientId", as_index=False)["Target"].max()

        self.image_dir = self.dataset_path / "stage_2_train_images"
        self.classes = ["Normal", "Pneumonia"]
        self.class_to_idx = {"Normal": 0, "Pneumonia": 1}

        n_total = len(df)
        n_test = int(n_total * test_fraction)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_total)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        self.df = df.iloc[train_idx if split == "train" else test_idx].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        patient_id = row["patientId"]
        label = int(row["Target"])

        dcm_path = self.image_dir / f"{patient_id}.dcm"
        try:
            dcm = pydicom.dcmread(str(dcm_path))
            image = dicom_to_pil(dcm)
        except Exception as e:
            print(f"Error loading {dcm_path}: {e}")
            image = Image.new("RGB", (518, 518))

        if self.transform is not None:
            image = self.transform(image)

        return image, label, patient_id


def collate_fn_rad_dino(processor):
    """Collate function that applies the HuggingFace processor to a batch of PIL Images."""
    def _collate(batch):
        images, labels, patient_ids = zip(*batch)
        inputs = processor(images=list(images), return_tensors="pt")
        labels = torch.tensor(labels, dtype=torch.long)
        return inputs, labels, list(patient_ids)
    return _collate


def collate_fn_dinov3(batch):
    """Collate function for pre-transformed tensors (DINOv3 path)."""
    images, labels, patient_ids = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels, list(patient_ids)


def make_dinov3_transform(resize_size: int = 518) -> v2.Compose:
    """ImageNet-normalised transform for DINOv3 (matches ViT-B/16 training)."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_rad_dino(model_name: str, device: torch.device):
    """Load RAD-DINO model and processor from HuggingFace."""
    print(f"Loading RAD-DINO model '{model_name}' ...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, processor


def load_dinov3(device: torch.device) -> torch.nn.Module:
    """Load DINOv3 ViT-B/16 from local torch.hub source."""
    print("Loading DINOv3 model from local hub ...")
    model = torch.hub.load(
        "../dinov3",
        "dinov3_vitb16",
        source="local",
        weights_path="model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    )
    model = model.to(device)
    model.eval()
    return model


def pool_patch_tokens(patch_tok: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Spatially pool patch tokens by averaging pool_size×pool_size neighbourhoods.

    Args:
        patch_tok: [B, P, D] where P must be a perfect square (H×H grid).
        pool_size: spatial kernel / stride (e.g. 2 pools 4 neighbours → P/4 tokens).

    Returns:
        [B, P', D]  where P' = ceil(H/pool_size)²
    """
    if pool_size == 1:
        return patch_tok

    B, P, D = patch_tok.shape
    H = int(math.isqrt(P))
    if H * H != P:
        raise ValueError(
            f"Number of patch tokens ({P}) is not a perfect square; "
            "cannot reshape into a 2-D spatial grid for pooling."
        )

    # [B, P, D] → [B, D, H, H] → avg_pool → [B, D, H', H'] → [B, P', D]
    x = patch_tok.permute(0, 2, 1).reshape(B, D, H, H)
    x = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    H_out = x.shape[-1]
    x = x.reshape(B, D, H_out * H_out).permute(0, 2, 1)  # [B, P', D]
    return x


@torch.no_grad()
def extract_features_rad_dino(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pool_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Extract CLS and patch tokens using RAD-DINO (HuggingFace outputs).

    Returns:
        cls_features:   [N, D]       float32
        patch_features: [N, P', D]   float16
        labels:         [N]
        patient_ids:    list[str]
    """
    all_cls, all_patches, all_labels, all_ids = [], [], [], []

    for inputs, labels, patient_ids in tqdm(dataloader, desc="Extracting (rad-dino)"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        hidden = outputs.last_hidden_state          # [B, 1+P, D]
        cls_tok = hidden[:, 0, :].cpu()             # [B, D]  float32
        patch_tok = hidden[:, 1:, :].cpu().float()  # [B, P, D]
        patch_tok = pool_patch_tokens(patch_tok, pool_size)

        all_cls.append(cls_tok)
        all_patches.append(patch_tok.half())
        all_labels.append(labels)
        all_ids.extend(patient_ids)

    return (
        torch.cat(all_cls, dim=0),
        torch.cat(all_patches, dim=0),
        torch.cat(all_labels, dim=0),
        all_ids,
    )


@torch.no_grad()
def extract_features_dinov3(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pool_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Extract CLS and patch tokens using DINOv3 (torch.hub forward_features).

    Returns:
        cls_features:   [N, D]       float32
        patch_features: [N, P', D]   float16
        labels:         [N]
        patient_ids:    list[str]
    """
    all_cls, all_patches, all_labels, all_ids = [], [], [], []

    for images, labels, patient_ids in tqdm(dataloader, desc="Extracting (dinov3)"):
        images = images.to(device)
        feat = model.forward_features(images)

        cls_tok = feat["x_norm_clstoken"].cpu()             # [B, D]
        patch_tok = feat["x_norm_patchtokens"].cpu().float() # [B, P, D]
        patch_tok = pool_patch_tokens(patch_tok, pool_size)

        all_cls.append(cls_tok)
        all_patches.append(patch_tok.half())
        all_labels.append(labels)
        all_ids.extend(patient_ids)

    return (
        torch.cat(all_cls, dim=0),
        torch.cat(all_patches, dim=0),
        torch.cat(all_labels, dim=0),
        all_ids,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract features for RSNA Pneumonia dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/project/aip-rahulgk/hermanb/datasets/rsna-pneumonia",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/scratch/hermanb/temp_datasets/extracted_features",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["rad-dino", "dinov3"],
        default="rad-dino",
        help="Feature extraction backbone: 'rad-dino' (HuggingFace) or 'dinov3' (local torch.hub)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/rad-dino",
        help="HuggingFace model identifier (only used when --backbone rad-dino)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=2,
        help="Spatial pooling kernel applied to patch tokens after extraction. "
             "pool-size=2 averages 2x2 neighbourhoods (4 tokens → 1), "
             "reducing 37x37=1369 patches to 19x19=361. "
             "Use 1 to disable pooling.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backbone == "rad-dino":
        model, processor = load_rad_dino(args.model_name, device)
        dataset = RSNADataset(
            dataset_path=args.dataset_path,
            split=args.split,
            processor=processor,
            seed=args.seed,
            test_fraction=args.test_fraction,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_rad_dino(processor),
        )
        extract_fn = extract_features_rad_dino
    else:  # dinov3
        model = load_dinov3(device)
        transform = make_dinov3_transform()
        dataset = RSNADataset(
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
            num_workers=args.num_workers,
            collate_fn=collate_fn_dinov3,
        )
        extract_fn = extract_features_dinov3

    print(f"Dataset size ({args.split}): {len(dataset)}")
    print(f"  Pneumonia: {(dataset.df['Target'] == 1).sum()}")
    print(f"  Normal:    {(dataset.df['Target'] == 0).sum()}")

    cls_features, patch_features, labels, patient_ids = extract_fn(
        model, dataloader, device,
        pool_size=args.pool_size,
    )

    backbone_tag = args.backbone.replace("-", "_")
    out_file = output_dir / f"rsna_{args.split}_{backbone_tag}_features.pt"
    torch.save(
        {
            "cls_features": cls_features,       # [N, D]       float32
            "patch_features": patch_features,   # [N, P', D]   float16
            "labels": labels,                   # [N]  0=Normal 1=Pneumonia
            "patient_ids": patient_ids,
            "classes": dataset.classes,
            "class_to_idx": dataset.class_to_idx,
        },
        out_file,
    )

    print(f"\nSaved to {out_file}")
    print(f"  CLS features:   {cls_features.shape}  float32")
    print(f"  Patch features: {patch_features.shape}  float16")
    print(f"  Labels:         {labels.shape}")


if __name__ == "__main__":
    main()
