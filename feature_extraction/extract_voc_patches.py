"""
Extract patch representations and labels from VOC2012 dataset.
Each patch becomes a row in a table with its DINOv3 features and segmentation label.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torchvision.transforms import v2
import os
from tqdm import tqdm
import numpy as np


def make_transform(resize_size: int = 224):
    """Create image transforms for DINOv3 input."""
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def make_mask_transform(resize_size: int = 224):
    """Create mask transforms (no normalization)."""
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(resize_size),
        transforms.PILToTensor(),
    ])


class VOCDataset(VOCSegmentation):
    """VOC dataset wrapper that returns images, masks, and dataset index."""
    def __init__(self, root, year, image_set, image_transform=None, mask_transform=None):
        super().__init__(root, year=year, image_set=image_set)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        if self.image_transform:
            img = self.image_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask).long().squeeze(0)
        return img, mask, idx


def extract_patch_features(model, images):
    """
    Extract patch features from images using DINOv3.
    
    Args:
        model: DINOv3 model
        images: [B, 3, H, W] tensor
        
    Returns:
        tokens: [B, N, D] patch tokens (excluding cls token)
    """
    with torch.no_grad():
        out = model.forward_features(images)
        tokens = out["x_norm_patchtokens"]  # [B, N, D]
    return tokens


def extract_voc_patches(
    dataset_name: str = "voc2012",
    image_set: str = "train",
    patch_size: int = 16,
    image_size: int = 224,
    batch_size: int = 4,
    num_workers: int = 4,
    num_classes: int = 21,
):
    """
    Extract patch representations and label distributions from VOC2012 dataset.
    
    Args:
        dataset_name: Name of dataset (for organizing output)
        image_set: 'train' or 'val'
        patch_size: Size of DINOv3 patches
        image_size: Input image size to DINOv3
        batch_size: Batch size for extraction
        num_workers: Number of data loading workers
        num_classes: Number of classes in VOC (default 21, including background)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = f"extracted_features/{dataset_name}_patches"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load DINOv3 model
    print("Loading DINOv3 model...")
    weights_path = "model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    REPO_DIR = '../dinov3'
    
    dinov3 = torch.hub.load(
        REPO_DIR, 'dinov3_vitb16', source='local', weights=weights_path
    )
    dinov3 = dinov3.to(device)
    dinov3.eval()
    for p in dinov3.parameters():
        p.requires_grad = False
    
    # Create dataset
    print(f"Loading VOC2012 {image_set} dataset...")
    image_transform = make_transform(image_size)
    mask_transform = make_mask_transform(image_size)
    
    dataset = VOCDataset(
        root="datasets",
        year="2012",
        image_set=image_set,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Image size: {image_size}, Patch size: {patch_size}")
    
    # Calculate dimensions
    h_patches = image_size // patch_size
    w_patches = image_size // patch_size
    num_patches = h_patches * w_patches
    
    print(f"Patches per image: {num_patches} ({h_patches} x {w_patches})")
    
    # Storage for all patches
    all_features = []
    all_label_distributions = []
    image_ids = []
    image_indices = []
    
    # Extract patches
    print(f"Extracting patches from {image_set} set...")
    for batch_idx, (images, masks, indices) in enumerate(tqdm(loader, desc=f"Extracting {image_set} patches")):
        images = images.to(device)
        masks = masks.to(device)
        B = images.shape[0]
        
        # Extract patch features
        patch_features = extract_patch_features(dinov3, images)  # [B, N, D]
        
        # For each patch, get the label distribution from the mask
        for b in range(B):
            mask = masks[b]  # [H, W]
            features = patch_features[b]  # [N, D]
            
            # Collect label distribution for each patch
            patch_label_dists = []
            for ph in range(h_patches):
                for pw in range(w_patches):
                    # Get all pixels within this patch region
                    mask_y_start = ph * patch_size
                    mask_y_end = (ph + 1) * patch_size
                    mask_x_start = pw * patch_size
                    mask_x_end = (pw + 1) * patch_size
                    
                    patch_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]  # [patch_size, patch_size]
                    
                    # Compute label distribution (histogram)
                    # Filter out ignore label (255) and compute distribution over valid labels
                    valid_labels = patch_mask[patch_mask < num_classes].flatten()
                    
                    if len(valid_labels) > 0:
                        # Count occurrences of each class
                        label_dist = torch.bincount(valid_labels, minlength=num_classes).float()
                        # Normalize to get probability distribution
                        label_dist = label_dist / label_dist.sum()
                    else:
                        # If all labels are invalid, create uniform distribution
                        label_dist = torch.ones(num_classes, device=device) / num_classes
                    
                    patch_label_dists.append(label_dist.cpu())
            
            patch_label_dists = torch.stack(patch_label_dists)  # [N, num_classes]
            
            # Store features and label distributions
            all_features.append(features.cpu())
            all_label_distributions.append(patch_label_dists.cpu())
            image_ids.append(f"{image_set}_{batch_idx}_{b}")
            image_indices.append(int(indices[b]))
    
    # Concatenate all patches
    print(f"Concatenating {len(all_features)} images worth of patches...")
    features_tensor = torch.cat(all_features, dim=0)  # [Total_patches, D]
    label_distributions_tensor = torch.cat(all_label_distributions, dim=0)  # [Total_patches, num_classes]
    
    print(f"Total patches: {features_tensor.shape[0]}")
    print(f"Feature dimension: {features_tensor.shape[1]}")
    print(f"Label distribution shape: {label_distributions_tensor.shape}")
    
    # Save to disk
    output_features_path = os.path.join(output_dir, f"{image_set}_features.pt")
    output_labels_path = os.path.join(output_dir, f"{image_set}_label_distributions.pt")
    output_metadata_path = os.path.join(output_dir, f"{image_set}_metadata.pt")
    
    print(f"Saving to {output_dir}...")
    torch.save(features_tensor, output_features_path)
    torch.save(label_distributions_tensor, output_labels_path)
    torch.save(
        {
            "image_ids": image_ids,
            "image_indices": image_indices,
            "patch_size": patch_size,
            "image_size": image_size,
            "num_classes": num_classes,
        },
        output_metadata_path,
    )
    
    print(f"Saved features to {output_features_path}")
    print(f"Saved label distributions to {output_labels_path}")
    print(f"Saved metadata to {output_metadata_path}")
    
    return features_tensor, label_distributions_tensor


if __name__ == "__main__":
    # Extract train set
    print("=" * 60)
    print("EXTRACTING VOC2012 TRAINING SET")
    print("=" * 60)
    train_features, train_labels = extract_voc_patches(
        dataset_name="voc2012",
        image_set="train",
        patch_size=16,
        image_size=224,
        batch_size=4,
        num_workers=4,
    )
    
    # Extract validation set
    print("\n" + "=" * 60)
    print("EXTRACTING VOC2012 VALIDATION SET")
    print("=" * 60)
    val_features, val_labels = extract_voc_patches(
        dataset_name="voc2012",
        image_set="val",
        patch_size=16,
        image_size=224,
        batch_size=4,
        num_workers=4,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Train: {train_features.shape[0]} patches, {train_features.shape[1]} features")
    print(f"Val:   {val_features.shape[0]} patches, {val_features.shape[1]} features")
    print(f"Train label distributions shape: {train_labels.shape}")
    print(f"Val label distributions shape:   {val_labels.shape}")
