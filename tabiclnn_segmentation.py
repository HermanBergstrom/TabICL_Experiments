"""
TabICL-based segmentation using nearest neighbor training set construction.

For each validation image:
1. Find the k nearest neighbors (in training set) for each patch
2. Aggregate these to form a training set for TabICL
3. Use TabICL to classify the validation patches
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torchvision.transforms import v2
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from tabicl import TabICLClassifier


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
    """VOC dataset wrapper that returns images and masks."""
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
        return img, mask


class PatchTrainingSetBuilder:
    """Builds training sets for TabICL using nearest neighbors or random sampling."""
    
    def __init__(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        k: int = 10,
        device: str = "cuda",
        use_random: bool = False,
        random_seed: int = 42,
    ):
        """
        Args:
            train_features: [N_train, D] training patch features
            train_labels: [N_train] training patch labels
            k: Number of nearest neighbors to retrieve per patch (or samples if random)
            device: Device to use for computation
            use_random: If True, randomly sample instead of using nearest neighbors
            random_seed: Random seed for reproducibility
        """
        self.train_features = train_features.to(device)
        self.train_labels = train_labels.to(device)
        self.k = k
        self.device = device
        self.use_random = use_random
        self.random_seed = random_seed
        if use_random:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
    def build_training_set(self, val_patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each validation patch, find k nearest neighbors (or random samples) in training set.
        Aggregate and return unique training set.
        
        Args:
            val_patches: [N_val, D] validation patch features
            
        Returns:
            training_features: [N_train_agg, D] aggregated training features
            training_labels: [N_train_agg] aggregated training labels
            neighbor_indices: [N_val, k] indices of selected neighbors
        """
        val_patches = val_patches.to(self.device)
        N_val = val_patches.shape[0]
        N_train = self.train_features.shape[0]
        
        if self.use_random:
            # Randomly sample k * N_val indices (with replacement)
            total_samples = self.k * N_val
            random_indices = torch.randint(0, N_train, (total_samples,), device=self.device)
            neighbor_indices = random_indices.reshape(N_val, self.k)
        else:
            # Compute pairwise euclidean distances
            # Using broadcasting: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a @ b^T
            val_norm = (val_patches ** 2).sum(dim=1, keepdim=True)  # [N_val, 1]
            train_norm = (self.train_features ** 2).sum(dim=1, keepdim=True)  # [N_train, 1]
            
            # Compute distances
            distances = val_norm + train_norm.t() - 2 * val_patches @ self.train_features.t()
            distances = torch.clamp(distances, min=0).sqrt()  # [N_val, N_train]
            
            # Find k nearest neighbors for each validation patch
            _, neighbor_indices = torch.topk(distances, self.k, dim=1, largest=False)
            # neighbor_indices: [N_val, k]
        
        # Flatten to get all neighbor indices (with potential duplicates)
        neighbor_indices_flat = neighbor_indices.flatten()
        
        # Get unique indices while preserving information
        unique_indices = torch.unique(neighbor_indices_flat)
        
        # Get the features and labels for unique neighbors
        training_features = self.train_features[unique_indices]
        training_labels = self.train_labels[unique_indices]
        
        return training_features.cpu(), training_labels.cpu(), neighbor_indices


class SimpleNNClassifier:
    """Simple 1-NN classifier as baseline for patches."""
    
    def __init__(self, train_features: torch.Tensor, train_labels: torch.Tensor, device: str = "cuda"):
        self.train_features = train_features.to(device)
        self.train_labels = train_labels.to(device)
        self.device = device
    
    def predict(self, val_patches: torch.Tensor) -> torch.Tensor:
        """
        Classify validation patches using 1-NN.
        
        Args:
            val_patches: [N_val, D] validation patch features
            
        Returns:
            predictions: [N_val] predicted labels
        """
        val_patches = val_patches.to(self.device)
        
        # Compute distances
        val_norm = (val_patches ** 2).sum(dim=1, keepdim=True)
        train_norm = (self.train_features ** 2).sum(dim=1, keepdim=True)
        distances = val_norm + train_norm.t() - 2 * val_patches @ self.train_features.t()
        distances = torch.clamp(distances, min=0).sqrt()
        
        # Find nearest neighbor
        _, nearest_indices = torch.min(distances, dim=1)
        
        # Get predictions
        predictions = self.train_labels[nearest_indices]
        
        return predictions.cpu()


class ConfusionMatrix:
    """Compute confusion matrix and mIoU."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, preds, targets):
        """
        preds:   [N] (int)
        targets: [N] (int)
        """
        preds = preds.view(-1)
        targets = targets.view(-1)
        mask = (
            (targets >= 0)
            & (targets < self.num_classes)
            & (preds >= 0)
            & (preds < self.num_classes)
        )
        preds = preds[mask]
        targets = targets[mask]
        inds = self.num_classes * targets + preds
        self.mat += torch.bincount(
            inds,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute_iou(self):
        h = torch.diag(self.mat)
        fp = self.mat.sum(0) - h
        fn = self.mat.sum(1) - h

        denom = h + fp + fn
        iou = h.float() / denom.float().clamp(min=1)

        return iou

    def mean_iou(self):
        return self.compute_iou().mean().item()
    
    def get_observed_classes(self):
        """Return classes that have been observed in targets."""
        # A class is observed if it appears in any row (ground truth)
        observed = (self.mat.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        return observed.tolist()
    
    def get_stats(self):
        """Return statistics about observed classes."""
        observed = self.get_observed_classes()
        num_observed = len(observed)
        num_unobserved = self.num_classes - num_observed
        return {
            "num_observed": num_observed,
            "num_unobserved": num_unobserved,
            "observed_classes": observed,
        }


def evaluate_with_classifier(
    features_dir: str = "extracted_features/voc2012_patches",
    k: int = 10,
    patch_size: int = 16,
    image_size: int = 224,
    batch_size: int = 4,
    num_workers: int = 4,
    classifier_fn = None,
    classifier_name: str = "Custom",
    use_random_sampling: bool = False,
    random_seed: int = 42,
):
    """
    Evaluate segmentation with custom classifier on constructed training sets.
    
    Args:
        features_dir: Directory containing extracted features
        k: Number of nearest neighbors to retrieve per patch (or random samples)
        patch_size: Size of DINOv3 patches
        image_size: Input image size
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        classifier_fn: Function(val_patches, train_features, train_labels) -> predictions
                      If None, uses 1-NN baseline
        classifier_name: Name of classifier for reporting
        use_random_sampling: If True, randomly sample training set instead of nearest neighbors
        random_seed: Random seed for reproducibility
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load features
    print(f"Loading features from {features_dir}...")
    train_features = torch.load(os.path.join(features_dir, "train_features.pt"))
    train_labels = torch.load(os.path.join(features_dir, "train_labels.pt"))
    val_features = torch.load(os.path.join(features_dir, "val_features.pt"))
    val_labels = torch.load(os.path.join(features_dir, "val_labels.pt"))
    val_metadata = torch.load(os.path.join(features_dir, "val_metadata.pt"))
    
    print(f"Train: {train_features.shape}, {train_labels.shape}")
    print(f"Val: {val_features.shape}, {val_labels.shape}")
    
    # Load VOC2012 validation dataset to get original masks
    print("Loading VOC2012 validation dataset for original masks...")
    image_transform = make_transform(image_size)
    mask_transform = make_mask_transform(image_size)
    
    val_dataset = VOCDataset(
        root="datasets",
        year="2012",
        image_set="val",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Extract image indices from metadata to properly align patches with masks
    val_image_indices = val_metadata.get("image_indices")
    if val_image_indices is None:
        raise ValueError(
            "val_metadata missing 'image_indices'. Re-run extract_voc_patches.py to regenerate."
        )
    
    # Initialize training set builder
    builder = PatchTrainingSetBuilder(
        train_features, 
        train_labels, 
        k=k, 
        device=device,
        use_random=use_random_sampling,
        random_seed=random_seed,
    )
    
    # Default to 1-NN baseline if no classifier provided
    if classifier_fn is None:
        def classifier_fn(val_patches, train_features, train_labels):
            classifier = SimpleNNClassifier(train_features, train_labels, device=device)
            return classifier.predict(val_patches)
        classifier_name = "1-NN Baseline"
    
    # Process each validation image
    h_patches = image_size // patch_size
    w_patches = image_size // patch_size
    num_patches_per_image = h_patches * w_patches
    
    cm = ConfusionMatrix(num_classes=21)
    
    print(f"\nEvaluating {len(val_features) // num_patches_per_image} validation images (k={k})...")
    print(f"Patches per image: {num_patches_per_image}")
    print(f"Classifier: {classifier_name}")
    print(f"Sampling method: {'Random' if use_random_sampling else 'Nearest Neighbors'}")
    
    total_time = 0
    num_images = len(val_features) // num_patches_per_image
    if len(val_image_indices) != num_images:
        raise ValueError(
            "Mismatch between metadata image_indices and number of images: "
            f"{len(val_image_indices)} vs {num_images}"
        )
    
    for img_idx in tqdm(range(num_images), desc="Validation images"):
        start_time = time.time()
        
        # Get patches for this image
        patch_start = img_idx * num_patches_per_image
        patch_end = patch_start + num_patches_per_image
        
        val_patches = val_features[patch_start:patch_end]
        
        # Map to original dataset index using metadata (one index per image)
        original_dataset_idx = int(val_image_indices[img_idx])
        
        # Load original mask for this image
        _, original_mask = val_dataset[original_dataset_idx]  # [H, W]
        original_mask = original_mask.cuda()
        
        # Build training set from nearest neighbors
        train_set_features, train_set_labels, _ = builder.build_training_set(val_patches)
        
        # Classify validation patches using provided classifier
        #print(val_patches.shape, train_set_features.shape, train_set_labels.shape)
        predictions = classifier_fn(val_patches, train_set_features, train_set_labels)
        
        # Convert to torch tensor if numpy array
        if isinstance(predictions, np.ndarray):
            predictions = torch.from_numpy(predictions)
        
        # Reshape predictions from [N_patches] to [1, 1, h_patches, w_patches]
        predictions_grid = predictions.reshape(1, 1, h_patches, w_patches).float().cuda()
        
        # Interpolate to original image size [1, 1, H, W]
        predictions_upsampled = F.interpolate(
            predictions_grid, 
            size=(image_size, image_size), 
            mode="nearest"
        )  # [1, 1, H, W]
        predictions_upsampled = predictions_upsampled.squeeze(0).squeeze(0).long()  # [H, W]
        
        #print(predictions_upsampled.shape, original_mask.shape)
        #print(predictions_upsampled.unique(), original_mask.unique())

        #Value bins
        #print(torch.bincount(predictions_upsampled.flatten(), minlength=21))
        #print(torch.bincount(original_mask.flatten(), minlength=21))

        # Update confusion matrix with full-resolution predictions
        cm.update(predictions_upsampled.cpu(), original_mask.cpu())
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if (img_idx + 1) % 50 == 0:
            miou = cm.mean_iou()
            stats = cm.get_stats()
            avg_time = total_time / (img_idx + 1)
            print(f"  Image {img_idx + 1}: mIoU={miou:.4f}, avg_time={avg_time:.2f}s/image")
            print(f"    Classes observed: {stats['num_observed']}/{cm.num_classes}, unobserved: {stats['num_unobserved']}")
    
    # Final results
    miou = cm.mean_iou()
    per_class_iou = cm.compute_iou()
    stats = cm.get_stats()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Method: {classifier_name} with k={k} nearest neighbors")
    print(f"Feature dimension: {train_features.shape[1]}")
    print(f"Final mIoU: {miou:.4f}")
    print(f"Classes observed: {stats['num_observed']}/{cm.num_classes}")
    print(f"Classes not observed: {stats['num_unobserved']}")
    print(f"Observed class IDs: {stats['observed_classes']}")
    print(f"Average time per image: {total_time / num_images:.2f}s")
    print(f"Per-class IoU:")
    for cls_idx, iou_val in enumerate(per_class_iou):
        print(f"  Class {cls_idx}: {iou_val:.4f}")
    
    return miou, per_class_iou


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="extracted_features/voc2012_patches",
                        help="Directory containing extracted features")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of nearest neighbors to retrieve per patch (or random samples)")
    parser.add_argument("--use-baseline", action="store_true",
                        help="Use 1-NN baseline classifier instead of placeholder")
    parser.add_argument("--use-random", action="store_true",
                        help="Use random sampling instead of nearest neighbors")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for random sampling")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # If baseline flag is set, use 1-NN; otherwise use placeholder
    if args.use_baseline:
        print("Using 1-NN baseline classifier\n")
        miou, per_class_iou = evaluate_with_classifier(
            features_dir=args.features_dir,
            k=args.k,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            classifier_fn=None,  # Uses 1-NN baseline by default
            classifier_name="1-NN Baseline",
            use_random_sampling=args.use_random,
            random_seed=args.random_seed,
        )
    else:
        print("WARNING: Using placeholder classifier (not yet integrated with TabICL)")
        print("Use --use-baseline to run 1-NN baseline\n")
        
        def tabicl_clf(val_patches, train_features, train_labels):
            clf = TabICLClassifier(n_estimators=1, random_state=1)
            clf.fit(train_features, train_labels)
            #classifier = SimpleNNClassifier(train_features, train_labels, device="cuda")
            return clf.predict(val_patches)
        
        miou, per_class_iou = evaluate_with_classifier(
            features_dir=args.features_dir,
            k=args.k,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            classifier_fn=tabicl_clf,
            classifier_name="TabICL (Placeholder)",
            use_random_sampling=args.use_random,
            random_seed=args.random_seed,
        )
