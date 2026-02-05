"""
Apply PCA dimensionality reduction to extracted VOC2012 patch features.
Fits PCA on training data and applies to both train and validation sets.
"""
import torch
import os
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm


def reduce_voc_patches_pca(
    n_components: int = 128,
    input_dir: str = "extracted_features/voc2012_patches",
    output_dir: str = None,
):
    """
    Apply PCA to reduce patch feature dimensionality.
    
    Args:
        n_components: Number of PCA components
        input_dir: Directory containing extracted features
        output_dir: Directory to save reduced features (auto-generated if None)
    """
    if output_dir is None:
        output_dir = f"extracted_features/voc2012_patches_pca{n_components}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading features from {input_dir}...")
    
    # Load training features
    train_features_path = os.path.join(input_dir, "train_features.pt")
    train_labels_path = os.path.join(input_dir, "train_labels.pt")
    train_metadata_path = os.path.join(input_dir, "train_metadata.pt")
    
    # Load validation features
    val_features_path = os.path.join(input_dir, "val_features.pt")
    val_labels_path = os.path.join(input_dir, "val_labels.pt")
    val_metadata_path = os.path.join(input_dir, "val_metadata.pt")
    
    if not all(os.path.exists(p) for p in [train_features_path, val_features_path]):
        raise FileNotFoundError(f"Feature files not found in {input_dir}")
    
    train_features = torch.load(train_features_path)
    train_labels = torch.load(train_labels_path)
    train_metadata = torch.load(train_metadata_path)
    
    val_features = torch.load(val_features_path)
    val_labels = torch.load(val_labels_path)
    val_metadata = torch.load(val_metadata_path)
    
    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")
    print(f"Original feature dimension: {train_features.shape[1]}")
    
    # Convert to numpy for PCA
    print(f"\nFitting PCA with {n_components} components...")
    train_features_np = train_features.numpy()
    val_features_np = val_features.numpy()
    
    # Fit PCA on training data
    pca = PCA(n_components=n_components)
    train_features_reduced = pca.fit_transform(train_features_np)
    
    # Transform validation data
    val_features_reduced = pca.transform(val_features_np)
    
    # Print PCA info
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Explained variance ratio: {explained_variance:.4f}")
    print(f"Train reduced shape: {train_features_reduced.shape}")
    print(f"Val reduced shape: {val_features_reduced.shape}")
    
    # Convert back to torch
    train_features_reduced = torch.from_numpy(train_features_reduced).float()
    val_features_reduced = torch.from_numpy(val_features_reduced).float()
    
    # Save reduced features
    print(f"\nSaving reduced features to {output_dir}...")
    
    train_output_features = os.path.join(output_dir, "train_features.pt")
    train_output_labels = os.path.join(output_dir, "train_labels.pt")
    train_output_metadata = os.path.join(output_dir, "train_metadata.pt")
    
    val_output_features = os.path.join(output_dir, "val_features.pt")
    val_output_labels = os.path.join(output_dir, "val_labels.pt")
    val_output_metadata = os.path.join(output_dir, "val_metadata.pt")
    
    torch.save(train_features_reduced, train_output_features)
    torch.save(train_labels, train_output_labels)
    torch.save(train_metadata, train_output_metadata)
    
    torch.save(val_features_reduced, val_output_features)
    torch.save(val_labels, val_output_labels)
    torch.save(val_metadata, val_output_metadata)
    
    # Save PCA model info
    pca_info = {
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_explained_variance": explained_variance,
        "original_n_features": train_features.shape[1],
    }
    torch.save(pca_info, os.path.join(output_dir, "pca_info.pt"))
    
    print(f"Train: {train_output_features}")
    print(f"Train: {train_output_labels}")
    print(f"Val: {val_output_features}")
    print(f"Val: {val_output_labels}")
    print(f"PCA info: {os.path.join(output_dir, 'pca_info.pt')}")
    
    return train_features_reduced, val_features_reduced, pca


if __name__ == "__main__":
    print("=" * 60)
    print("PCA REDUCTION: 128 COMPONENTS")
    print("=" * 60)
    reduce_voc_patches_pca(n_components=128)
    
    print("\n" + "=" * 60)
    print("PCA REDUCTION: 64 COMPONENTS")
    print("=" * 60)
    reduce_voc_patches_pca(n_components=64)
    
    print("\n" + "=" * 60)
    print("PCA REDUCTION COMPLETE")
    print("=" * 60)
    print("Generated directories:")
    print("  - extracted_features/voc2012_patches_pca128/")
    print("  - extracted_features/voc2012_patches_pca64/")
