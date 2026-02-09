"""Simple experiment script for TabPFN on concatenated image+tabular features.

This script loads pre-extracted features, concatenates image and tabular features,
and trains TabPFN classifier on the combined representation.

Example:
    python tabpfn_concat_experiment.py --dataset petfinder --num-seeds 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA
#from tabpfn import TabPFNClassifier
#import shapiq

from tabpfn_extensions import TabPFNClassifier
# Import tabpfn adapters from interpretability module
from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq

import matplotlib.pyplot as plt


def load_features(
    features_path: Path | str,
    dataset: str = "petfinder",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-extracted features for dataset.

    Args:
        features_path: Path to the saved features file.
        dataset: Dataset name (for display purposes).

    Returns:
        Tuple of (image_features, tabular_features, targets).
    """
    features_path = Path(features_path)
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    print(f"Loading {dataset} features from {features_path}...")
    data = torch.load(features_path, map_location="cpu")
    
    image_features = data["image_features"].numpy()
    
    # Tabular features may not exist for all datasets
    if "tabular_features" in data:
        tabular_features = data["tabular_features"].numpy()
    else:
        # Create dummy tabular features if not present
        tabular_features = np.zeros((len(image_features), 0), dtype=np.float32)
    
    targets = data["targets"].numpy()
    
    print(f"Loaded {len(image_features)} samples")
    print(f"Image features shape: {image_features.shape}")
    print(f"Tabular features shape: {tabular_features.shape}")
    print(f"Target distribution: {np.bincount(targets)}")
    
    return image_features, tabular_features, targets


def run_single_seed(
    X_img: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float = 0.2,
    pca_dims: int | None = None,
) -> dict:
    """Run a single training/evaluation with a specific seed.
    
    Args:
        X_img: Image features.
        X_tab: Tabular features.
        y: Labels.
        seed: Random seed for this run.
        test_size: Validation set fraction.
        pca_dims: Number of PCA components to reduce image features to (None to skip).
    
    Returns:
        Dictionary with metrics.
    """
    # Split data
    X_train_img, X_val_img, X_train_tab, X_val_tab, y_train, y_val = train_test_split(
        X_img,
        X_tab,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    
    # Apply PCA to image features if requested
    if pca_dims is not None:
        pca_img = PCA(n_components=pca_dims, random_state=seed)
        X_train_img = pca_img.fit_transform(X_train_img)
        X_val_img = pca_img.transform(X_val_img)
    
    # Concatenate features
    X_train = np.concatenate([X_train_img, X_train_tab], axis=1).astype(np.float32)
    X_val = np.concatenate([X_val_img, X_val_tab], axis=1).astype(np.float32)
    
    print(f"[Seed {seed}] Training TabPFN on concatenated features (shape: {X_train.shape})...")
    
    # Train TabPFN
    clf = TabPFNClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred_proba = clf.predict_proba(X_val)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }
    
    # Compute AUROC
    try:
        metrics["auroc"] = roc_auc_score(y_val, y_pred_proba, multi_class="ovr", average="weighted")
    except Exception as e:
        print(f"Warning: Could not compute AUROC - {e}")
        metrics["auroc"] = None
    
    print(f"[Seed {seed}] Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
    return metrics


def print_results(all_metrics: list[dict]) -> None:
    """Print summary statistics across multiple seeds.
    
    Args:
        all_metrics: List of metric dictionaries from each seed.
    """
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Concatenated Image + Tabular Features with TabPFN)")
    print("=" * 70)
    
    metric_names = ["accuracy", "precision", "recall", "f1", "auroc"]
    
    print(f"\n{'Metric':<15} {'Mean':<15} {'Std':<15}")
    print("-" * 45)
    
    for metric in metric_names:
        values = [m.get(metric) for m in all_metrics]
        values = [v for v in values if v is not None]
        
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric:<15} {mean_val:<15.4f} {std_val:<15.4f}")
        else:
            print(f"{metric:<15} N/A            N/A")
    
    print("=" * 70)


def main(
    dataset: str = "petfinder",
    features_path: Path | str | None = None,
    test_size: float = 0.2,
    num_seeds: int = 5,
    pca_dims: int | None = None,
) -> None:
    """Main experiment script.

    Args:
        dataset: Dataset name.
        features_path: Path to extracted features file. If None, uses default.
        test_size: Fraction of data to use for validation.
        num_seeds: Number of random seeds to run.
        pca_dims: Number of PCA components to reduce image features to (None to skip).
    """
    print("=" * 70)
    print(f"TabPFN Concatenated Features Experiment: {dataset.upper()}")
    print("=" * 70)
    
    # Set default features path if not provided
    if features_path is None:
        features_path = Path(f"extracted_features/{dataset}_dinov3_features.pt")
    
    # Load features
    X_img, X_tab, y = load_features(features_path=features_path, dataset=dataset)
    
    print(f"\nRunning {num_seeds} seeds with test_size={test_size}...")
    if pca_dims is not None:
        print(f"Image PCA dimensions: {pca_dims}")
    print()
    
    seed = 0
    # Split data
    X_train_img, X_val_img, X_train_tab, X_val_tab, y_train, y_val = train_test_split(
        X_img,
        X_tab,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    
    # Apply PCA to image features if requested
    if pca_dims is not None:
        pca_img = PCA(n_components=pca_dims, random_state=seed)
        X_train_img = pca_img.fit_transform(X_train_img)
        X_val_img = pca_img.transform(X_val_img)
    
    # Concatenate features
    X_train = np.concatenate([X_train_img, X_train_tab], axis=1).astype(np.float32)
    X_val = np.concatenate([X_val_img, X_val_tab], axis=1).astype(np.float32)
    
    X_train = X_train[:250]  # TabPFN has a max of 256 samples for explanation
    y_train = y_train[:250]
    
    model = TabPFNClassifier(random_state=seed)

    print(X_train.shape, y_train.shape)
    # Get an Shapley Interaction Explainer (here we use the Faithful Shapley Interaction Index)
    explainer = tabpfn_shapiq.get_tabpfn_explainer(
        model=model,
        data=X_train,
        labels=y_train,
        index="FSII",  # SV: Shapley Value, FSII: Faithful Shapley Interaction Index
        max_order=2,  # maximum order of the Shapley interactions (2 for pairwise interactions)
        verbose=True,  # show a progress bar during explanation
    )

    # Get shapley interaction values
    print("Calculating Shapley interaction values...")
    shapley_interaction_values = explainer.explain(x=X_train[0], budget=1000)

    #Save the Shapley interaction values for later use
    shapley_interaction_values.to_json_file("shapley_interaction_values.json")


    # Plot the upset plot for visualizing the interactions
    shapley_interaction_values.plot_upset()#feature_names=feature_names)

    #explainer = shapiq.TabPFNExplainer(   # setup the explainer
    #    model=model,
    #    data=X_train,
    #    labels=y_train,
    #    index="FSII"
    #)
    
    #fsii_values = explainer.explain(X_train[0], budget=256)  # explain with Faithful Shapley values
    #fsii_values.plot_force()

    plt.savefig("fsii_force_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    """

    # Run experiments across multiple seeds
    all_metrics = []
    for seed in range(num_seeds):x
        metrics = run_single_seed(
            X_img=X_img,
            X_tab=X_tab,
            y=y,
            seed=seed,
            test_size=test_size,
            pca_dims=pca_dims,
        )
        all_metrics.append(metrics)
    
    # Print summary
    print_results(all_metrics)
    """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TabPFN experiment on concatenated image+tabular features."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["petfinder", "covid19", "skin-cancer", "paintings"],
        default="petfinder",
        help="Dataset to use",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Path to extracted features file (defaults to extracted_features/<dataset>_dinov3_features.pt)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of random seeds to run for evaluation",
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=None,
        help="Number of PCA components to reduce image features to (e.g., 128). If not specified, PCA is not applied.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    main(
        dataset=args.dataset,
        features_path=args.features_path,
        test_size=args.test_size,
        num_seeds=args.num_seeds,
        pca_dims=args.pca_dims,
    )
