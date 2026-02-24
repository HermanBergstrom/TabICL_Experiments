"""Simple TabICL script for PetFinder tabular features.

This script loads the PetFinder dataset, splits it into training and validation,
fits a TabICL model using only tabular features, and evaluates performance.

Example:
  python multimodal_tabicl.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from baseline_heads import LinearProbe, MLPHead, train_head, predict_head


def load_features(
    features_path: Path | str,
    dataset: str = "petfinder",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Load pre-extracted features for any dataset (both image and tabular).

    Args:
        features_path: Path to the saved features file containing both modalities.
        dataset: Dataset name (for display purposes).

    Returns:
        Tuple of (image_features, tabular_features, targets, sample_ids).
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
    #sample_ids = data["sample_ids"]
    
    print(f"Loaded {len(image_features)} samples")
    print(f"Image features shape: {image_features.shape}")
    print(f"Tabular features shape: {tabular_features.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Target distribution: {np.bincount(targets)}")
    
    return image_features, tabular_features, targets#, sample_ids


def run_single_seed(
    X_img: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
    classifier: str,
    modality: str,
    n_estimators: int,
    pca_dims: int | None,
    rp_dims: int | None,
    train_size: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run a single training/evaluation with a specific seed.
    
    Args:
        X_img: Image features.
        X_tab: Tabular features.
        y: Labels.
        seed: Random seed for this run.
        test_size: Validation set fraction.
        classifier: Type of classifier ("tabicl", "tabpfn", "logistic_regression", or "mlp").
        modality: Which features to use ("tabular", "image", or "concat").
        n_estimators: Number of estimators.
        pca_dims: PCA dimensions (None to skip).
        rp_dims: Random projection dimensions (None to skip).
        train_size: Number of training samples to use (None to use all).
        verbose: Whether to print progress and results.
    
    Returns:
        Dictionary with metrics for each modality.
    """
    # Split all features with the given seed
    X_train_img, X_val_img, X_train_tab, X_val_tab, y_train, y_val = train_test_split(
        X_img,
        X_tab,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    
    # Subset training data if train_size is specified
    if train_size is not None and train_size < len(X_train_img):
        rng = np.random.RandomState(seed)
        train_indices = rng.choice(len(X_train_img), size=train_size, replace=False)
        X_train_img = X_train_img[train_indices]
        X_train_tab = X_train_tab[train_indices]
        y_train = y_train[train_indices]

    #Limit validation to 1000 samples for faster evaluation (only for larger datasets)
    if len(X_val_img) > 1000:
        rng = np.random.RandomState(seed)
        val_indices = rng.choice(len(X_val_img), size=1000, replace=False)
        X_val_img = X_val_img[val_indices]
        X_val_tab = X_val_tab[val_indices]
        y_val = y_val[val_indices]

    print(f"Training samples: {len(X_train_img)}, Validation samples: {len(X_val_img)}")

    # Apply PCA to image features if requested
    if pca_dims is not None:
        pca_img = PCA(n_components=pca_dims, random_state=seed)
        X_train_img = pca_img.fit_transform(X_train_img)
        X_val_img = pca_img.transform(X_val_img)
    
    # Apply random projection to image features if requested
    if rp_dims is not None:
        rp_img = GaussianRandomProjection(n_components=rp_dims, random_state=seed)
        X_train_img = rp_img.fit_transform(X_train_img)
        X_val_img = rp_img.transform(X_val_img)
    
    results = {}
    
    if modality in ("tabular", "concat"):
        results["tabular"] = train_and_evaluate(
            X_train=X_train_tab,
            X_val=X_val_tab,
            y_train=y_train,
            y_val=y_val,
            classifier=classifier,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=verbose,
        )
    
    if modality in ("image", "concat"):
        results["image"] = train_and_evaluate(
            X_train=X_train_img,
            X_val=X_val_img,
            y_train=y_train,
            y_val=y_val,
            classifier=classifier,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=verbose,
        )

    if modality == "concat":
        X_train_concat = np.concatenate([X_train_img, X_train_tab], axis=1)
        X_val_concat = np.concatenate([X_val_img, X_val_tab], axis=1)
        results["concat"] = train_and_evaluate(
            X_train=X_train_concat,
            X_val=X_val_concat,
            y_train=y_train,
            y_val=y_val,
            classifier=classifier,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=verbose,
        )
    
    return results


def train_and_evaluate(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    classifier: str = "tabicl",
    n_estimators: int = 1,
    random_state: int = 42,
    verbose: bool = True,
    tabular_model: str = "tabicl",
) -> dict:
    """Train classifier and evaluate on validation set.

    Args:
        X_train: Training features.
        X_val: Validation features.
        y_train: Training targets.
        y_val: Validation targets.
        classifier: Type of classifier ("tabicl", "tabpfn", "logistic_regression", or "mlp").
        n_estimators: Number of estimators for TabICL/TabPFN (ignored for other classifiers).
        random_state: Random seed.
        verbose: Whether to print results.

    Returns:
        Dictionary with evaluation metrics.
    """
    if verbose:
        print(f"\nTraining {classifier} model...")
    
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    
    # Get number of classes
    num_classes = len(np.unique(y_train))
    
    if classifier == "tabicl":
        clf = TabICLClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_pred = clf.y_encoder_.inverse_transform(y_pred)
    
    elif classifier == "tabpfn":
        clf = TabPFNClassifier(random_state=random_state)
        #clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, ignore_pretraining_limits=True)
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    elif classifier == "logistic_regression":
        # Linear probe using PyTorch
        model = LinearProbe(input_dim=X_train.shape[1], num_classes=num_classes)
        model, _ = train_head(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_classes=num_classes,
            learning_rate=1e-3,
            num_epochs=100,
            batch_size=32,
            early_stopping_patience=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=verbose,
        )
        y_pred, y_pred_proba = predict_head(model, X_val, device="cuda" if torch.cuda.is_available() else "cpu")
    
    elif classifier == "mlp":
        # 2-layer MLP using PyTorch
        model = MLPHead(input_dim=X_train.shape[1], num_classes=num_classes, hidden_dim=256)
        model, _ = train_head(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_classes=num_classes,
            learning_rate=1e-4,
            num_epochs=100,
            batch_size=32,
            early_stopping_patience=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=verbose,
        )
        y_pred, y_pred_proba = predict_head(model, X_val, device="cuda" if torch.cuda.is_available() else "cpu")
    
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    if verbose:
        print("Making predictions on validation set...")

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }

    # Compute AUROC (handles multi-class with ovr strategy)
    try:
        if y_pred_proba.shape[1] == 2:
            # Binary classification case
            metrics["auroc"] = roc_auc_score(y_val, y_pred_proba[:, 1])
        else:
            metrics["auroc"] = roc_auc_score(y_val, y_pred_proba, multi_class="ovr", average="weighted")
    except Exception as e:
        #if verbose:
        print(f"Warning: Could not compute AUROC - {e}")
        metrics["auroc"] = None

    if verbose:
        print("\nValidation Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if metrics["auroc"] is not None:
            print(f"  AUROC:     {metrics['auroc']:.4f}")

    return metrics


def train_icl_fusion(
    X_train_img: np.ndarray,
    X_val_img: np.ndarray,
    X_train_tab: np.ndarray,
    X_val_tab: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 1,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Extract TabICL features from tabular data and fuse with images, then train classifier.

    Args:
        X_train_img: Training image features.
        X_val_img: Validation image features.
        X_train_tab: Training tabular features.
        X_val_tab: Validation tabular features.
        y_train: Training targets.
        y_val: Validation targets.
        n_estimators: Number of estimators for TabICL.
        random_state: Random seed.
        verbose: Whether to print results.

    Returns:
        Dictionary with evaluation metrics.
    """
    if verbose:
        print("\nTraining TabICL on tabular features...")
    
    # Train TabICL on tabular features
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=random_state)
    X_train_tab = np.asarray(X_train_tab, dtype=np.float32)
    X_val_tab = np.asarray(X_val_tab, dtype=np.float32)
    clf.fit(X_train_tab, y_train)
    
    # Extract ICL representations (512-dim)
    if verbose:
        print("Extracting ICL representations...")
    train_icl_feats = clf.get_icl_features(X_train_tab)
    val_icl_feats = clf.get_icl_features(X_val_tab)
    
    if verbose:
        print(f"ICL features shape: {train_icl_feats.shape}")
    
    # Concatenate with image features
    X_train_fused = np.concatenate([X_train_img, train_icl_feats], axis=1)
    X_val_fused = np.concatenate([X_val_img, val_icl_feats], axis=1)
    
    if verbose:
        print(f"Fused features shape: {X_train_fused.shape}")
        print("Training linear probe...")
    
    # Train linear probe (LogisticRegression)
    probe = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        multi_class='ovr',
    )
    
    probe.fit(X_train_fused, y_train)
    
    if verbose:
        print("Making predictions...")
    y_pred = probe.predict(X_val_fused)
    y_pred_proba = probe.predict_proba(X_val_fused)
    
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
        if verbose:
            print(f"Warning: Could not compute AUROC - {e}")
        metrics["auroc"] = None
    
    if verbose:
        print("\nValidation Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if metrics["auroc"] is not None:
            print(f"  AUROC:     {metrics['auroc']:.4f}")
    
    return metrics


def save_results(
    all_results: dict,
    config: dict,
    output_dir: Path | str = "results/multimodal_results",
) -> Path:
    """Save results and configuration to a JSON file.
    
    Args:
        all_results: Dictionary containing results organized by train_size and modality.
        config: Dictionary with all configuration parameters.
        output_dir: Directory to save results to.
    
    Returns:
        Path to the saved results file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = config.get("dataset", "unknown")
    classifier = config.get("classifier", "unknown")
    filename = f"{dataset}_{classifier}_{timestamp}.json"
    filepath = output_dir / filename
    
    # Prepare data to save
    save_data = {
        "config": config,
        "results": all_results,
    }
    
    # Save to JSON (convert numpy types to Python types for JSON serialization)
    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def main(
    features_path: Path | str = "extracted_features/petfinder_dinov3_features.pt",
    dataset: str = "petfinder",
    test_size: float = 0.2,
    num_seeds: int = 5,
    classifier: str = "tabicl",
    modality: str = "both",
    n_estimators: int = 1,
    pca_dims: int | None = None,
    rp_dims: int | None = None,
    train_sizes: list[int] | None = None,
    save_results_flag: bool = True,
    verbose: bool = True,
) -> None:
    """Main script: load data, split, train, and evaluate on any dataset.

    Args:
        features_path: Path to extracted features file (contains both modalities).
        dataset: Dataset name (petfinder, covid19, skin-cancer, paintings).
        test_size: Fraction of data to use for validation.
        num_seeds: Number of random seeds to run.
        classifier: Classifier to use ("tabicl", "tabpfn", "logistic_regression", or "mlp").
        modality: Which features to use ("tabular", "image", or "concat").
        n_estimators: Number of estimators for TabICL/TabPFN (ignored for other classifiers).
        pca_dims: Number of PCA components to reduce image features to (None to skip PCA).
        rp_dims: Number of random projection dimensions for image features (None to skip RP).
        train_sizes: List of training set sizes to evaluate (e.g., [10, 100, 1000]). If None, uses all training data.
        save_results_flag: Whether to save results to a JSON file.
        verbose: Whether to print progress and results.
    """
    print("=" * 70)
    print(f"Multimodal Learning on {dataset.upper()} Dataset (Multi-Seed Evaluation)")
    print(f"Classifier: {classifier.upper()}")
    print(f"Modality: {modality.upper()}")
    print("=" * 70)
    
    # Load all features from the .pt file
    print("\nLoading features...")
    X_img, X_tab, y = load_features(features_path=features_path, dataset=dataset)
    
    # If no train_sizes specified, use all training data (None)
    if train_sizes is None:
        train_sizes = [None]
    
    # Validate train_sizes against actual training set size
    max_train_size = int(len(y) * (1 - test_size))
    if train_sizes != [None]:
        original_train_sizes = train_sizes.copy()
        train_sizes = [min(size, max_train_size) for size in train_sizes]
        
        # Check if any sizes were adjusted
        if train_sizes != original_train_sizes:
            print(f"\nWarning: Some requested train sizes exceed maximum available ({max_train_size}):")
            for orig, adjusted in zip(original_train_sizes, train_sizes):
                if orig != adjusted:
                    print(f"  {orig} -> {adjusted}")
            print(f"Adjusted train_sizes: {train_sizes}")
    
    print(f"\nRunning {num_seeds} seeds for {len(train_sizes)} training set size(s)...")
    print(f"Validation set size: {test_size*100:.0f}%")
    print(f"Training set size(s): {train_sizes}")
    if pca_dims is not None:
        print(f"Image PCA dimensions: {pca_dims}")
    if rp_dims is not None:
        print(f"Image random projection dimensions: {rp_dims}")
    
    # Store all results organized by train_size
    all_results_by_train_size = {}
    
    # Loop over each training size
    for train_size in train_sizes:
        print(f"\n" + "=" * 70)
        if train_size is None:
            print("Using ALL training data")
        else:
            print(f"Training with {train_size} samples")
        print("=" * 70)
        
        # Collect results from all seeds for this train_size
        all_results = {
            "tabular": [],
            "image": [],
            "concat": [],
        }
        
        for seed in range(num_seeds):
            print(f"\n[Seed {seed}]")
            results = run_single_seed(
                X_img=X_img,
                X_tab=X_tab,
                y=y,
                seed=seed,
                test_size=test_size,
                classifier=classifier,
                modality=modality,
                n_estimators=n_estimators,
                pca_dims=pca_dims,
                rp_dims=rp_dims,
                train_size=train_size,
                verbose=verbose,
            )
            
            for mod_name, metrics in results.items():
                all_results[mod_name].append(metrics)
        
        # Compute and display statistics for this train_size
        print("\n" + "=" * 70)
        print(f"RESULTS FOR TRAIN_SIZE={train_size if train_size else 'ALL'}")
        print("=" * 70)
        
        metric_names = ["accuracy", "precision", "recall", "f1", "auroc"]
        
        if modality in ("tabular", "both"):
            print("\n" + "-" * 70)
            print("TABULAR-ONLY FEATURES")
            print("-" * 70)
            _print_metric_stats(all_results["tabular"], metric_names)
        
        if modality in ("image", "both"):
            print("\n" + "-" * 70)
            print("IMAGE-ONLY FEATURES (DinoV3)")
            print("-" * 70)
            _print_metric_stats(all_results["image"], metric_names)
        
        if modality == "concat":
            print("\n" + "-" * 70)
            print("CONCATENATED FEATURES (Image + Tabular)")
            print("-" * 70)
            _print_metric_stats(all_results["concat"], metric_names)
        
        if modality == "both":
            print("\n" + "=" * 70)
            print("COMPARISON: TABULAR vs IMAGE")
            print("=" * 70)
            _print_comparison(all_results["tabular"], all_results["image"], metric_names)
        
        # Store results for this train_size
        key = str(train_size) if train_size else "all"
        all_results_by_train_size[key] = all_results
    
    print("\n" + "=" * 70)
    print("Script completed successfully!")
    print("=" * 70)
    
    # Save results if requested
    if save_results_flag:
        config = {
            "dataset": dataset,
            "classifier": classifier,
            "modality": modality,
            "train_sizes": train_sizes if train_sizes != [None] else None,
            "test_size": test_size,
            "num_seeds": num_seeds,
            "n_estimators": n_estimators,
            "pca_dims": pca_dims,
            "rp_dims": rp_dims,
        }
        save_results(all_results_by_train_size, config)


def _print_metric_stats(metrics_list: list, metric_names: list) -> None:
    """Print mean and std of metrics across multiple runs."""
    print(f"\n{'Metric':<15} {'Mean':<15} {'Std':<15}")
    print("-" * 45)
    
    for metric in metric_names:
        values = [m.get(metric) for m in metrics_list]
        values = [v for v in values if v is not None]
        
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric:<15} {mean_val:<15.4f} {std_val:<15.4f}")
        else:
            print(f"{metric:<15} N/A            N/A")


def _print_comparison(metrics_list_1: list, metrics_list_2: list, metric_names: list) -> None:
    """Print comparison between two sets of metrics."""
    print(f"\n{'Metric':<15} {'Tabular':<20} {'Image':<20} {'Difference':<20}")
    print("-" * 75)
    
    for metric in metric_names:
        values_1 = [m.get(metric) for m in metrics_list_1]
        values_2 = [m.get(metric) for m in metrics_list_2]
        values_1 = [v for v in values_1 if v is not None]
        values_2 = [v for v in values_2 if v is not None]
        
        if values_1 and values_2:
            mean_1 = np.mean(values_1)
            std_1 = np.std(values_1)
            mean_2 = np.mean(values_2)
            std_2 = np.std(values_2)
            diff_mean = mean_2 - mean_1
            diff_std = np.sqrt(std_1**2 + std_2**2)
            
            print(f"{metric:<15} {mean_1:.4f}±{std_1:.4f}   {mean_2:.4f}±{std_2:.4f}   {diff_mean:+.4f}±{diff_std:.4f}")
        else:
            print(f"{metric:<15} N/A                    N/A                    N/A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate TabICL on multimodal dataset features."
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
        "--n-estimators",
        type=int,
        default=1,
        help="Number of TabICL estimators",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["tabicl", "tabpfn", "logistic_regression", "mlp"],
        default="tabicl",
        help="Classifier to use for prediction",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["tabular", "image", "concat", "both"],
        default="both",
        help="Which features to use: 'tabular', 'image', 'concat' (image+tabular concatenated), or 'both' (comparison)",
    )
 
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=None,
        help="Number of PCA components to reduce image features to (e.g., 128). If not specified, PCA is not applied.",
    )
    parser.add_argument(
        "--rp-dims",
        type=int,
        default=None,
        help="Number of random projection dimensions for image features (e.g., 128). If not specified, random projection is not applied.",
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="*",
        default=None,
        help="List of training set sizes to evaluate (e.g., --train-sizes 10 100 1000 10000). If not specified, all training data is used.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print progress and results.",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set default features path if not provided
    if args.features_path is None:
            args.features_path = Path(f"extracted_features/{args.dataset}_dinov3_features.pt")
    
    #TODO: Currently only works for petfinder, need to debug other datasets.
    # Convert empty list from nargs="*" to None
    train_sizes = args.train_sizes if args.train_sizes else None
    
    main(
        features_path=args.features_path,
            dataset=args.dataset,
        test_size=args.test_size,
        num_seeds=args.num_seeds,
        classifier=args.classifier,
        modality=args.modality,
        n_estimators=args.n_estimators,
        pca_dims=args.pca_dims,
        rp_dims=args.rp_dims,
        train_sizes=train_sizes,
        save_results_flag=True,
        verbose=args.verbose,
    )
