"""Compare linear probing vs foundation models on image features.

This script loads image features, splits into train/validation/test,
and compares linear probing to TabPFN/TabICL foundation models.
Validation is used for model selection in linear probing only.

Example:
  python image_tabfm.py --dataset petfinder --classifier linear_vs_fm
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
from sklearn.neighbors import KNeighborsClassifier
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier

from baseline_heads import LinearProbe, MLPHead, train_head, predict_head


def load_features(
    features_path: Path | str,
    dataset: str = "petfinder",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load pre-extracted image features for any dataset.

    Args:
        features_path: Path to the saved features file.
        dataset: Dataset name (for display purposes).

    Returns:
        Tuple of (image_features, targets).
    """
    features_path = Path(features_path)
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    print(f"Loading {dataset} features from {features_path}...")
    data = torch.load(features_path, map_location="cpu")
    
    image_features = data["image_features"].numpy()
    targets = data["targets"].numpy()
    
    print(f"Loaded {len(image_features)} samples")
    print(f"Image features shape: {image_features.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Target distribution: {np.bincount(targets)}")
    
    return image_features, targets


def run_single_seed(
    X_img: np.ndarray,
    y: np.ndarray,
    seed: int,
    val_size: float = 0.15,
    test_size: float = 0.2,
    n_estimators: int = 1,
    pca_dims: int | None = None,
    rp_dims: int | None = None,
    train_size: int | None = None,
    cheat_lp: bool = False,
    use_tabicl: bool = True,
    verbose: bool = True,
) -> dict:
    """Run a single training/evaluation with train/val/test split.
    
    Validation is used only for linear probing model selection.
    Foundation models (TabPFN, TabICL) only use training data.
    
    Args:
        X_img: Image features.
        y: Labels.
        seed: Random seed for this run.
        val_size: Validation set fraction (used only for linear probing).
        test_size: Test set fraction.
        n_estimators: Number of estimators for foundation models.
        pca_dims: PCA dimensions (None to skip).
        rp_dims: Random projection dimensions (None to skip).
        train_size: Number of training samples to use (None to use all).
        cheat_lp: If True, use test set for linear probe validation (not for evaluation).
        verbose: Whether to print progress and results.
    
    Returns:
        Dictionary with metrics for linear_probe and foundation_model.
    """
    # First, split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_img,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    
    # Then, split train_val into train and validation (for linear probe only)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size / (1 - test_size),  # Adjust for already removed test set
        random_state=seed,
        stratify=y_train_val,
    )
    
    # Subset training data if train_size is specified using stratified sampling
    if train_size is not None and train_size < len(X_train):
        # Use stratified sampling to ensure all classes are represented
        remaining_fraction = train_size / len(X_train)
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=remaining_fraction,
            random_state=seed,
            stratify=y_train,
        )


    #Limit test size:
    test_size_limit = 1000
    if len(X_test) > test_size_limit:
        X_test, _, y_test, _ = train_test_split(
            X_test,
            y_test,
            train_size=test_size_limit,
            random_state=seed,
            stratify=y_test,
        )

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Apply PCA to image features if requested
    if pca_dims is not None:
        pca_img = PCA(n_components=pca_dims, random_state=seed)
        X_train = pca_img.fit_transform(X_train)
        X_val = pca_img.transform(X_val)
        X_test = pca_img.transform(X_test)
    
    # Apply random projection to image features if requested
    if rp_dims is not None:
        rp_img = GaussianRandomProjection(n_components=rp_dims, random_state=seed)
        X_train = rp_img.fit_transform(X_train)
        X_val = rp_img.transform(X_val)
        X_test = rp_img.transform(X_test)
    
    results = {}
    
    # Linear probe: run both with and without cheat if cheat_lp is enabled
    # Standard version (validation set for model selection)
    results["linear_probe"] = train_and_evaluate_linear_probe(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        random_state=seed,
        cheat_lp=False,
        verbose=verbose,
    )
    
    # Cheat version (test set for model selection) - only if cheat_lp is enabled
    if cheat_lp:
        results["linear_probe_cheat"] = train_and_evaluate_linear_probe(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            X_test=X_test,
            y_test=y_test,
            random_state=seed,
            cheat_lp=True,
            verbose=verbose,
        )
    
    # Foundation models use only train, evaluate on test
    results["foundation_model"] = train_and_evaluate_foundation_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_estimators=n_estimators,
        random_state=seed,
        verbose=verbose,
    )
    
    # TabICL uses only train, evaluate on test (if enabled)
    if use_tabicl:
        results["tabicl"] = train_and_evaluate_tabicl(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_estimators=n_estimators,
            random_state=seed,
            verbose=verbose,
        )
    
    # KNN: run both with and without cheat if cheat_lp is enabled
    # Standard version (validation set for hyperparameter selection)
    results["knn"] = train_and_evaluate_knn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        random_state=seed,
        cheat_lp=False,
        verbose=verbose,
    )
    
    # Cheat version (test set for hyperparameter selection) - only if cheat_lp is enabled
    if cheat_lp:
        results["knn_cheat"] = train_and_evaluate_knn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            X_test=X_test,
            y_test=y_test,
            random_state=seed,
            cheat_lp=True,
            verbose=verbose,
        )
    
    return results



def train_and_evaluate_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    cheat_lp: bool = False,
    verbose: bool = True,
) -> dict:
    """Train linear probe with validation for model selection, evaluate on test set.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features (for model selection).
        y_val: Validation targets.
        X_test: Test features.
        y_test: Test targets.
        random_state: Random seed.
        cheat_lp: If True, indicating that X_val/y_val are from test split.
        verbose: Whether to print results.

    Returns:
        Dictionary with evaluation metrics on test set.
    """
    if verbose:
        if cheat_lp:
            print(f"\nTraining linear probe with CHEAT validation mode (using test set for model selection)...")
        else:
            print(f"\nTraining linear probe with validation mode...")
    
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    
    # Get number of classes
    num_classes = len(np.unique(y_train))
    
    # Train linear probe
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
        early_stopping_patience=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=verbose,
    )
    
    if verbose:
        print("Making predictions on test set...")

    # Predict on test set
    y_pred, y_pred_proba = predict_head(model, X_test, device="cuda" if torch.cuda.is_available() else "cpu")

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # Compute AUROC (handles multi-class with ovr strategy)
    try:
        if y_pred_proba.shape[1] == 2:
            # Binary classification case
            metrics["auroc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            metrics["auroc"] = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute AUROC - {e}")
        metrics["auroc"] = None

    if verbose:
        print("\nLinear Probe Test Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if metrics["auroc"] is not None:
            print(f"  AUROC:     {metrics['auroc']:.4f}")

    return metrics


def train_and_evaluate_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    cheat_lp: bool = False,
    verbose: bool = True,
) -> dict:
    """Train KNN with validation for hyperparameter selection, evaluate on test set.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features (for hyperparameter selection).
        y_val: Validation targets.
        X_test: Test features.
        y_test: Test targets.
        random_state: Random seed.
        cheat_lp: If True, indicating that X_val/y_val are from test split.
        verbose: Whether to print results.

    Returns:
        Dictionary with evaluation metrics on test set.
    """
    if verbose:
        if cheat_lp:
            print(f"\nTraining KNN with CHEAT hyperparameter selection (using test set)...")
        else:
            print(f"\nTraining KNN with validation hyperparameter selection...")
    
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    
    # Select best k using validation set
    k_values = [1, 3, 5, 7, 9, 15, 21, 31]
    best_k = 1
    best_accuracy = -1
    
    if verbose:
        print("Selecting best k...")
    
    for k in k_values:
        if k >= len(X_train):
            break
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        val_accuracy = knn.score(X_val, y_val)
        if verbose:
            print(f"  k={k}: {val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_k = k
    
    if verbose:
        print(f"Selected k={best_k}")
    
    # Train final model with best k
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train, y_train)
    
    if verbose:
        print("Making predictions on test set...")
    
    # Predict on test set
    y_pred = knn_final.predict(X_test)
    
    # Get probabilities for AUROC
    try:
        y_pred_proba = knn_final.predict_proba(X_test)
    except:
        # If predict_proba is not available, use one-hot encoding of predictions
        num_classes = len(np.unique(y_train))
        y_pred_proba = np.eye(num_classes)[y_pred]
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    
    # Compute AUROC
    try:
        if y_pred_proba.shape[1] == 2:
            # Binary classification case
            metrics["auroc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            metrics["auroc"] = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute AUROC - {e}")
        metrics["auroc"] = None
    
    if verbose:
        print("\nKNN Test Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if metrics["auroc"] is not None:
            print(f"  AUROC:     {metrics['auroc']:.4f}")
    
    return metrics


def train_and_evaluate_foundation_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 1,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Train foundation model (TabPFN) on training data only, evaluate on test set.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        n_estimators: Number of estimators (for TabICL).
        random_state: Random seed.
        verbose: Whether to print results.

    Returns:
        Dictionary with evaluation metrics on test set.
    """
    if verbose:
        print(f"\nTraining foundation model (TabPFN)...")
    
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    
    # Train TabPFN
    clf = TabPFNClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    
    if verbose:
        print("Making predictions on test set...")

    y_pred_proba = clf.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # Compute AUROC
    try:
        if y_pred_proba.shape[1] == 2:
            # Binary classification case
            metrics["auroc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            metrics["auroc"] = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute AUROC - {e}")
        metrics["auroc"] = None

    if verbose:
        print("\nFoundation Model Test Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if metrics["auroc"] is not None:
            print(f"  AUROC:     {metrics['auroc']:.4f}")

    return metrics


def train_and_evaluate_tabicl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 16,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Train TabICL on training data only, evaluate on test set.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        n_estimators: Number of estimators for TabICL.
        random_state: Random seed.
        verbose: Whether to print results.

    Returns:
        Dictionary with evaluation metrics on test set.
    """
    if verbose:
        print(f"\nTraining TabICL (n_estimators={n_estimators})...")
    
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    
    # Train TabICL
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    
    if verbose:
        print("Making predictions on test set...")

    y_pred_proba = clf.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # Compute AUROC
    try:
        if y_pred_proba.shape[1] == 2:
            # Binary classification case
            metrics["auroc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            metrics["auroc"] = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute AUROC - {e}")
        metrics["auroc"] = None

    if verbose:
        print("\nTabICL Test Results:")
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
    output_dir: Path | str = "results/image_only_results",
    filename: str | None = None,
) -> Path:
    """Save results and configuration to a JSON file.
    
    Args:
        all_results: Dictionary containing results organized by train_size.
        config: Dictionary with all configuration parameters.
        output_dir: Directory to save results to.
        filename: Optional specific filename to use. If None, creates filename from dataset name.
    
    Returns:
        Path to the saved results file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename if not provided
    if filename is None:
        dataset = config.get("dataset", "unknown")
        filename = f"{dataset}_linear_vs_fm.json"
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
    num_seeds: int = 5,
    val_size: float = 0.15,
    test_size: float = 0.2,
    pca_dims: int | None = None,
    rp_dims: int | None = None,
    train_sizes: list[int] | None = None,
    cheat_lp: bool = False,
    save_name: str | None = None,
    use_tabicl: bool = True,
    save_results_flag: bool = True,
    verbose: bool = True,
) -> None:
    """Main script: load image features, split into train/val/test, compare approaches.

    Args:
        features_path: Path to extracted features file (image features only).
        dataset: Dataset name (petfinder, covid19, skin-cancer, paintings).
        num_seeds: Number of random seeds to run.
        val_size: Validation set fraction (used only for linear probing model selection).
        test_size: Test set fraction (used for evaluation).
        pca_dims: Number of PCA components to reduce image features to (None to skip PCA).
        rp_dims: Number of random projection dimensions for image features (None to skip RP).
        train_sizes: List of training set sizes to evaluate (e.g., [10, 100, 1000]). If None, uses all training data.
        cheat_lp: If True, use test set for linear probe validation (not for evaluation).
        save_name: Optional name prefix to prepend to the results filename.
        use_tabicl: Whether to include TabICL in evaluation (can be disabled for memory constraints).
        save_results_flag: Whether to save results to a JSON file.
        verbose: Whether to print progress and results.
    """
    print("=" * 70)
    print(f"Linear Probing vs Foundation Models on {dataset.upper()} (Image Only)")
    print("=" * 70)
    
    # Load image features only
    print("\nLoading image features...")
    X_img, y = load_features(features_path=features_path, dataset=dataset)
    
    # If no train_sizes specified, use all training data (None)
    if train_sizes is None:
        train_sizes = [None]
    
    # Validate train_sizes against actual training set size
    max_train_size = int(len(y) * (1 - test_size) * (1 - val_size / (1 - test_size)))
    if train_sizes != [None]:
        original_train_sizes = train_sizes.copy()
        train_sizes = [min(size, max_train_size) for size in train_sizes]
        
        # Check if any sizes were adjusted
        if train_sizes != original_train_sizes:
            print(f"\nWarning: Some requested train sizes exceed maximum available (~{max_train_size}):")
            for orig, adjusted in zip(original_train_sizes, train_sizes):
                if orig != adjusted:
                    print(f"  {orig} -> {adjusted}")
    
    print(f"\nRunning {num_seeds} seeds for {len(train_sizes)} training set size(s)...")
    print(f"Data split: Test={test_size*100:.0f}%, Val={val_size*100:.0f}% (of train+val), Train={100-test_size*100-(val_size/(1-test_size))*(1-test_size)*100:.0f}%")
    print(f"Training set size(s): {train_sizes}")
    if pca_dims is not None:
        print(f"Image PCA dimensions: {pca_dims}")
    if rp_dims is not None:
        print(f"Image random projection dimensions: {rp_dims}")
    
    # Create single results filename with timestamp (for this execution)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_name:
        results_filename = f"{save_name}_{dataset}_linear_vs_fm_{timestamp}.json"
    else:
        results_filename = f"{dataset}_linear_vs_fm_{timestamp}.json"
    
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
            "linear_probe": [],
            "foundation_model": [],
            "knn": [],
        }
        if use_tabicl:
            all_results["tabicl"] = []
        if cheat_lp:
            all_results["linear_probe_cheat"] = []
            all_results["knn_cheat"] = []
        
        for seed in range(num_seeds):
            print(f"\n[Seed {seed}]")
            results = run_single_seed(
                X_img=X_img,
                y=y,
                seed=seed,
                val_size=val_size,
                test_size=test_size,
                n_estimators=1,
                pca_dims=pca_dims,
                rp_dims=rp_dims,
                train_size=train_size,
                cheat_lp=cheat_lp,
                use_tabicl=use_tabicl,
                verbose=verbose,
            )
            
            for approach_name, metrics in results.items():
                all_results[approach_name].append(metrics)
        
        # Compute and display statistics for this train_size
        print("\n" + "=" * 70)
        print(f"RESULTS FOR TRAIN_SIZE={train_size if train_size else 'ALL'}")
        print("=" * 70)
        
        metric_names = ["accuracy", "precision", "recall", "f1", "auroc"]
        
        print("\nLinear Probe Results:")
        _print_metric_stats(all_results["linear_probe"], metric_names)
        
        print("\nFoundation Model (TabPFN) Results:")
        _print_metric_stats(all_results["foundation_model"], metric_names)
        
        if use_tabicl:
            print("\nTabICL Results:")
            _print_metric_stats(all_results["tabicl"], metric_names)
        
        print("\nKNN Results:")
        _print_metric_stats(all_results["knn"], metric_names)
        
        if cheat_lp:
            print("\nLinear Probe Results (CHEAT):")
            _print_metric_stats(all_results["linear_probe_cheat"], metric_names)
            
            print("\nKNN Results (CHEAT):")
            _print_metric_stats(all_results["knn_cheat"], metric_names)
        
        print("\n" + "=" * 70)
        if use_tabicl:
            print("COMPARISON: Linear Probe vs TabPFN vs TabICL vs KNN")
            print("=" * 70)
            _print_comparison_four_methods(
                all_results["linear_probe"],
                all_results["foundation_model"],
                all_results["tabicl"],
                all_results["knn"],
                metric_names,
            )
        else:
            print("COMPARISON: Linear Probe vs TabPFN vs KNN")
            print("=" * 70)
            _print_comparison_three_methods(
                all_results["linear_probe"],
                all_results["foundation_model"],
                all_results["knn"],
                metric_names,
            )
        
        # Store results for this train_size
        key = str(train_size) if train_size else "all"
        all_results_by_train_size[key] = all_results
        
        # Save intermediate results after each train_size (overwriting same file)
        if save_results_flag:
            config = {
                "dataset": dataset,
                "train_sizes": train_sizes if train_sizes != [None] else None,
                "val_size": val_size,
                "test_size": test_size,
                "num_seeds": num_seeds,
                "pca_dims": pca_dims,
                "rp_dims": rp_dims,
                "cheat_lp": cheat_lp,
                "use_tabicl": use_tabicl,
            }
            save_results(all_results_by_train_size, config, filename=results_filename)
    
    print("\n" + "=" * 70)
    print("Script completed successfully!")
    print("=" * 70)
    
    # Save results if requested
    if save_results_flag:
        config = {
            "dataset": dataset,
            "train_sizes": train_sizes if train_sizes != [None] else None,
            "val_size": val_size,
            "test_size": test_size,
            "num_seeds": num_seeds,
            "pca_dims": pca_dims,
            "rp_dims": rp_dims,
            "cheat_lp": cheat_lp,
            "use_tabicl": use_tabicl,
        }
        save_results(all_results_by_train_size, config, filename=results_filename)



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
    print(f"\n{'Metric':<15} {'Linear Probe':<20} {'TabPFN':<20} {'Difference':<20}")
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


def _print_comparison_three_methods(metrics_list_lp: list, metrics_list_fm: list, metrics_list_knn: list, metric_names: list) -> None:
    """Print comparison between three methods: Linear Probe, Foundation Model, and KNN."""
    print(f"\n{'Metric':<15} {'Linear Probe':<20} {'TabPFN':<20} {'KNN':<20}")
    print("-" * 75)
    
    for metric in metric_names:
        values_lp = [m.get(metric) for m in metrics_list_lp]
        values_fm = [m.get(metric) for m in metrics_list_fm]
        values_knn = [m.get(metric) for m in metrics_list_knn]
        values_lp = [v for v in values_lp if v is not None]
        values_fm = [v for v in values_fm if v is not None]
        values_knn = [v for v in values_knn if v is not None]
        
        if values_lp and values_fm and values_knn:
            mean_lp = np.mean(values_lp)
            std_lp = np.std(values_lp)
            mean_fm = np.mean(values_fm)
            std_fm = np.std(values_fm)
            mean_knn = np.mean(values_knn)
            std_knn = np.std(values_knn)
            
            print(f"{metric:<15} {mean_lp:.4f}±{std_lp:.4f}   {mean_fm:.4f}±{std_fm:.4f}   {mean_knn:.4f}±{std_knn:.4f}")
        else:
            print(f"{metric:<15} N/A                    N/A                    N/A")


def _print_comparison_four_methods(
    metrics_list_lp: list, 
    metrics_list_fm: list, 
    metrics_list_tabicl: list,
    metrics_list_knn: list, 
    metric_names: list
) -> None:
    """Print comparison between four methods: Linear Probe, TabPFN, TabICL, and KNN."""
    print(f"\n{'Metric':<15} {'Linear Probe':<20} {'TabPFN':<20} {'TabICL':<20} {'KNN':<20}")
    print("-" * 95)
    
    for metric in metric_names:
        values_lp = [m.get(metric) for m in metrics_list_lp]
        values_fm = [m.get(metric) for m in metrics_list_fm]
        values_tabicl = [m.get(metric) for m in metrics_list_tabicl]
        values_knn = [m.get(metric) for m in metrics_list_knn]
        values_lp = [v for v in values_lp if v is not None]
        values_fm = [v for v in values_fm if v is not None]
        values_tabicl = [v for v in values_tabicl if v is not None]
        values_knn = [v for v in values_knn if v is not None]
        
        if values_lp and values_fm and values_tabicl and values_knn:
            mean_lp = np.mean(values_lp)
            std_lp = np.std(values_lp)
            mean_fm = np.mean(values_fm)
            std_fm = np.std(values_fm)
            mean_tabicl = np.mean(values_tabicl)
            std_tabicl = np.std(values_tabicl)
            mean_knn = np.mean(values_knn)
            std_knn = np.std(values_knn)
            
            print(f"{metric:<15} {mean_lp:.4f}±{std_lp:.4f}   {mean_fm:.4f}±{std_fm:.4f}   {mean_tabicl:.4f}±{std_tabicl:.4f}   {mean_knn:.4f}±{std_knn:.4f}")
        else:
            print(f"{metric:<15} N/A                    N/A                    N/A                    N/A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare linear probing vs foundation models on image features."
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
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of train+val data to use for validation (used for linear probe model selection)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of all data to use for test set",
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
    parser.add_argument(
        "--cheat-lp",
        action="store_true",
        help="If set, linear probe uses test set for validation (model selection). This is a cheat for improving results.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Optional name prefix to prepend to the results filename (e.g., --save-name exp1 creates exp1_<dataset>_linear_vs_fm_<timestamp>.json).",
    )
    parser.add_argument(
        "--skip-tabicl",
        action="store_true",
        help="If set, TabICL is not run (useful for memory-constrained environments).",
    )
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    
    # Set default features path if not provided
    if args.features_path is None:
        args.features_path = Path(f"extracted_features/{args.dataset}_dinov3_features.pt")
    
    # Convert empty list from nargs="*" to None
    train_sizes = args.train_sizes if args.train_sizes else None
    
    main(
        features_path=args.features_path,
        dataset=args.dataset,
        num_seeds=args.num_seeds,
        val_size=args.val_size,
        test_size=args.test_size,
        pca_dims=args.pca_dims,
        rp_dims=args.rp_dims,
        train_sizes=train_sizes,
        cheat_lp=args.cheat_lp,
        save_name=args.save_name,
        use_tabicl=not args.skip_tabicl,
        save_results_flag=True,
        verbose=args.verbose,
    )
