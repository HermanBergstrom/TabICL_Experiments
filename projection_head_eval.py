import argparse
from pathlib import Path
import json
import time

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import accuracy_score

from tabicl import TabICLClassifier
from experiments import _load_data, DATASET_CONFIGS, _evaluate_model

from imagenet_projection.projection_methods import build_projection_module
from imagenet_projection.save_checkpoints import extract_projection_state_dict


def load_butterfly_features(
    features_dir: Path, split: str = "train", seed: int = 42, val_fraction: float = 0.2
):
    """Load butterfly features from extracted .pt files and optionally split train into train/val.
    
    Args:
        features_dir: Directory containing butterfly_*_dinov3_features.pt files
        split: "train", "val", or "test"
        seed: Random seed for train/val split
        val_fraction: Fraction of training data to use for validation
    
    Returns:
        features: [N, D] numpy array of image features
        labels: [N] numpy array of labels
    """
    features_dir = Path(features_dir)
    
    # Load train or test features
    if split in ["train", "val"]:
        pt_file = features_dir / "butterfly_train_dinov3_features.pt"
    else:  # split == "test"
        pt_file = features_dir / "butterfly_test_dinov3_features.pt"
    
    if not pt_file.exists():
        raise FileNotFoundError(f"Features file not found: {pt_file}")
    
    ckpt = torch.load(pt_file, map_location="cpu")
    features = ckpt["features"].numpy()
    labels = ckpt["labels"].numpy()
    
    # Split train into train/val if needed
    if split in ["train", "val"]:
        n_train = len(features)
        n_val = int(n_train * val_fraction)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_train)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        if split == "train":
            features = features[train_indices]
            labels = labels[train_indices]
        else:  # split == "val"
            features = features[val_indices]
            labels = labels[val_indices]
    
    return features, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="petfinder")
    parser.add_argument("--checkpoint", type=Path, default=Path("model_weights/hypernetwork_checkpoints/ortho_reg_k768_hdim768_log_sampling_2000_6000_minc50_maxc200_20260324_230038/best_heldout.pt"))
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-tabular", action="store_true", help="If set, concatenates tabular features with projected image features.")
    parser.add_argument("--max-train", type=int, default=None, help="Max train samples")
    parser.add_argument("--features-dir", type=Path, default=None, help="Path to extracted features directory (for image-only datasets like butterfly)")
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    meta = ckpt["projection_meta"]
    print(ckpt.keys())

    
    # Adjust dictionary to match what `build_projection_module` expects
    # older checkpoints might have 'hyper_use_random_projection_init'
    use_random_init = meta.get("hyper_use_random_projection_init", False)
    if "hyper_attn_use_pos_embed" not in meta:
        meta["hyper_attn_use_pos_embed"] = False

    module, _ = build_projection_module(
        method=meta["projection_method"],
        input_dim=meta["input_dim"],
        output_dim=meta["output_dim"],
        head_type=meta["head_type"],
        hidden_dim=meta["head_hidden_dim"],
        dropout=meta["head_dropout"],
        zca_epsilon=meta["zca_epsilon"],
        hyper_top_k=meta["hyper_top_k"],
        hyper_encoder_type=meta["hyper_encoder_type"],
        hyper_attn_heads=meta["hyper_attn_heads"],
        hyper_attn_layers=meta["hyper_attn_layers"],
        hyper_attn_use_pos_embed=meta.get("hyper_attn_use_pos_embed", False),
        use_random_projection_init=use_random_init,
        device=torch.device(args.device)
    )
    
    state_dict = extract_projection_state_dict(ckpt)
    #module.load_state_dict(state_dict)
    module.eval()
    module.to(args.device)
    print("Hypernetwork loaded successfully.")

    # Load data based on dataset type
    has_tabular = True
    
    if args.dataset == "butterfly":
        # Load butterfly features from extracted .pt files
        features_dir = args.features_dir or Path("extracted_features")
        print(f"Loading {args.dataset} features from {features_dir}...")
        X_img_train, y_train = load_butterfly_features(features_dir, split="train", seed=args.seed)
        X_img_val, y_val = load_butterfly_features(features_dir, split="val", seed=args.seed)
        X_img_test, y_test = load_butterfly_features(features_dir, split="test", seed=args.seed)
        X_tab_train = np.zeros((len(X_img_train), 1))  # Dummy tabular features
        X_tab_val = np.zeros((len(X_img_val), 1))
        X_tab_test = np.zeros((len(X_img_test), 1))
        has_tabular = False
    else:
        # Load standard datasets via experiments.py
        ds_cfg = DATASET_CONFIGS[args.dataset]
        print(f"Loading {args.dataset} data...")
        _, splits, _ = _load_data(
            dataset=args.dataset,
            data_dir=ds_cfg["data_dir"],
            module_path=ds_cfg["module_path"],
            need_images=True,
        )
        
        X_tab_train, X_img_train, _, y_train = splits["train"]
        X_tab_val, X_img_val, _, y_val = splits["val"]
        X_tab_test, X_img_test, _, y_test = splits["test"]

    if args.max_train and args.max_train < len(X_img_train):
        print(f"Subsampling training data to {args.max_train} samples")
        indices = np.random.RandomState(args.seed).choice(len(X_img_train), args.max_train, replace=False)
        X_tab_train = X_tab_train[indices]
        X_img_train = X_img_train[indices]
        y_train = y_train[indices]

    target_dim = meta["output_dim"]

    results = {}

    def evaluate_model_on_features(method_name, train_feats, val_feats, test_feats):
        print(f"\n--- Evaluating with {method_name} ---")
        if args.use_tabular and has_tabular:
            train_feats = np.concatenate([X_tab_train, train_feats], axis=1)
            val_feats = np.concatenate([X_tab_val, val_feats], axis=1)
            test_feats = np.concatenate([X_tab_test, test_feats], axis=1)
            
        clf = TabICLClassifier(n_estimators=args.n_estimators, random_state=args.seed)
        t0 = time.time()
        print(train_feats.shape)
        clf.fit(train_feats, y_train)
        fit_time = time.time() - t0
        
        y_pred = clf.predict(test_feats)
        acc = accuracy_score(y_test, y_pred)
        results[method_name] = acc
        print(f"{method_name} Test Accuracy: {acc:.4f} (Fit time: {fit_time:.2f}s)")


    # 0. Baseline: No Projection
    print(f"\nEvaluating No Projection (Raw) -> {X_img_train.shape[1]}")
    evaluate_model_on_features("NoProjection", X_img_train, X_img_val, X_img_test)

    # 1. Baseline: PCA
    print(f"\nFitting PCA -> {target_dim}")
    pca = PCA(n_components=target_dim, random_state=args.seed)
    pca_train = pca.fit_transform(X_img_train)
    pca_val = pca.transform(X_img_val)
    pca_test = pca.transform(X_img_test)

    print(X_img_train.shape, X_img_val.shape, X_img_test.shape)
    evaluate_model_on_features("PCA", pca_train, pca_val, pca_test)

    # 2. Baseline: Random Projection
    print(f"\nFitting Random Projection -> {target_dim}")
    rp = GaussianRandomProjection(n_components=target_dim, random_state=args.seed)
    rp_train = rp.fit_transform(X_img_train)
    rp_val = rp.transform(X_img_val)
    rp_test = rp.transform(X_img_test)
    evaluate_model_on_features("RandomProjection", rp_train, rp_val, rp_test)

    # 3. Hypernetwork
    print(f"\nProjecting with Spectral Hypernetwork -> {target_dim}")
    with torch.no_grad():
        all_features = np.concatenate([X_img_train, X_img_val, X_img_test], axis=0)
        all_features_t = torch.tensor(all_features, device=args.device, dtype=torch.float32)
        support_indices = torch.arange(len(X_img_train), device=args.device)
        
        if meta["projection_method"] == "projection_head":
             projected_t = module(features=all_features_t)
        else:
             projected_t = module(features=all_features_t, support_indices=support_indices)

        if isinstance(projected_t, tuple):
             projected_t = projected_t[0]

        projected = projected_t.cpu().numpy()

    hn_train = projected[:len(X_img_train)]
    hn_val = projected[len(X_img_train):len(X_img_train)+len(X_img_val)]
    hn_test = projected[-len(X_img_test):]
    
    evaluate_model_on_features("SpectralHypernetwork", hn_train, hn_val, hn_test)

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
