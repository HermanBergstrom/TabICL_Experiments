"""Patch-quality evaluation: are individual DINOv3 patch embeddings discriminative?

For each sampled training image we:
  1. Build a support set from *mean-pooled* training embeddings (one row per image).
  2. Feed every patch of the sampled image as an individual TabICL query.
  3. Record how many patches predict the correct class.
  4. Visualise the results overlaid on the original image.

Usage:
    python local_embedding_patch_quality.py \\
        [--n-sample 8] [--n-estimators 1] [--pca-dim 128] \\
        [--seed 42] [--output-dir patch_quality_results]
"""

from __future__ import annotations

import argparse
import csv as _csv
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLClassifier
from torch.utils.data import Dataset
from tqdm import tqdm

from adaptive_patch_pooling.patch_pooling import (
    _pool_features_with_clf,
    _ridge_pool_weights,
    group_patches,
    refine_dataset_features,
)
from adaptive_patch_pooling.patch_visualisation import summary_figure, visualise_image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_PATH = Path("/project/aip-rahulgk/hermanb/datasets/butterfly-image-classification")
FEATURES_DIR = Path("extracted_features")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ButterflyPatchDataset(Dataset):
    """Loads pre-extracted DINOv3 patch features for the butterfly dataset.

    Each sample is ``(patches, label)`` where:
        patches : float tensor of shape [num_patches, embed_dim]  (e.g. [196, 768])
        label   : int class index

    Args:
        features_dir: Directory containing the .pt files produced by the
            feature extraction script.
        split: ``"train"`` or ``"test"``.
        dtype: Cast features to this dtype on load (default: float32 for model compat).
    """

    def __init__(
        self,
        features_dir: Path = FEATURES_DIR,
        split: str = "train",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        features_dir = Path(features_dir)
        pt_path = features_dir / f"butterfly_{split}_dinov3_patch_features.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Patch features not found at {pt_path}. "
                "Run the feature extraction script first."
            )

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.features: torch.Tensor = ckpt["features"].to(dtype)   # [N, P, D]
        self.labels:   torch.Tensor = ckpt["labels"].long()        # [N]

        N, P, D = self.features.shape
        print(
            f"[info] ButterflyPatchDataset ({split}): "
            f"N={N}  num_patches={P}  embed_dim={D}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Reconstruct training-split file paths (must match extraction order)
# ---------------------------------------------------------------------------

def _get_image_paths(
    dataset_path: Path = DATASET_PATH,
    split: str = "train",
    seed: int = 42,
    test_fraction: float = 0.2,
) -> tuple[list[Path], list[int], dict[int, str]]:
    """Return (paths, labels, idx_to_class) for the requested split.

    Uses the exact same shuffle/split logic as ButterflyPatchDataset so that
    index i here corresponds to row i in the .pt feature file.
    """
    csv_path = Path(dataset_path) / "Training_set.csv"
    img_dir  = Path(dataset_path) / "train"

    rows: list[tuple[str, str]] = []
    all_labels: list[str] = []
    with csv_path.open(newline="") as f:
        for row in _csv.DictReader(f):
            rows.append((row["filename"], row["label"]))
            all_labels.append(row["label"])

    class_to_idx: dict[str, int] = {
        lbl: i for i, lbl in enumerate(sorted(set(all_labels)))
    }
    idx_to_class: dict[int, str] = {v: k for k, v in class_to_idx.items()}

    n_total = len(rows)
    rng     = np.random.RandomState(seed)
    idx     = rng.permutation(n_total)
    n_test  = int(n_total * test_fraction)
    selected = [rows[i] for i in (idx[:n_test] if split == "test" else idx[n_test:])]

    paths  = [img_dir / fname   for fname, _   in selected]
    labels = [class_to_idx[lbl] for _,    lbl  in selected]
    return paths, labels, idx_to_class


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _compute_accuracy_from_features(
    support_features: np.ndarray,   # [N_train, d]
    support_labels:   np.ndarray,   # [N_train]
    query_features:   np.ndarray,   # [N_test, d]  already projected into support space
    query_labels:     np.ndarray,   # [N_test]
    n_estimators:     int = 1,
    seed:             int = 42,
) -> float:
    """Classify pre-projected query features against a support set."""
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(support_features, support_labels)
    return float((clf.predict(query_features) == query_labels).mean())


def _compute_accuracy(
    support_features: np.ndarray,   # [N_train, d]
    support_labels:   np.ndarray,   # [N_train]
    test_patches:     np.ndarray,   # [N_test, P, D]
    test_labels:      np.ndarray,   # [N_test]
    pca:              Optional[PCA],
    n_estimators:     int = 1,
    seed:             int = 42,
) -> float:
    """Accuracy of TabICL on the held-out test set using mean-pooled test queries."""
    test_query = test_patches.mean(axis=1)   # [N_test, D]
    if pca is not None:
        test_query = pca.transform(test_query).astype(np.float32)
    return _compute_accuracy_from_features(
        support_features, support_labels, test_query, test_labels, n_estimators, seed
    )


# ---------------------------------------------------------------------------
# Visual evaluation loop
# ---------------------------------------------------------------------------

def _run_visual_eval(
    tag:              str,
    support_features: np.ndarray,      # [N_train, d]
    train_labels:     np.ndarray,      # [N_train]
    split_configs:    list,            # list of (split_name, patches, labels, image_paths, sample_idx)
    idx_to_class:     dict[int, str],
    pca:              Optional[PCA],
    n_estimators:     int,
    patch_size:       int,
    seed:             int,
    output_dir:       Path,
    temperature:      float = 1.0,
    gamma:            float = 1.0,
    weight_method:    str   = "logit",
    ridge_model:      Optional[Ridge] = None,
    feature_scaler:   Optional[StandardScaler] = None,
) -> dict[str, float]:
    """Run the patch-quality visual evaluation for one support set variant.

    Saves per-image heatmaps and a summary bar chart under output_dir/tag/<split>/.
    Returns a dict mapping split_name → mean correct-class probability.
    """
    n_classes = int(train_labels.max()) + 1
    mean_probs: dict[str, float] = {}

    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(support_features, train_labels)

    for split_name, patches_all, labels_all, image_paths, sample_idx in split_configs:
        split_out_dir = output_dir / tag / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []
        bar = tqdm(enumerate(sample_idx), total=len(sample_idx),
                   desc=f"[{tag}] {split_name}", unit="img")
        for i, img_idx in bar:
            patches_i  = patches_all[img_idx]       # [P, D]
            true_label = int(labels_all[img_idx])
            class_name = idx_to_class[true_label]

            query_features: np.ndarray = (
                pca.transform(patches_i) if pca is not None else patches_i
            )

            probs = clf.predict_proba(query_features)   # [P, n_classes]

            correct_probs     = probs[:, true_label]
            mean_correct_prob = float(correct_probs.mean())
            patch_preds       = probs.argmax(axis=1)
            unique, counts    = np.unique(patch_preds, return_counts=True)
            modal_class       = unique[counts.argmax()]

            bar.set_postfix(
                true=class_name,
                P_true=f"{mean_correct_prob:.3f}",
                modal=idx_to_class[modal_class],
            )

            results.append(
                dict(img_idx=img_idx, label=true_label, class_name=class_name,
                     mean_correct_prob=mean_correct_prob)
            )

            ridge_pred_logits = None
            if ridge_model is not None:
                ridge_pred_logits = _ridge_pool_weights(
                    patches_i[None], ridge_model, feature_scaler
                )[0]   # [P]

            img = Image.open(image_paths[img_idx]).convert("RGB")
            fig = visualise_image(
                img, probs, true_label, idx_to_class,
                n_classes=n_classes,
                patch_size=patch_size,
                temperature=temperature,
                gamma=gamma,
                weight_method=weight_method,
                ridge_pred_logits=ridge_pred_logits,
            )
            out_path = (
                split_out_dir
                / f"patch_quality_{i:02d}_img{img_idx}_{class_name.replace(' ', '_')}.png"
            )
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        sfig     = summary_figure(results)
        sum_path = split_out_dir / "summary.png"
        sfig.savefig(sum_path, dpi=150, bbox_inches="tight")
        plt.close(sfig)

        mean_prob = float(np.mean([r["mean_correct_prob"] for r in results]))
        mean_probs[split_name] = mean_prob
        tqdm.write(f"[{tag}] {split_name}  mean P(true)={mean_prob:.3f}  summary → {sum_path}")

    return mean_probs


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_patch_quality_eval(
    features_dir:      Path          = FEATURES_DIR,
    dataset_path:      Path          = DATASET_PATH,
    n_sample:          int           = 8,
    n_train:           Optional[int] = None,
    n_estimators:      int           = 1,
    pca_dim:           Optional[int] = 128,
    seed:              int           = 42,
    output_dir:        Path          = Path("patch_quality_results"),
    patch_size:        int           = 16,
    patch_group_sizes: list          = [1],
    refine:            bool          = False,
    temperature:       float         = 1.0,
    batch_size:        int           = 10,
    weight_method:     str           = "logit",
    gamma:             float         = 1.0,
    mix_lambda:        float         = 1.0,
    fit_ridge:         bool          = False,
    ridge_alpha:       float         = 1.0,
    normalize_features: bool         = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pre-extracted patch features (always at original resolution) ---
    train_ds      = ButterflyPatchDataset(features_dir, split="train")
    train_patches = train_ds.features.numpy()   # [N, P, D]
    train_labels  = train_ds.labels.numpy()     # [N]

    if n_train is not None and n_train < len(train_labels):
        rng_sub = np.random.RandomState(seed)
        sub_idx = rng_sub.choice(len(train_labels), size=n_train, replace=False)
        sub_idx.sort()
        train_patches = train_patches[sub_idx]
        train_labels  = train_labels[sub_idx]
        print(f"[info] Training set subsampled: {n_train} / {train_ds.features.shape[0]} images")

    test_ds      = ButterflyPatchDataset(features_dir, split="test")
    test_patches = test_ds.features.numpy()   # [N_test, P, D]
    test_labels  = test_ds.labels.numpy()     # [N_test]

    N_train, _P, D = train_patches.shape
    n_classes      = int(train_labels.max()) + 1
    n_stages       = len(patch_group_sizes)

    # Normalise temperature / ridge_alpha → one value per stage.
    # Scalar or single-element list → broadcast to all stages.
    # Multi-element list → must match n_stages exactly.
    def _broadcast(val, label: str) -> list:
        if isinstance(val, (int, float)):
            return [float(val)] * n_stages
        vals = list(val)
        if len(vals) == 1:
            return vals * n_stages
        if len(vals) != n_stages:
            raise ValueError(
                f"{label}: {len(vals)} value(s) given for {n_stages} stage(s) in "
                f"--patch-group-sizes; pass a single value (broadcast to all stages) "
                f"or exactly {n_stages} value(s)."
            )
        return vals

    temperatures = _broadcast(temperature,  "--temperature")
    ridge_alphas = _broadcast(ridge_alpha,  "--ridge-alpha")

    # --- Baseline support: mean-pool original patches → optional PCA ---
    baseline_support_raw = train_patches.mean(axis=1)   # [N_train, D]
    pca: Optional[PCA] = None
    if pca_dim is not None:
        n_comp = min(pca_dim, N_train, D)
        pca    = PCA(n_components=n_comp, random_state=seed)
        baseline_support = pca.fit_transform(baseline_support_raw).astype(np.float32)
        print(f"[info] PCA: {D}D → {n_comp}D")
    else:
        baseline_support = baseline_support_raw

    # --- Reconstruct image paths and draw fixed visualisation sample indices ---
    train_image_paths, _, idx_to_class = _get_image_paths(dataset_path, split="train", seed=seed)
    test_image_paths,  _, _            = _get_image_paths(dataset_path, split="test",  seed=seed)

    rng              = np.random.RandomState(seed)
    train_sample_idx = rng.choice(len(train_labels), size=min(n_sample, len(train_labels)), replace=False)
    test_sample_idx  = rng.choice(len(test_labels),  size=min(n_sample, len(test_labels)),  replace=False)

    # --- Baseline: accuracy + visual eval at original patch resolution ---
    baseline_acc = _compute_accuracy(
        baseline_support, train_labels, test_patches, test_labels,
        pca=pca, n_estimators=n_estimators, seed=seed,
    )
    print(f"\n[baseline] test accuracy: {baseline_acc:.4f}")

    split_configs_orig = [
        ("train", train_patches, train_labels, train_image_paths, train_sample_idx),
        ("test",  test_patches,  test_labels,  test_image_paths,  test_sample_idx),
    ]
    baseline_mean_probs = _run_visual_eval(
        "baseline", baseline_support, train_labels, split_configs_orig, idx_to_class,
        pca=pca, n_estimators=n_estimators, patch_size=patch_size,
        seed=seed, output_dir=output_dir,
        temperature=temperatures[0], gamma=gamma, weight_method=weight_method,
        ridge_model=None, feature_scaler=None,
    )

    if not refine:
        return

    # ---------------------------------------------------------------------------
    # Iterative multi-scale refinement
    # ---------------------------------------------------------------------------
    # Each stage groups the original DINO patches at a given resolution, visualises
    # patch quality scores under the *current* (pre-refinement) support, refines the
    # support via quality-weighted pooling, then evaluates accuracy using the same
    # clf and pooling that drove the refinement (ensuring query pooling matches
    # how training embeddings were constructed).
    # ---------------------------------------------------------------------------

    current_support = baseline_support
    current_pca     = pca
    all_results: list[tuple[str, float, dict]] = [
        ("baseline", baseline_acc, baseline_mean_probs)
    ]

    for stage_idx, group_size in enumerate(patch_group_sizes):
        stage_temp  = temperatures[stage_idx]
        stage_alpha = ridge_alphas[stage_idx]
        group_side   = int(round(group_size ** 0.5))
        eff_patch_sz = patch_size * group_side
        tag          = f"iter_{stage_idx}_g{group_size}"

        print(f"\n[{tag}] group_size={group_size}  ({group_side}×{group_side} patches per group)  "
              f"T={stage_temp}" + (f"  ridge_alpha={stage_alpha}" if fit_ridge else ""))

        train_grouped = group_patches(train_patches, group_size)   # [N, P', D]
        test_grouped  = group_patches(test_patches,  group_size)   # [N_test, P', D]
        P_grouped     = train_grouped.shape[1]

        # -- Visualise patch quality under the *input* support (before refinement) --
        # This matches exactly what will drive the pooling weights this stage.
        split_configs_iter = [
            ("train", train_grouped, train_labels, train_image_paths, train_sample_idx),
            ("test",  test_grouped,  test_labels,  test_image_paths,  test_sample_idx),
        ]
        iter_mean_probs = _run_visual_eval(
            tag, current_support, train_labels, split_configs_iter, idx_to_class,
            pca=current_pca, n_estimators=n_estimators, patch_size=eff_patch_sz,
            seed=seed, output_dir=output_dir,
            temperature=stage_temp, gamma=gamma, weight_method=weight_method,
            ridge_model=None, feature_scaler=None,
        )

        # -- Refine support (clf fitted on current_support internally) --
        print(f"[{tag}] Refining support "
              f"(method={weight_method}, T={stage_temp}) ...")
        new_support, new_pca, _weights, ridge_model, feature_scaler, scoring_clf = \
            refine_dataset_features(
                train_grouped, train_labels, current_support, current_pca,
                n_estimators=n_estimators, temperature=stage_temp, seed=seed,
                batch_size=batch_size, weight_method=weight_method, gamma=gamma,
                mix_lambda=mix_lambda, fit_ridge=fit_ridge, ridge_alpha=stage_alpha,
                normalize_features=normalize_features,
            )

        if fit_ridge and ridge_model is not None:
            ridge_path = output_dir / f"ridge_quality_model_{tag}.joblib"
            joblib.dump(ridge_model, ridge_path)
            print(f"[ridge] Model saved → {ridge_path}")

        # -- Pool test queries using the *same* clf and pooling as the training pass --
        # Ridge takes precedence when available (matches how training was repooled).
        if ridge_model is not None:
            w_ridge       = _ridge_pool_weights(test_grouped, ridge_model, feature_scaler)
            test_repooled = (w_ridge[:, :, None] * test_grouped).sum(axis=1)   # [N_test, D]
        else:
            test_repooled, _ = _pool_features_with_clf(
                test_grouped, test_labels,
                scoring_clf, current_pca,
                stage_temp, weight_method, gamma, batch_size,
                desc=f"[{tag}] Pooling test queries",
            )

        test_query = (
            new_pca.transform(test_repooled).astype(np.float32)
            if new_pca is not None else test_repooled
        )

        # -- Evaluate accuracy with quality-pooled test queries --
        iter_acc = _compute_accuracy_from_features(
            new_support, train_labels, test_query, test_labels,
            n_estimators=n_estimators, seed=seed,
        )
        print(f"[{tag}] test accuracy (quality-pooled queries): {iter_acc:.4f}")

        all_results.append((tag, iter_acc, iter_mean_probs))
        current_support = new_support
        current_pca     = new_pca

    # --- Summary table ---
    col_w = max(len(r[0]) for r in all_results) + 2
    print("\n" + "=" * (col_w + 42))
    print("ITERATIVE REFINEMENT SUMMARY")
    print("=" * (col_w + 42))
    print(f"  {'Stage':<{col_w}}  {'Test Acc':>10}  {'Δ Acc':>8}  "
          f"{'P(true)/train':>14}  {'P(true)/test':>13}")
    print("-" * (col_w + 42))
    for stage_name, acc, mean_probs in all_results:
        delta_str = "" if stage_name == "baseline" else f"{acc - baseline_acc:+.4f}"
        print(
            f"  {stage_name:<{col_w}}  {acc:>10.4f}  {delta_str:>8}"
            f"  {mean_probs.get('train', float('nan')):>14.3f}"
            f"  {mean_probs.get('test', float('nan')):>13.3f}"
        )
    print("=" * (col_w + 42))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch quality evaluation with TabICL")
    p.add_argument("--features-dir",  type=Path,  default=FEATURES_DIR)
    p.add_argument("--dataset-path",  type=Path,  default=DATASET_PATH)
    p.add_argument("--n-sample",      type=int,   default=8)
    p.add_argument("--n-train",       type=int,   default=None,
                   help="Limit the support set to this many training images (random subsample)")
    p.add_argument("--n-estimators",  type=int,   default=1)
    p.add_argument("--pca-dim",       type=int,   default=128)
    p.add_argument("--no-pca",        action="store_true",
                   help="Disable PCA (use full 768-D embeddings)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output-dir",    type=Path,  default=Path("patch_quality_results"))
    p.add_argument("--patch-size",        type=int,   default=16)
    p.add_argument("--patch-group-sizes", type=int,   nargs="+",  default=[1],
                   help="Ordered list of patch group sizes for iterative refinement "
                        "(must each be a perfect square: 1, 4, 9, 16, …). "
                        "A single value runs one refinement stage at that group size. "
                        "Multiple values (e.g. --patch-group-sizes 16 4 1) chain stages "
                        "from coarse to fine, each reusing the previous stage's support. "
                        "1 = no grouping (individual patches).")
    p.add_argument("--refine",        action="store_true",
                   help="Refine support features with patch-quality weighting before eval")
    p.add_argument("--temperature",    type=float, nargs="+",  default=[1.0],
                   help="Softmax temperature for patch pooling weights. "
                        "Pass one value to use it for all stages, or one value per entry in "
                        "--patch-group-sizes to set a different temperature per stage. "
                        "Large → uniform/mean pooling; small → peaked on best patch.")
    p.add_argument("--batch-size",     type=int,   default=100,
                   help="Number of images per TabICL call during refinement")
    p.add_argument("--weight-method",  type=str,   default="logit",
                   choices=["logit", "prob", "entropy", "entropy_logit", "combined"],
                   help="How to derive patch pooling weights from TabICL probabilities: "
                        "'logit' (default) uses inverse-sigmoid + temperature-softmax; "
                        "'prob' uses power normalisation p^gamma; "
                        "'entropy' uses (ln(C)-H)^gamma power normalisation; "
                        "'entropy_logit' maps normalised entropy to [0,1], applies log, "
                        "then temperature-scaled softmax (uses --temperature, not --gamma); "
                        "'combined' averages the logit and entropy_logit weights post-softmax")
    p.add_argument("--gamma",          type=float, default=1.0,
                   help="Power exponent for the 'prob' weight method "
                        "(1 → proportional to prob, ∞ → winner-take-all, 0 → uniform)")
    p.add_argument("--mix-lambda",     type=float, default=1.0,
                   help="Interpolation weight between refined and mean-pooled embeddings "
                        "(1.0 → fully refined, 0.0 → fully mean-pooled; requires --refine)")
    p.add_argument("--fit-ridge",    action="store_true",
                   help="Fit a Ridge regression model to predict patch quality logits from raw DINO "
                        "features; model is saved to <output-dir>/ridge_quality_model.joblib")
    p.add_argument("--ridge-alpha",  type=float, nargs="+",  default=[1.0],
                   help="Regularisation strength for the Ridge quality model. "
                        "Pass one value to use it for all stages, or one value per entry in "
                        "--patch-group-sizes to set a different alpha per stage.")
    p.add_argument("--normalize-features", action="store_true",
                   help="Fit a StandardScaler on training patches before Ridge fitting "
                        "(normalises each feature dimension across all N×P patches; "
                        "recommended for Ridge regression); scaler is applied at predict time too")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_patch_quality_eval(
        features_dir=args.features_dir,
        dataset_path=args.dataset_path,
        n_sample=args.n_sample,
        n_train=args.n_train,
        n_estimators=args.n_estimators,
        pca_dim=None if args.no_pca else args.pca_dim,
        seed=args.seed,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        patch_group_sizes=args.patch_group_sizes,
        refine=args.refine,
        temperature=args.temperature,
        batch_size=args.batch_size,
        weight_method=args.weight_method,
        gamma=args.gamma,
        mix_lambda=args.mix_lambda,
        fit_ridge=args.fit_ridge,
        ridge_alpha=args.ridge_alpha,
        normalize_features=args.normalize_features,
    )
