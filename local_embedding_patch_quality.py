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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLClassifier
from tqdm import tqdm

from local_embedding_experiments import (
    DATASET_PATH,
    FEATURES_DIR,
    ButterflyPatchDataset,
)

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

    Uses the exact same shuffle/split logic as _ButterflyImageDataset so that
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
# Per-image patch prediction
# ---------------------------------------------------------------------------

def _attn_class_scores(
    attn_weights:   torch.Tensor,   # (n_blocks, 1, n_heads, N+P, N)
    support_labels: np.ndarray,     # [N]
    n_classes:      int,
    n_train:        int,
    head: Optional[int] = None,     # None = average all heads
    block: int = -1,                # which ICL block to use
) -> np.ndarray:                    # [P, n_classes]  rows sum to 1
    """Aggregate per-class attention: for each query patch, sum attention
    directed at support rows of each class (last block, one or all heads)."""
    # a: (n_heads, P, N)
    a = attn_weights[block, :, n_train:, :].cpu().float().numpy()
    if head is not None:
        a = a[head]          # (P, N)
    else:
        a = a.mean(axis=0)   # (P, N)

    P = a.shape[0]
    scores = np.zeros((P, n_classes), dtype=np.float32)
    for c in range(n_classes):
        mask = support_labels == c
        if mask.any():
            scores[:, c] = a[:, mask].sum(axis=1)
    # rows already sum to ~1 (softmax over N keys), but renormalise for safety
    row_sums = scores.sum(axis=1, keepdims=True)
    scores /= np.where(row_sums > 0, row_sums, 1.0)
    return scores


def _get_patch_distributions(
    clf:                TabICLClassifier,
    query_features:     np.ndarray,   # [Q, d]
    train_labels:       np.ndarray,   # [N]
    n_classes:          int,
    n_train:            int,
    distribution_source: str = "softmax",   # "softmax" | "attention"
) -> np.ndarray:                            # [Q, n_classes]
    """Return per-patch class distributions from either predict_proba or avg-heads attention."""
    if distribution_source == "attention":
        _, attn_weights = clf.predict(query_features, return_attn=True)
        return _attn_class_scores(attn_weights, train_labels, n_classes, n_train,
                                  head=None, block=-1)
    return clf.predict_proba(query_features)


# ---------------------------------------------------------------------------
# Dataset refinement via patch-quality weighting
# ---------------------------------------------------------------------------

def compute_patch_entropy(patch_probs: np.ndarray) -> np.ndarray:
    """Per-patch Shannon entropy in nats.

    Parameters
    ----------
    patch_probs : np.ndarray, shape [P, n_classes]

    Returns
    -------
    np.ndarray, shape [P]
        Raw entropy values in [0, ln(n_classes)].
    """
    eps = 1e-9
    return -(patch_probs * np.log(patch_probs + eps)).sum(axis=1)



def compute_patch_pooling_weights(
    patch_probs:   np.ndarray,   # [P, n_classes]
    true_label:    int,
    temperature:   float = 1.0,
    weight_method: str   = "logit",
    gamma:         float = 1.0,
) -> np.ndarray:                 # [P]  sums to 1
    """Derive per-patch pooling weights from TabICL softmax predictions.

    Three methods are supported, selected by *weight_method*:

    ``"logit"`` (default)
        1. Extract the true-class probability for each patch  →  p_i in (0, 1).
        2. Take the log-probability                           →  ln(p_i).
        3. Apply a temperature-scaled softmax across patches  →  weights w_i.

        temperature → ∞  : log-probs collapse to zero  →  uniform (mean) pooling
        temperature = 1  : weights ∝ p_true
        temperature → 0  : all weight on the single most-confident patch

    ``"prob"``
        1. Extract the true-class probability for each patch  →  p_i in (0, 1).
        2. Raise to the power gamma                           →  p_i ^ gamma.
        3. Normalise to sum to 1                              →  weights w_i.

        gamma = 1  : weights proportional to true-class probability
        gamma → ∞  : all weight on the single most-confident patch (winner-take-all)
        gamma → 0  : uniform (mean) pooling

    ``"entropy"``
        1. Compute Shannon entropy H_i = -Σ p_c log p_c for each patch.
        2. Convert to a quality score: s_i = ln(C) - H_i  (low entropy → high score).
        3. Apply power normalisation: s_i ^ gamma, then normalise to sum to 1.

        Patches that predict confidently (low entropy) receive higher weight.
        gamma = 1  : weights proportional to ln(C) - H
        gamma → ∞  : all weight on the single most-confident patch (winner-take-all)
        gamma → 0  : uniform (mean) pooling

        Note: *true_label* is not used by this method.

    ``"entropy_logit"``
        1. Compute Shannon entropy H_i = -Σ p_c log p_c for each patch.
        2. Normalise to [0, 1]: score_i = 1 - H_i / ln(C)  (0 = max entropy, 1 = zero entropy).
        3. Apply log to get a logit: logit_i = ln(score_i)  (in (-∞, 0]).
        4. Apply temperature-scaled softmax across patches  →  weights w_i.

        temperature → ∞  : logits collapse to zero  →  uniform (mean) pooling
        temperature → 0  : all weight on the single lowest-entropy patch
        *gamma* is not used by this method; use *temperature* to sharpen/smooth.

        Note: *true_label* is not used by this method.

    ``"combined"``
        Average (arithmetic mean) of the per-patch weights produced by the
        ``"logit"`` and ``"entropy_logit"`` methods, both computed with the
        same *temperature*.  The result is renormalised to sum to 1.

        Combines class-discriminative information (logit uses the true-class
        probability) with class-agnostic confidence information (entropy_logit
        uses overall prediction sharpness).
    """
    if weight_method == "prob":
        true_class_probs = patch_probs[:, true_label].clip(1e-7, 1.0)   # [P]
        powered = true_class_probs ** gamma                               # [P]
        weights = powered / powered.sum()
        return weights                                                     # [P]

    if weight_method == "entropy":
        n_classes = patch_probs.shape[1]
        raw_entropy = compute_patch_entropy(patch_probs)          # [P] in [0, ln(C)]
        scores = (np.log(n_classes) - raw_entropy).clip(0.0)      # [P] in [0, ln(C)]
        powered = scores ** gamma                                  # [P]
        total = powered.sum()
        if total > 0:
            return powered / total
        return np.full(len(powered), 1.0 / len(powered), dtype=np.float32)

    if weight_method == "entropy_logit":
        n_classes = patch_probs.shape[1]
        raw_entropy = compute_patch_entropy(patch_probs)          # [P] in [0, ln(C)]
        scores = (1.0 - raw_entropy / np.log(n_classes)).clip(1e-7, 1.0)  # [P] in (0, 1]
        logits = np.log(scores)                                    # [P] in (-inf, 0]
        logits_scaled = logits / temperature
        logits_scaled -= logits_scaled.max()                       # numerical stability (TODO: Should we predict these, or the unshifted scores?)
        weights = np.exp(logits_scaled)
        weights /= weights.sum()
        return weights

    if weight_method == "combined":
        w_logit        = compute_patch_pooling_weights(patch_probs, true_label, temperature, "logit",        gamma)
        w_entropy_logit = compute_patch_pooling_weights(patch_probs, true_label, temperature, "entropy_logit", gamma)
        combined = (w_logit + w_entropy_logit) / 2.0
        combined /= combined.sum()
        return combined

    # --- default: logit method ---
    true_class_probs = patch_probs[:, true_label].clip(1e-7, 1.0 - 1e-7)  # [P]
    logits = np.log(true_class_probs)
    logits_scaled = logits / temperature
    logits_scaled -= logits_scaled.max()                               # numerical stability
    weights = np.exp(logits_scaled)
    weights /= weights.sum()

    #breakpoint()  # for debugging

    return weights                                                      # [P]


def compute_patch_quality_logits(
    patch_probs:   np.ndarray,   # [P, n_classes]
    true_label:    int,
    temperature:   float = 1.0,
    weight_method: str   = "logit",
    gamma:         float = 1.0,
) -> np.ndarray:                 # [P]  pre-normalization scaled logits
    """Return the pre-normalization score for each patch, matching the
    intermediate value computed inside compute_patch_pooling_weights.

    These values are suitable as Ridge regression targets: fitting a Ridge
    model to predict them from raw DINO features transfers the patch-quality
    signal into a lightweight, label-free scorer.

    Method correspondence
    ---------------------
    ``"logit"``        → log(p_true) / temperature        (pre-stability-shift logit)
    ``"prob"``         → gamma * log(p_true)              (log of unnormalised weight)
    ``"entropy"``      → gamma * log(ln(C) - H)           (log of unnormalised score)
    ``"entropy_logit"``→ log(1 - H/ln(C)) / temperature  (pre-stability-shift logit)
    ``"combined"``     → arithmetic mean of logit + entropy_logit targets
    """
    if weight_method == "prob":
        p = patch_probs[:, true_label].clip(1e-7, 1.0)
        return (gamma * np.log(p)).astype(np.float32)

    if weight_method == "entropy":
        n_classes = patch_probs.shape[1]
        raw_entropy = compute_patch_entropy(patch_probs)               # [P]
        scores = (np.log(n_classes) - raw_entropy).clip(1e-9)          # [P]
        return (gamma * np.log(scores)).astype(np.float32)

    if weight_method == "entropy_logit":
        n_classes = patch_probs.shape[1]
        raw_entropy = compute_patch_entropy(patch_probs)               # [P]
        scores = (1.0 - raw_entropy / np.log(n_classes)).clip(1e-7, 1.0)
        return (np.log(scores) / temperature).astype(np.float32)

    if weight_method == "combined":
        l_logit         = compute_patch_quality_logits(
            patch_probs, true_label, temperature, "logit",         gamma)
        l_entropy_logit = compute_patch_quality_logits(
            patch_probs, true_label, temperature, "entropy_logit", gamma)
        return ((l_logit + l_entropy_logit) / 2.0).astype(np.float32)

    # --- default: logit method ---
    p = patch_probs[:, true_label].clip(1e-7, 1.0 - 1e-7)
    return (np.log(p) / temperature).astype(np.float32)


def group_patches(patches: np.ndarray, patch_group_size: int) -> np.ndarray:
    """Mean-pool spatially neighbouring patches into groups.

    Parameters
    ----------
    patches : np.ndarray
        Shape ``[N, P, D]`` (batch) or ``[P, D]`` (single image).
    patch_group_size : int
        Number of original patches per group.  Must be a perfect square
        (1, 4, 9, 16, …).  ``1`` is the identity (no grouping).

    Returns
    -------
    np.ndarray
        Shape ``[N, P', D]`` or ``[P', D]`` where ``P' = P // patch_group_size``.
        Dtype is preserved from *patches*.
    """
    if patch_group_size == 1:
        return patches

    group_side = int(round(patch_group_size ** 0.5))
    if group_side * group_side != patch_group_size:
        raise ValueError(
            f"patch_group_size must be a perfect square, got {patch_group_size}"
        )

    single = patches.ndim == 2
    if single:
        patches = patches[None]   # [1, P, D]

    N, P, D = patches.shape
    n_side = int(round(P ** 0.5))
    if n_side * n_side != P:
        raise ValueError(f"P={P} is not a perfect square; cannot form a spatial grid")
    if n_side % group_side != 0:
        raise ValueError(
            f"Grid side {n_side} is not divisible by group_side {group_side} "
            f"(patch_group_size={patch_group_size})"
        )

    new_n_side = n_side // group_side
    # Reshape into spatial blocks and average within each block
    grouped = (
        patches
        .reshape(N, new_n_side, group_side, new_n_side, group_side, D)
        .mean(axis=(2, 4))          # [N, new_n_side, new_n_side, D]
    ).reshape(N, new_n_side * new_n_side, D)

    if single:
        grouped = grouped[0]
    return grouped.astype(patches.dtype)


def refine_dataset_features(
    train_patches:    np.ndarray,      # [N, P, D]  raw DINO patch features
    train_labels:     np.ndarray,      # [N]
    support_features: np.ndarray,      # [N, d]  initial mean-pooled (post-PCA) features
    pca:              Optional[PCA],   # PCA fitted on the baseline support set
    n_estimators:     int   = 1,
    temperature:      float = 1.0,
    seed:             int   = 42,
    batch_size:       int   = 100,
    weight_method:    str   = "logit",
    gamma:            float = 1.0,
    mix_lambda:          float = 1.0,
    distribution_source: str  = "softmax",
    fit_ridge:           bool  = False,
    ridge_alpha:         float = 1.0,
    normalize_features:  bool  = False,
) -> tuple[np.ndarray, Optional[PCA], np.ndarray, Optional[Ridge], Optional[StandardScaler]]:
    """Replace mean-pooled support features with quality-weighted patch pooling,
    and optionally fit a Ridge regressor on the patch quality logits — all in a
    single TabICL forward pass over the training set.

    Flow
    ----
    1. Query TabICL in batches of *batch_size* images against the *initial*
       mean-pooled support to get per-patch class distributions.
    2. Derive pooling weights via *weight_method* (see compute_patch_pooling_weights).
       If *fit_ridge* is True, also compute the pre-normalisation quality logits
       (compute_patch_quality_logits) and store them alongside the raw patch features.
    3. Apply weights to the **raw DINO** patch features → repooled [N, D] features.
    4. Mix with original mean-pooled raw features:
       ``mixed_raw = mix_lambda * repooled_raw + (1 - mix_lambda) * mean_pooled_raw``
    5. Re-fit PCA on *mixed_raw*.
    6. If *fit_ridge* is True, fit Ridge(alpha=*ridge_alpha*) on the collected
       [N*P, D] raw features and [N*P] logit targets.

    Returns
    -------
    mixed : np.ndarray, shape [N, d]
    new_pca : PCA or None
    weights_all : np.ndarray, shape [N, P]
    ridge_model : Ridge or None
    feature_scaler : StandardScaler or None
    """
    N, P, D = train_patches.shape
    repooled_raw = np.zeros((N, D), dtype=np.float32)
    weights_all  = np.zeros((N, P), dtype=np.float32)

    if fit_ridge:
        all_features = np.empty((N * P, D), dtype=np.float32)
        all_targets  = np.empty(N * P,      dtype=np.float32)

    # Fit one shared classifier (support set is fixed for all queries)
    n_classes = int(train_labels.max()) + 1
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(support_features, train_labels)

    for batch_start in tqdm(range(0, N, batch_size),
                            desc="Computing patch quality scores", unit="batch"):
        batch_end     = min(batch_start + batch_size, N)
        batch_patches = train_patches[batch_start:batch_end]   # [B, P, D]
        B = batch_end - batch_start

        query_raw: np.ndarray = batch_patches.reshape(B * P, D)   # [B*P, D]
        query_features: np.ndarray = (
            pca.transform(query_raw) if pca is not None else query_raw
        )                                                           # [B*P, d]

        probs = _get_patch_distributions(
            clf, query_features, train_labels, n_classes, N, distribution_source
        )                                                           # [B*P, n_classes]
        probs = probs.reshape(B, P, -1)                            # [B, P, n_classes]

        for j in range(B):
            idx        = batch_start + j
            true_label = int(train_labels[idx])
            weights = compute_patch_pooling_weights(
                probs[j], true_label, temperature, weight_method, gamma,
            )                                                       # [P]
            weights_all[idx] = weights
            repooled_raw[idx] = (weights[:, None] * batch_patches[j]).sum(axis=0)  # [D]

            if fit_ridge:
                all_features[idx * P:(idx + 1) * P] = batch_patches[j]
                all_targets [idx * P:(idx + 1) * P] = compute_patch_quality_logits(
                    probs[j], true_label, temperature, weight_method, gamma,
                )

    # Mix repooled features with original mean-pooled raw features
    if mix_lambda < 1.0:
        mean_pooled_raw = train_patches.mean(axis=1)               # [N, D]
        mixed_raw = (
            mix_lambda * repooled_raw + (1.0 - mix_lambda) * mean_pooled_raw
        ).astype(np.float32)                                       # [N, D]
    else:
        mixed_raw = repooled_raw

    # Re-fit PCA on the mixed raw features so downstream transforms stay valid
    new_pca: Optional[PCA] = None
    if pca is not None:
        new_pca = PCA(n_components=pca.n_components_, random_state=seed)
        mixed = new_pca.fit_transform(mixed_raw).astype(np.float32)   # [N, d]
    else:
        mixed = mixed_raw

    ridge_model: Optional[Ridge] = None
    feature_scaler: Optional[StandardScaler] = None
    if fit_ridge:
        if normalize_features:
            print("[ridge] Fitting StandardScaler on training patches ...")
            feature_scaler = StandardScaler()
            all_features = feature_scaler.fit_transform(all_features)

        print(f"[ridge] Fitting Ridge(alpha={ridge_alpha}) on {N * P:,} patch samples "
              f"(D={D}, method={weight_method}) ...")
        ridge_model = Ridge(alpha=ridge_alpha)
        ridge_model.fit(all_features, all_targets)
        print(f"[ridge] Train R²: {ridge_model.score(all_features, all_targets):.4f}")

        # Redo support repooling using Ridge-predicted quality logits
        print("[ridge] Repooling support set with Ridge-predicted weights ...")
        flat_patches = train_patches.reshape(N * P, D)
        if feature_scaler is not None:
            flat_patches = feature_scaler.transform(flat_patches)
        all_logits = ridge_model.predict(flat_patches).reshape(N, P).astype(np.float32)  # [N, P]
        all_logits -= all_logits.max(axis=1, keepdims=True)            # numerical stability
        exp_logits  = np.exp(all_logits)                               # [N, P]
        weights_ridge = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # [N, P]
        repooled_raw  = (weights_ridge[:, :, None] * train_patches).sum(axis=1)  # [N, D]

        # Recompute mixed_raw and PCA with the Ridge-repooled features
        if mix_lambda < 1.0:
            mean_pooled_raw = train_patches.mean(axis=1)
            mixed_raw = (
                mix_lambda * repooled_raw + (1.0 - mix_lambda) * mean_pooled_raw
            ).astype(np.float32)
        else:
            mixed_raw = repooled_raw

        if pca is not None:
            new_pca = PCA(n_components=pca.n_components_, random_state=seed)
            mixed   = new_pca.fit_transform(mixed_raw).astype(np.float32)
        else:
            mixed = mixed_raw

    return mixed, new_pca, weights_all, ridge_model, feature_scaler


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _upscale_grid(flat: np.ndarray, n_side: int, patch_size: int) -> np.ndarray:
    """Upscale a flat [P] patch array to a pixel grid [H, W]."""
    return np.repeat(np.repeat(flat.reshape(n_side, n_side), patch_size, axis=0), patch_size, axis=1)


def _add_prob_overlay(
    ax: plt.Axes,
    fig: plt.Figure,
    img_rgb: np.ndarray,
    pixel_grid: np.ndarray,   # [H, W] raw values
    title: str,
    alpha: float,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Overlay a heatmap on top of the image with a colourbar.
    
    vmin/vmax: if provided, use these for color scale; otherwise auto-scale.
    """
    ax.imshow(img_rgb)
    im = ax.imshow(pixel_grid, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _visualise_image(
    image: Image.Image,
    patch_probs:       np.ndarray,            # [P, n_classes]  softmax distribution
    true_label:        int,
    idx_to_class:      dict[int, str],
    n_classes:         int,
    patch_size:        int   = 16,
    alpha:             float = 0.55,
    temperature:       float = 1.0,
    gamma:             float = 1.0,
    weight_method:     str   = "logit",
    attn_avg_scores:   Optional[np.ndarray] = None,   # [P, n_classes]  attention distribution
    ridge_pred_logits: Optional[np.ndarray] = None,   # [P]  Ridge-predicted quality logits
) -> plt.Figure:
    """Figure with 1–2 rows of overlay panels, one row per distribution source.
    Each row: original | P(true) | logit weights | entropy | entropy/entropy-logit weights
    Row 2 (if attn_avg_scores): same panels using the attention-based distribution.
    """
    P      = len(patch_probs)
    n_side = int(round(P ** 0.5))

    # Summary stats from softmax distribution (for suptitle)
    correct_probs     = patch_probs[:, true_label]
    mean_correct_prob = float(correct_probs.mean())
    patch_preds       = patch_probs.argmax(axis=1)
    unique, counts    = np.unique(patch_preds, return_counts=True)
    modal_class       = unique[counts.argmax()]
    consensus_frac    = counts.max() / P
    mean_entropy      = float(compute_patch_entropy(patch_probs).mean() / np.log(n_classes))

    img_rgb = np.array(image.resize((n_side * patch_size, n_side * patch_size)))

    def _up(vals: np.ndarray) -> np.ndarray:
        return _upscale_grid(vals, n_side, patch_size)

    def _dist_panels(dist: np.ndarray, label: str) -> list[tuple[str, Optional[np.ndarray], dict]]:
        """Build the 6 panels (original + 5 overlays) for a [P, n_classes] distribution."""
        p_true          = dist[:, true_label]
        e_norm          = compute_patch_entropy(dist) / np.log(n_classes)
        w_logit         = compute_patch_pooling_weights(dist, true_label, temperature, "logit",         gamma)
        w_entropy_logit = compute_patch_pooling_weights(dist, true_label, temperature, "entropy_logit", gamma)
        w_combined      = compute_patch_pooling_weights(dist, true_label, temperature, "combined",      gamma)
        if weight_method in ("entropy_logit", "combined"):
            w_entropy_panel     = w_entropy_logit
            entropy_panel_title = "Entropy-logit pooling weights"
        else:
            w_entropy_panel     = compute_patch_pooling_weights(dist, true_label, temperature, "entropy", gamma)
            entropy_panel_title = "Entropy pooling weights"
        return [
            (f"Original image\n[{label}]", None, {}),
            (f"P(true class)  (mean={p_true.mean():.3f})",
             p_true, {"vmin": 0.0, "vmax": 1.0}),
            ("Logit pooling weights",
             w_logit, {"vmin": w_logit.min(), "vmax": w_logit.max()}),
            (f"Entropy (normalised)  (mean={e_norm.mean():.3f})",
             e_norm, {"cmap": "RdYlGn_r", "vmin": e_norm.min(), "vmax": e_norm.max()}),
            (entropy_panel_title,
             w_entropy_panel, {"vmin": w_entropy_panel.min(), "vmax": w_entropy_panel.max()}),
            ("Combined pooling weights",
             w_combined, {"vmin": w_combined.min(), "vmax": w_combined.max()}),
        ]

    all_rows = [_dist_panels(patch_probs, "Softmax")]
    if attn_avg_scores is not None:
        all_rows.append(_dist_panels(attn_avg_scores, "Attention"))

    if ridge_pred_logits is not None:
        ridge_panel = (
            f"Ridge pooling weights  (max={ridge_pred_logits.max():.4f})",
            ridge_pred_logits,
            {"cmap": "RdYlGn",
             "vmin": ridge_pred_logits.min(),
             "vmax": ridge_pred_logits.max()},
        )
        all_rows = [row + [ridge_panel] for row in all_rows]

    n_cols = max(len(r) for r in all_rows)
    fig, axes = plt.subplots(len(all_rows), n_cols,
                             figsize=(n_cols * 4.5, len(all_rows) * 5),
                             squeeze=False)

    fig.suptitle(
        f"True class: {idx_to_class[true_label]!r}  |  "
        f"mean P(true): {mean_correct_prob:.3f}  |  "
        f"modal pred: {idx_to_class[modal_class]!r} ({consensus_frac:.0%})  |  "
        f"mean entropy: {mean_entropy:.3f}",
        fontsize=11,
    )

    for row_idx, row_panels in enumerate(all_rows):
        for col_idx, (title, vals, kwargs) in enumerate(row_panels):
            ax = axes[row_idx, col_idx]
            if vals is None:
                ax.imshow(img_rgb)
                ax.set_title(title)
                ax.axis("off")
            else:
                _add_prob_overlay(ax, fig, img_rgb, _up(vals), title, alpha, **kwargs)
        for col_idx in range(len(row_panels), n_cols):
            axes[row_idx, col_idx].axis("off")

    fig.tight_layout()
    return fig


def _summary_figure(results: list[dict]) -> plt.Figure:
    """Bar chart of per-image mean correct-class probability."""
    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.2), 4))
    xs     = np.arange(len(results))
    probs  = [r["mean_correct_prob"] for r in results]
    labels = [r["class_name"]        for r in results]
    ax.bar(xs, probs, color="steelblue")
    ax.axhline(np.mean(probs), color="red", linestyle="--", label=f"mean={np.mean(probs):.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean P(true class) across patches")
    ax.set_ylim(0, 1)
    ax.set_title("Per-image patch prediction quality")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main experiment helpers
# ---------------------------------------------------------------------------

def _compute_accuracy(
    support_features: np.ndarray,   # [N_train, d]
    support_labels:   np.ndarray,   # [N_train]
    test_patches:     np.ndarray,   # [N_test, P, D]
    test_labels:      np.ndarray,   # [N_test]
    pca:              Optional[PCA],
    n_estimators:     int          = 1,
    seed:             int          = 42,
    ridge_model:      Optional[Ridge] = None,
    feature_scaler:   Optional[StandardScaler] = None,
) -> float:
    """Accuracy of TabICL on the held-out test set.

    When *ridge_model* is provided, test images are represented as
    Ridge-repooled patch features (predict quality logits → softmax →
    weighted pool), matching how the refined support set was built.
    Otherwise, test images are mean-pooled across patches.
    """
    N_test, P, D = test_patches.shape
    if ridge_model is not None:
        flat = test_patches.reshape(N_test * P, D)
        if feature_scaler is not None:
            flat = feature_scaler.transform(flat)
        logits = ridge_model.predict(flat).reshape(N_test, P).astype(np.float32)  # [N_test, P]
        logits -= logits.max(axis=1, keepdims=True)                    # numerical stability
        exp_l  = np.exp(logits)
        weights = exp_l / exp_l.sum(axis=1, keepdims=True)             # [N_test, P]
        test_query: np.ndarray = (weights[:, :, None] * test_patches).sum(axis=1)  # [N_test, D]
    else:
        test_query = test_patches.mean(axis=1)                         # [N_test, D]
    if pca is not None:
        test_query = pca.transform(test_query)                         # [N_test, d]
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(support_features, support_labels)
    preds = clf.predict(test_query)
    return float((preds == test_labels).mean())


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
    temperature:         float = 1.0,
    gamma:               float = 1.0,
    weight_method:       str   = "logit",
    visualize_attention: bool  = False,
    ridge_model:         Optional[Ridge] = None,   # fitted Ridge quality model
    feature_scaler:      Optional[StandardScaler] = None,
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

            attn_avg_scores = None
            if visualize_attention:
                N_train = len(support_features)
                _, attn_weights = clf.predict(query_features, return_attn=True)
                attn_avg_scores = _attn_class_scores(
                    attn_weights, train_labels, n_classes, N_train, head=None, block=-1
                )

            ridge_pred_logits = None
            if ridge_model is not None:
                patches_for_ridge = patches_i
                if feature_scaler is not None:
                    patches_for_ridge = feature_scaler.transform(patches_i)
                raw = ridge_model.predict(patches_for_ridge).astype(np.float32)  # [P]
                raw -= raw.max()                                                   # numerical stability
                exp = np.exp(raw)
                ridge_pred_logits = exp / exp.sum()                       # [P] softmax weights

            img = Image.open(image_paths[img_idx]).convert("RGB")
            fig = _visualise_image(
                img, probs, true_label, idx_to_class,
                n_classes=n_classes,
                patch_size=patch_size,
                temperature=temperature,
                gamma=gamma,
                weight_method=weight_method,
                attn_avg_scores=attn_avg_scores,
                ridge_pred_logits=ridge_pred_logits,
            )
            out_path = (
                split_out_dir
                / f"patch_quality_{i:02d}_img{img_idx}_{class_name.replace(' ', '_')}.png"
            )
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        sfig     = _summary_figure(results)
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
    features_dir:  Path = FEATURES_DIR,
    dataset_path:  Path = DATASET_PATH,
    n_sample:      int  = 8,
    n_train:       Optional[int] = None,
    n_estimators:  int  = 1,
    pca_dim:       Optional[int] = 128,
    seed:          int  = 42,
    output_dir:    Path = Path("patch_quality_results"),
    patch_size:    int  = 16,
    patch_group_size: int = 1,
    refine:        bool  = False,
    temperature:   float = 1.0,
    batch_size:    int   = 10,
    weight_method: str   = "logit",
    gamma:         float = 1.0,
    mix_lambda:          float = 1.0,
    distribution_source: str  = "softmax",
    visualize_attention: bool = False,
    fit_ridge:          bool  = False,
    ridge_alpha:        float = 1.0,
    normalize_features: bool  = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pre-extracted training features (always the support set) ---
    train_ds = ButterflyPatchDataset(features_dir, split="train")
    train_patches = train_ds.features.numpy()   # [N, P, D]
    train_labels  = train_ds.labels.numpy()     # [N]

    if n_train is not None and n_train < len(train_labels):
        rng_sub = np.random.RandomState(seed)
        sub_idx = rng_sub.choice(len(train_labels), size=n_train, replace=False)
        sub_idx.sort()
        train_patches = train_patches[sub_idx]
        train_labels  = train_labels[sub_idx]
        print(f"[info] Training set subsampled: {n_train} / {train_ds.features.shape[0]} images")

    # --- Load test features ---
    test_ds = ButterflyPatchDataset(features_dir, split="test")
    test_patches = test_ds.features.numpy()   # [N_test, P, D]
    test_labels  = test_ds.labels.numpy()     # [N_test]

    # --- Group patches in DINO space (before any PCA) ---
    if patch_group_size > 1:
        group_side = int(round(patch_group_size ** 0.5))
        P_orig        = train_patches.shape[1]
        train_patches = group_patches(train_patches, patch_group_size)
        test_patches  = group_patches(test_patches,  patch_group_size)
        print(f"[info] Patch grouping {group_side}×{group_side}: "
              f"P {P_orig} → {train_patches.shape[1]} patches per image")

    N_train, _P, D = train_patches.shape
    effective_patch_size = patch_size * int(round(patch_group_size ** 0.5))

    # --- Support set: mean-pool → [N_train, D] ---
    baseline_support = train_patches.mean(axis=1)   # [N_train, D]

    # --- Optional PCA fitted on support set (applied to all queries) ---
    pca: Optional[PCA] = None
    if pca_dim is not None:
        n_comp = min(pca_dim, N_train, D)
        pca = PCA(n_components=n_comp, random_state=seed)
        baseline_support = pca.fit_transform(baseline_support)   # [N_train, n_comp]
        print(f"[info] PCA: {D}D → {n_comp}D")

    # --- Single quality-scoring pass: optionally refine support and/or fit Ridge ---
    ridge_model:     Optional[Ridge]          = None
    feature_scaler:  Optional[StandardScaler] = None
    refined_support: Optional[np.ndarray]     = None
    refined_pca:     Optional[PCA]            = None
    if fit_ridge or refine:
        print(f"\n[info] Computing patch quality scores "
              f"(method={weight_method}, source={distribution_source}, "
              f"temperature={temperature}, gamma={gamma}"
              + (f", mix_lambda={mix_lambda}" if refine else "")
              + (f", ridge_alpha={ridge_alpha}" if fit_ridge else "") + ") ...")
        refined_support, refined_pca, _, ridge_model, feature_scaler = refine_dataset_features(
            train_patches, train_labels, baseline_support,
            pca=pca, n_estimators=n_estimators,
            temperature=temperature, seed=seed, batch_size=batch_size,
            weight_method=weight_method, gamma=gamma, mix_lambda=mix_lambda,
            distribution_source=distribution_source,
            fit_ridge=fit_ridge, ridge_alpha=ridge_alpha,
            normalize_features=normalize_features,
        )
        if fit_ridge:
            ridge_path = output_dir / "ridge_quality_model.joblib"
            joblib.dump(ridge_model, ridge_path)
            print(f"[ridge] Model saved → {ridge_path}")

    # --- Reconstruct image paths for both splits ---
    train_image_paths, _, idx_to_class = _get_image_paths(dataset_path, split="train", seed=seed)
    test_image_paths,  _, _            = _get_image_paths(dataset_path, split="test",  seed=seed)

    rng = np.random.RandomState(seed)
    train_sample_idx = rng.choice(len(train_labels), size=min(n_sample, len(train_labels)), replace=False)
    test_sample_idx  = rng.choice(len(test_labels),  size=min(n_sample, len(test_labels)),  replace=False)

    split_configs = [
        ("train", train_patches, train_labels, train_image_paths, train_sample_idx),
        ("test",  test_patches,  test_labels,  test_image_paths,  test_sample_idx),
    ]

    common_kwargs = dict(
        train_labels=train_labels,
        split_configs=split_configs,
        idx_to_class=idx_to_class,
        pca=pca,
        n_estimators=n_estimators,
        patch_size=effective_patch_size,
        seed=seed,
        output_dir=output_dir,
        temperature=temperature,
        gamma=gamma,
        weight_method=weight_method,
        visualize_attention=visualize_attention,
        ridge_model=ridge_model,
        feature_scaler=feature_scaler,
    )

    # --- Baseline accuracy + visual eval ---
    baseline_acc = _compute_accuracy(
        baseline_support, train_labels, test_patches, test_labels,
        pca=pca, n_estimators=n_estimators, seed=seed,
    )
    print(f"\n[baseline] test accuracy: {baseline_acc:.4f}")
    baseline_mean_probs = _run_visual_eval("baseline", baseline_support, **common_kwargs)

    if not refine:
        return

    # --- Refined accuracy + visual eval (support already computed above) ---
    print(f"[info] Refinement complete. Support shape: {refined_support.shape}")

    # Always evaluate with mean-pooled queries so the query representation is
    # held constant and only the support changes.
    refined_acc_mean = _compute_accuracy(
        refined_support, train_labels, test_patches, test_labels,
        pca=refined_pca, n_estimators=n_estimators, seed=seed,
        ridge_model=None,
    )
    print(f"\n[refined]  test accuracy (mean-pooled queries):  {refined_acc_mean:.4f}")

    # When a Ridge model is available, also evaluate with Ridge-repooled queries.
    refined_acc_ridge: Optional[float] = None
    if ridge_model is not None:
        refined_acc_ridge = _compute_accuracy(
            refined_support, train_labels, test_patches, test_labels,
            pca=refined_pca, n_estimators=n_estimators, seed=seed,
            ridge_model=ridge_model, feature_scaler=feature_scaler,
        )
        print(f"[refined]  test accuracy (ridge-pooled queries): {refined_acc_ridge:.4f}")

    refined_common_kwargs = {**common_kwargs, "pca": refined_pca}
    refined_mean_probs = _run_visual_eval("refined", refined_support, **refined_common_kwargs)

    # --- Side-by-side comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Test accuracy  baseline={baseline_acc:.4f}"
          f"  refined/mean-q={refined_acc_mean:.4f}  Δ={refined_acc_mean - baseline_acc:+.4f}")
    if refined_acc_ridge is not None:
        print(f"  Test accuracy  baseline={baseline_acc:.4f}"
              f"  refined/ridge-q={refined_acc_ridge:.4f}  Δ={refined_acc_ridge - baseline_acc:+.4f}")
    for split_name in baseline_mean_probs:
        b = baseline_mean_probs[split_name]
        r = refined_mean_probs.get(split_name, float("nan"))
        print(f"  Mean P(true) [{split_name:5s}]  baseline={b:.3f}  refined={r:.3f}"
              f"  Δ={r - b:+.3f}")
    print("=" * 60)


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
    p.add_argument("--patch-size",       type=int,   default=16)
    p.add_argument("--patch-group-size", type=int,   default=1,
                   help="Number of neighbouring patches to mean-pool into one group "
                        "before passing to TabICL (must be a perfect square: 1, 4, 9, 16, …). "
                        "1 = no grouping (default).")
    p.add_argument("--refine",        action="store_true",
                   help="Refine support features with patch-quality weighting before eval")
    p.add_argument("--temperature",    type=float, default=1.0,
                   help="Softmax temperature for patch pooling weights "
                        "(logit method only; large → uniform/mean pooling, small → peaked on best patch)")
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
    p.add_argument("--distribution-source", type=str, default="softmax",
                   choices=["softmax", "attention"],
                   help="Source of the per-patch class distribution used to derive pooling weights: "
                        "'softmax' (default) uses TabICL predict_proba; "
                        "'attention' uses avg-heads attention class scores")
    p.add_argument("--visualize-attention", action="store_true",
                   help="Include attention-based class score visualizations (head 3 and avg-heads)")
    p.add_argument("--fit-ridge",    action="store_true",
                   help="Fit a Ridge regression model to predict patch quality logits from raw DINO "
                        "features; model is saved to <output-dir>/ridge_quality_model.joblib")
    p.add_argument("--ridge-alpha",  type=float, default=1.0,
                   help="Regularisation strength for the Ridge quality model (default 1.0)")
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
        patch_group_size=args.patch_group_size,
        refine=args.refine,
        temperature=args.temperature,
        batch_size=args.batch_size,
        weight_method=args.weight_method,
        gamma=args.gamma,
        mix_lambda=args.mix_lambda,
        distribution_source=args.distribution_source,
        visualize_attention=args.visualize_attention,
        fit_ridge=args.fit_ridge,
        ridge_alpha=args.ridge_alpha,
        normalize_features=args.normalize_features,
    )
