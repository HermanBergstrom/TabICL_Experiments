"""Core patch quality scoring and pooling algorithms.

Functions here are pure NumPy/sklearn — no I/O, no visualisation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLClassifier
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Patch entropy and pooling weights
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
        logits_scaled -= logits_scaled.max()                       # numerical stability
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


# ---------------------------------------------------------------------------
# Patch grouping
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Ridge softmax pooling helper
# ---------------------------------------------------------------------------

def _ridge_pool_weights(
    patches:        np.ndarray,               # [N, P, D]
    ridge_model:    Ridge,
    feature_scaler: Optional[StandardScaler],
) -> np.ndarray:                              # [N, P]  softmax weights summing to 1
    """Compute per-patch softmax pooling weights from a fitted Ridge model."""
    N, P, D = patches.shape
    flat = patches.reshape(N * P, D)
    if feature_scaler is not None:
        flat = feature_scaler.transform(flat)
    logits = ridge_model.predict(flat).reshape(N, P).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)   # numerical stability
    exp_l   = np.exp(logits)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Mix-lambda blending and PCA refit
# ---------------------------------------------------------------------------

def _mix_and_project(
    repooled_raw: np.ndarray,    # [N, D]  quality-weighted pooled features
    raw_patches:  np.ndarray,    # [N, P, D]  original patches (for mean-pool fallback)
    mix_lambda:   float,
    pca:          Optional[PCA],
    seed:         int,
) -> tuple[np.ndarray, Optional[PCA]]:
    """Apply mix-lambda blending with mean-pool and re-fit PCA.

    Returns the projected features [N, d] and the newly fitted PCA (or None).
    """
    if mix_lambda < 1.0:
        mean_pooled_raw = raw_patches.mean(axis=1)
        mixed_raw = (mix_lambda * repooled_raw + (1.0 - mix_lambda) * mean_pooled_raw).astype(np.float32)
    else:
        mixed_raw = repooled_raw

    if pca is not None:
        new_pca = PCA(n_components=pca.n_components_, random_state=seed)
        return new_pca.fit_transform(mixed_raw).astype(np.float32), new_pca
    return mixed_raw, None


# ---------------------------------------------------------------------------
# Batched quality-weighted pooling
# ---------------------------------------------------------------------------

def _pool_features_with_clf(
    grouped_patches: np.ndarray,     # [N, P', D]
    labels:          np.ndarray,     # [N]   true label per image (for weight computation)
    clf:             TabICLClassifier,
    scoring_pca:     Optional[PCA],  # PCA that was fitted on the support
    temperature:     float,
    weight_method:   str,
    gamma:           float,
    batch_size:      int,
    desc:            str = "Pooling",
) -> tuple[np.ndarray, np.ndarray]:     # (repooled_raw [N, D], weights [N, P'])
    """Pool grouped patches using quality weights from a pre-fitted TabICL classifier.

    Scoring is performed in the PCA-projected space that matches the support, while
    the weighted pooling is applied to the raw (pre-PCA) grouped patch features so
    that distances in the output space are not distorted by a PCA fitted on a
    different set of vectors.
    """
    N, P, D = grouped_patches.shape
    repooled_raw = np.zeros((N, D), dtype=np.float32)
    weights_all  = np.zeros((N, P), dtype=np.float32)

    for batch_start in tqdm(range(0, N, batch_size), desc=desc, unit="batch", leave=False):
        batch_end = min(batch_start + batch_size, N)
        batch     = grouped_patches[batch_start:batch_end]   # [B, P', D]
        B         = batch_end - batch_start

        query_raw:  np.ndarray = batch.reshape(B * P, D)
        query_feat: np.ndarray = (
            scoring_pca.transform(query_raw) if scoring_pca is not None else query_raw
        )

        probs = clf.predict_proba(query_feat)          # [B*P', n_classes]
        probs = probs.reshape(B, P, -1)                # [B, P', n_classes]

        for j in range(B):
            idx        = batch_start + j
            true_label = int(labels[idx])
            weights    = compute_patch_pooling_weights(
                probs[j], true_label, temperature, weight_method, gamma
            )
            weights_all[idx]  = weights
            repooled_raw[idx] = (weights[:, None] * batch[j]).sum(axis=0)

    return repooled_raw, weights_all


# ---------------------------------------------------------------------------
# Full refinement pass
# ---------------------------------------------------------------------------

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
    mix_lambda:       float = 1.0,
    fit_ridge:        bool  = False,
    ridge_alpha:      float = 1.0,
    normalize_features: bool = False,
) -> tuple[np.ndarray, Optional[PCA], np.ndarray, Optional[Ridge], Optional[StandardScaler], TabICLClassifier]:
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
    clf : TabICLClassifier  (the classifier fitted on the *input* support_features)
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

        probs = clf.predict_proba(query_features)                   # [B*P, n_classes]
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

    mixed, new_pca = _mix_and_project(repooled_raw, train_patches, mix_lambda, pca, seed)

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
        weights_ridge = _ridge_pool_weights(train_patches, ridge_model, feature_scaler)
        repooled_raw  = (weights_ridge[:, :, None] * train_patches).sum(axis=1)  # [N, D]
        mixed, new_pca = _mix_and_project(repooled_raw, train_patches, mix_lambda, pca, seed)

    return mixed, new_pca, weights_all, ridge_model, feature_scaler, clf
