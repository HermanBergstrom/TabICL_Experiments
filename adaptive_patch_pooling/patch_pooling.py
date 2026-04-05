"""Core patch quality scoring and pooling algorithms.

Functions here are pure NumPy/sklearn — no I/O, no visualisation.
"""

from __future__ import annotations

import math
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
        Shape ``[N, P', D]`` or ``[P', D]`` where
        ``P' = ceil(sqrt(P) / group_side) ** 2``.
        When the grid side is not divisible by ``group_side``, boundary groups
        cover fewer patches (equivalent to ``ceil_mode=True`` avg-pooling).
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

    new_n_side = math.ceil(n_side / group_side)

    if n_side % group_side == 0:
        # Fast path: exact divisibility — no padding needed
        grouped = (
            patches
            .reshape(N, new_n_side, group_side, new_n_side, group_side, D)
            .mean(axis=(2, 4))          # [N, new_n_side, new_n_side, D]
        ).reshape(N, new_n_side * new_n_side, D)
    else:
        # Pad spatial dims to next multiple of group_side, then average only
        # valid elements (ceil_mode equivalent — boundary groups are smaller).
        pad = new_n_side * group_side - n_side
        x = patches.astype(np.float32).reshape(N, n_side, n_side, D)
        x = np.pad(x, ((0, 0), (0, pad), (0, pad), (0, 0)), mode="constant", constant_values=0.0)
        # Count valid patches per output cell (shared across N and D)
        valid = np.ones((n_side, n_side), dtype=np.float32)
        valid = np.pad(valid, ((0, pad), (0, pad)), mode="constant", constant_values=0.0)
        counts = valid.reshape(new_n_side, group_side, new_n_side, group_side).sum(axis=(1, 3))  # [new_n_side, new_n_side]
        # Sum within each group then divide by valid count
        grouped_sum = (
            x.reshape(N, new_n_side, group_side, new_n_side, group_side, D)
            .sum(axis=(2, 4))           # [N, new_n_side, new_n_side, D]
        )
        grouped = (grouped_sum / counts[None, :, :, None]).reshape(N, new_n_side * new_n_side, D)

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
        mean_pooled_raw = raw_patches.astype(np.float32).mean(axis=1)
        mixed_raw = (mix_lambda * repooled_raw + (1.0 - mix_lambda) * mean_pooled_raw).astype(np.float32)
    else:
        mixed_raw = repooled_raw

    if pca is not None:
        new_pca = PCA(n_components=pca.n_components_, random_state=seed)
        return new_pca.fit_transform(mixed_raw).astype(np.float32), new_pca
    return mixed_raw, None


# ---------------------------------------------------------------------------
# Full refinement pass
# ---------------------------------------------------------------------------

def refine_dataset_features(
    train_patches:      np.ndarray,      # [N, P, D]  raw DINO patch features
    train_labels:       np.ndarray,      # [N]
    support_features:   np.ndarray,      # [N, d]  initial mean-pooled (post-PCA) features
    pca:                Optional[PCA],   # PCA fitted on the baseline support set
    n_estimators:       int   = 1,
    temperature:        float = 1.0,
    seed:               int   = 42,
    batch_size:         int   = 100,
    weight_method:      str   = "logit",
    gamma:              float = 1.0,
    mix_lambda:         float = 1.0,
    ridge_alpha:        float = 1.0,
    normalize_features: bool  = False,
    max_query_rows:        Optional[int]       = None,
    use_random_subsampling: bool               = False,
    aoe_mask:              Optional[np.ndarray] = None,  # [N] bool; True = absence-of-evidence class
    aoe_handling:          str                 = "filter",  # "filter" | "entropy"
) -> tuple[np.ndarray, Optional[PCA], np.ndarray, Ridge, Optional[StandardScaler], TabICLClassifier]:
    """Refine mean-pooled support features with Ridge-predicted quality-weighted pooling.

    A TabICL classifier fitted on the *input* support is used solely to generate
    quality-logit training targets for a Ridge regressor.  The Ridge model then
    predicts per-patch quality logits from raw DINO features and drives the final
    pooling — making the pooling label-free and fast at inference time.

    If *aoe_mask* is provided, its effect on steps 2–4 depends on *aoe_handling*:

    ``"filter"``
        AoE-class images are excluded from the TabICL forward pass and Ridge
        fitting entirely.  The Ridge model sees only non-AoE patches.

    ``"entropy"``
        AoE-class images are included, but their per-patch quality logits are
        computed with the ``"entropy_logit"`` method regardless of *weight_method*.
        This lets the Ridge model learn AoE-patch quality without relying on labels.

    In both cases, AoE-class images are still included in the Ridge-based support
    pooling (step 5) and in the returned refined support.

    Flow
    ----
    1. Fit a ``TabICLClassifier`` on *support_features* (fixed scorer for this stage).
    2. Collect ``(patch_feature, quality_logit)`` pairs for Ridge fitting:

       - **Default** (``max_query_rows`` is ``None`` or active rows ≤ ``max_query_rows``):
         forward all active patch rows in batches of *batch_size*.
       - **Subsampled** (active rows > ``max_query_rows``): draw *max_query_rows*
         ``(image, patch)`` pairs uniformly at random from active images and
         forward them in a single pass.  The fraction of rows forwarded is printed.

    3. Optionally fit a ``StandardScaler`` on the collected patch features
       (``normalize_features=True``).
    4. Fit ``Ridge(alpha=ridge_alpha)`` on the collected pairs.
    5. Predict quality logits for **all** patches of **all** images with Ridge
       (full-image pooling including AoE-class images, regardless of step 2).
    6. Apply softmax weights derived from Ridge logits → ``repooled_raw [N, D]``.
    7. Mix with mean-pooled raw features (``mix_lambda``) and re-fit PCA.

    Returns
    -------
    mixed : np.ndarray, shape [N, d]
    new_pca : PCA or None
    weights_ridge : np.ndarray, shape [N, P]   (Ridge softmax weights, full images)
    ridge_model : Ridge
    feature_scaler : StandardScaler or None
    clf : TabICLClassifier  (scorer fitted on the *input* support_features)
    """
    N, P, D = train_patches.shape

    # Determine which images contribute to Ridge fitting and their per-image method.
    # "filter": only non-AoE images; all use weight_method.
    # "entropy": all images; AoE images use "entropy_logit" instead of weight_method.
    if aoe_mask is not None and aoe_handling == "filter":
        active_indices = np.where(~aoe_mask)[0]
    else:
        active_indices = np.arange(N)
    N_active          = len(active_indices)
    active_patches    = train_patches[active_indices]   # [N_active, P, D]
    active_labels     = train_labels[active_indices]    # [N_active]
    active_total_rows = N_active * P

    # Per-image effective weight method (None = all images use weight_method)
    if aoe_mask is not None and aoe_handling == "entropy":
        active_is_aoe = aoe_mask[active_indices]   # [N_active] bool
    else:
        active_is_aoe = None

    def _eff_method(local_idx: int) -> str:
        if active_is_aoe is not None and active_is_aoe[local_idx]:
            return "entropy_logit"
        return weight_method

    # Fit one shared classifier (support set is fixed for all queries this stage)
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(support_features, train_labels)

    # Decide forward-pass strategy:
    #   one_pass=True  → forward a contiguous block of rows in a single predict_proba call
    #                    (either all active rows, or a random subset when subsampling)
    #   one_pass=False → forward in batches of batch_size images (fallback / default)
    exceeded = max_query_rows is not None and active_total_rows > max_query_rows
    one_pass = max_query_rows is not None and (not exceeded or use_random_subsampling)

    if one_pass:
        if exceeded:
            # Subsampled: draw max_query_rows (image, patch) pairs from active images
            n_fwd    = max_query_rows
            aoe_note = f" (aoe_handling={aoe_handling})" if aoe_mask is not None else ""
            print(f"[sampling] Subsampling {n_fwd:,} / {active_total_rows:,} patch-group rows "
                  f"({100 * n_fwd / active_total_rows:.1f}%) for Ridge fitting{aoe_note}")
            rng           = np.random.RandomState(seed)
            sampled_flat  = rng.choice(active_total_rows, size=n_fwd, replace=False)
            sampled_flat.sort()                          # sort → group by image for searchsorted
            local_img_idx = sampled_flat // P            # index into active_patches
            patch_idx_all = sampled_flat % P
            query_raw     = active_patches[local_img_idx, patch_idx_all].astype(np.float32)
            img_boundaries = np.searchsorted(local_img_idx, np.arange(N_active + 1))
        else:
            # All active rows in one pass; sequential layout matches reshape order
            n_fwd          = active_total_rows
            query_raw      = active_patches.reshape(active_total_rows, D).astype(np.float32)
            img_boundaries = np.arange(N_active + 1) * P     # [0, P, 2P, ..., N_active*P]

        query_features = pca.transform(query_raw) if pca is not None else query_raw
        probs_flat     = clf.predict_proba(query_features)   # [n_fwd, n_classes]

        all_features = query_raw
        all_targets  = np.empty(n_fwd, dtype=np.float32)

        for idx in range(N_active):
            start, end = int(img_boundaries[idx]), int(img_boundaries[idx + 1])
            if start == end:
                continue
            all_targets[start:end] = compute_patch_quality_logits(
                probs_flat[start:end], int(active_labels[idx]),
                temperature, _eff_method(idx), gamma,
            )

    else:
        # Batched loop: used when max_query_rows is None, or when cap is exceeded
        # but --use-random-subsampling was not requested.
        all_features = np.empty((active_total_rows, D), dtype=np.float32)
        all_targets  = np.empty(active_total_rows,      dtype=np.float32)
        row_ptr      = 0

        for batch_start in tqdm(range(0, N, batch_size),
                                desc="Computing patch quality scores", unit="batch"):
            batch_end = min(batch_start + batch_size, N)
            # For "filter" mode, exclude AoE images from the batch.
            # For "entropy" mode (and no AoE), include all images.
            if aoe_mask is not None and aoe_handling == "filter":
                active_in_batch = [j for j in range(batch_start, batch_end) if not aoe_mask[j]]
                if not active_in_batch:
                    continue
                batch_patches_arr = train_patches[active_in_batch]   # [B_a, P, D]
                batch_labels_arr  = train_labels[active_in_batch]
                batch_is_aoe      = None
            else:
                batch_patches_arr = train_patches[batch_start:batch_end]   # [B, P, D]
                batch_labels_arr  = train_labels[batch_start:batch_end]
                batch_is_aoe      = (aoe_mask[batch_start:batch_end]
                                     if aoe_mask is not None else None)

            B_a = len(batch_patches_arr)
            query_raw      = batch_patches_arr.reshape(B_a * P, D)
            query_features = pca.transform(query_raw) if pca is not None else query_raw

            probs = clf.predict_proba(query_features).reshape(B_a, P, -1)   # [B_a, P, n_classes]

            for j in range(B_a):
                eff = ("entropy_logit"
                       if batch_is_aoe is not None and batch_is_aoe[j]
                       else weight_method)
                all_features[row_ptr * P:(row_ptr + 1) * P] = batch_patches_arr[j]
                all_targets [row_ptr * P:(row_ptr + 1) * P] = compute_patch_quality_logits(
                    probs[j], int(batch_labels_arr[j]), temperature, eff, gamma,
                )
                row_ptr += 1

    # --- Fit Ridge on collected (patch_feature, quality_logit) pairs ---
    feature_scaler: Optional[StandardScaler] = None
    if normalize_features:
        print("[ridge] Fitting StandardScaler on training patches ...")
        feature_scaler = StandardScaler()
        all_features = feature_scaler.fit_transform(all_features)

    print(f"[ridge] Fitting Ridge(alpha={ridge_alpha}) on {len(all_features):,} patch samples "
          f"(D={D}, method={weight_method}) ...")
    ridge_model = Ridge(alpha=ridge_alpha)
    ridge_model.fit(all_features, all_targets)
    print(f"[ridge] Train R²: {ridge_model.score(all_features, all_targets):.4f}")

    # --- Pool all training images with Ridge weights (always full images) ---
    print("[ridge] Pooling support set with Ridge-predicted weights ...")
    weights_ridge = _ridge_pool_weights(train_patches, ridge_model, feature_scaler)   # [N, P]
    repooled_raw  = (weights_ridge[:, :, None] * train_patches).sum(axis=1)           # [N, D]

    mixed, new_pca = _mix_and_project(repooled_raw, train_patches, mix_lambda, pca, seed)

    return mixed, new_pca, weights_ridge, ridge_model, feature_scaler, clf
