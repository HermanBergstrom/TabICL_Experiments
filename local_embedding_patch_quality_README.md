# Adaptive Patch Pooling

Evaluates whether individual DINOv2 patch embeddings are discriminative, and optionally
refines the support set through iterative multi-scale quality-weighted pooling.

## File structure

```
local_embedding_patch_quality.py   ← entry point (CLI + experiment runner)
adaptive_patch_pooling/
    __init__.py                    ← package; re-exports public API
    patch_pooling.py               ← core algorithms (entropy, weights, refinement)
    patch_visualisation.py         ← matplotlib figure generation
```

---

## Prerequisites

### Python dependencies

- `tabicl` — `TabICLClassifier`
- `sklearn` — `PCA`, `Ridge`, `StandardScaler`
- `torch`, `PIL`, `matplotlib`, `numpy`, `tqdm`, `joblib`

### Required files

| Path | Description |
|------|-------------|
| `extracted_features/butterfly_train_dinov3_patch_features.pt` | Pre-extracted DINOv2 patch features for the training split — `{"features": float16 [N, P, D], "labels": int64 [N]}` |
| `extracted_features/butterfly_test_dinov3_patch_features.pt` | Same for the test split |
| `extracted_features/butterfly_train_dinov3_features.pt` | CLS token embeddings for the training split — `{"features": float32 [N, D], "labels": int64 [N], ...}`; optional, used for the CLS-token baseline |
| `extracted_features/butterfly_test_dinov3_features.pt` | Same for the test split (optional) |
| `<dataset_path>/Training_set.csv` | CSV with columns `filename`, `label` — used to reconstruct image paths and class names |
| `<dataset_path>/train/<filename>` | Raw training images (read at visualisation time only) |

Feature files are produced by `local_embedding_experiments.py extract`. Default values:
- `FEATURES_DIR = extracted_features/`
- `DATASET_PATH = /project/aip-rahulgk/hermanb/datasets/butterfly-image-classification`

Typical feature dimensions: `N ≈ 4800 (train) / 1200 (test)`, `P = 196` patches per image
(224 px image, 16 px patches), `D = 768` (DINOv2 ViT-B/16).

---

## Execution flow

### 1. Data loading

`ButterflyPatchDataset` is loaded for both splits, yielding `train_patches [N, P, D]` and
`test_patches [N_test, P, D]`. **Patches are always loaded at the original resolution** and
grouped on-the-fly at each refinement stage — there is no global pre-grouping step.

Image paths and class names are reconstructed from the CSV using `_get_image_paths`, which
replicates the exact shuffle/split logic used during feature extraction (controlled by
`--seed`).

If `--n-train` is set, a random subsample of that many training images is drawn (using
`--seed`) and used as the support set. The subsample indices are sorted to preserve the
original dataset order, which keeps alignment with `train_image_paths`.

A fixed random sample of `--n-sample` image indices is drawn once per split and reused
for all visual evaluations, ensuring the same images are compared across all stages.

### 2. Baseline support set

Each training image is mean-pooled across all original patches: `baseline_support [N, D]`.

If `--pca-dim` is set (default 128), PCA is fitted on `baseline_support` and applied
to reduce it to `[N, d]`. The same PCA object is used to transform all query features
in the baseline evaluation.

### 3. Baseline evaluation

Two baselines are computed and printed before any visualisation or refinement:

**CLS-token baseline**: if `butterfly_{split}_dinov3_features.pt` files exist in
`--features-dir`, their pre-extracted CLS token embeddings (`[N, 768]`) are loaded and
used as a direct support set. A separate PCA (same `--pca-dim`) is fitted on the CLS
training embeddings and applied to the test embeddings. TabICL accuracy is reported as
`[cls-token]`.

**Mean-pool baseline** (`_compute_accuracy`): test images are mean-pooled across their
original patches, optionally PCA-transformed, and classified by TabICL fitted on the
mean-pool baseline support. Reported as `[mean-pool]`.

**Visual evaluation** (`_run_visual_eval`, tag `"baseline"`): skipped when `--n-sample 0`
(default) or `--post-refinement-viz` is set. Otherwise a single `TabICLClassifier` is
fitted on the baseline support, then each sampled image's `P` original patches are queried
together via `clf.predict_proba`. This produces `probs [P, n_classes]` — per-patch TabICL
softmax probabilities. A figure is saved per image (see Visualisation section) and a
summary bar chart per split.

### 4. Iterative multi-scale refinement (optional, `--refine`)

When `--refine` is set, the support set is iteratively refined across the resolutions
specified in `--patch-group-sizes` (e.g. `--patch-group-sizes 4 1` for two stages).
Each stage proceeds as follows:

#### Per-stage flow

Let `current_support [N, d]` be the support entering this stage (baseline for stage 0,
the previous stage's refined support thereafter), and `current_pca` the corresponding PCA.

1. **Group original patches** at `group_size`: mean-pool spatially adjacent patches into
   groups of `group_size` (must be a perfect square), yielding `grouped [N, P', D]` where
   `P' = P / group_size`. Both train and test patches are grouped independently at each
   stage from the original (ungrouped) DINO features.

2. **Pre-refinement visual evaluation** (tag `iter_{k}_g{group_size}`): skipped when
   `--n-sample 0` or `--post-refinement-viz` is set. Otherwise fits a classifier on
   `current_support`, queries each sampled image's `P'` grouped patches, and saves
   heatmaps. Because this uses `current_support` (before refinement), the visualised
   scores match exactly what will drive the pooling weights for this stage.

3. **Refine the support** (`refine_dataset_features`):
   a. Fit a `TabICLClassifier` on `current_support` (the scorer, held fixed for the pass).
   b. For each batch of images, transform grouped patches through `current_pca` and call
      `_get_patch_distributions` to get `dist [B×P', n_classes]`.
   c. Compute pooling weights via `compute_patch_pooling_weights`.
   d. Apply weights to the **raw** (pre-PCA) grouped features → `repooled_raw [N, D]`.
   e. Re-fit PCA on `repooled_raw` → `new_pca`; project → `new_support [N, d]`.
   f. Fit a `Ridge` model on `(grouped_patch_features, quality_logits)` and use it to
      redo the repooling (Ridge-predicted weights replace TabICL weights). The Ridge model
      is saved as `ridge_quality_model_iter_{k}_g{group_size}.joblib`. AoE class exclusion
      / entropy handling is applied here (see below).

4. **Post-refinement visual evaluation** (tag `iter_{k}_g{group_size}_post`): produced
   only when `--post-refinement-viz` and `--n-sample > 0`. Shows Ridge
   pooling weight panels alongside the softmax-based panels, using the classifier and
   Ridge model just trained in step 3.

5. **Test query pooling**: pool test images using the **same scorer** (the classifier
   fitted on `current_support` in step 3a) and the same pooling method. When Ridge is
   available it takes precedence. Pooled test features are projected through `new_pca`.

6. **Evaluate accuracy** (`_compute_accuracy_from_features`): classify the pooled test
   features against `new_support`.

7. **Advance**: set `current_support = new_support`, `current_pca = new_pca`.

After all stages a comparison table is printed to stdout.

#### Why raw-feature pooling?

Pooling in PCA space would distort distances because PCA whitens non-uniformly. Pooling
in the original DINO space and then re-fitting PCA preserves the geometric structure of
the learned features.

#### Why use the input support for both scoring and visualisation?

Using the same classifier (fitted on the support *entering* this stage) for both the
training repooling and the test query pooling ensures that the two representations are
constructed identically. The visualisations reflect this scorer, making them directly
interpretable as the quality signal that drove each stage's refinement.

---

## Pooling weight algorithm (`compute_patch_pooling_weights`)

Input: `dist [P, n_classes]`, `true_label`, `temperature`, `weight_method`, `class_prior`.

The distribution `dist` comes from `predict_proba` (TabICL softmax) — `[P, n_classes]` with rows summing to 1.

```
# correct_class_prob method (default)
p_i  = dist[i, true_label].clip(1e-7, 1-1e-7)   # true-class probability per patch
l_i  = ln(p_i)                                   # log-probability score
l̃_i = l_i / temperature                          # temperature scaling
w_i  = softmax(l̃_i)                              # weights summing to 1

# entropy method
H_i    = -Σ_c dist[i,c] log dist[i,c]            # Shannon entropy
q_i    = 1 - H_i / ln(C)                         # normalised quality score in [0, 1]
l_i    = ln(q_i)                                  # log-score (logit), in (-∞, 0]
l̃_i   = l_i / temperature                        # temperature scaling
w_i    = softmax(l̃_i)                            # weights summing to 1

# kl_div method  (requires class_prior [n_classes] — empirical class frequencies)
KL_i   = Σ_c dist[i,c] · ln(dist[i,c] / prior_c) # KL divergence from prediction to prior
max_KL = -ln(min_c prior_c)                        # maximum achievable KL (point mass on rarest class)
q_i    = (KL_i / max_KL).clip(1e-7, 1)            # normalised quality score in (0, 1]
l_i    = ln(q_i)                                   # log-score (logit), in (-∞, 0]
l̃_i   = l_i / temperature                         # temperature scaling
w_i    = softmax(l̃_i)                             # weights summing to 1
```

**Temperature behaviour:**
- `T → ∞`: logits collapse to zero → uniform weights → equivalent to mean pooling
- `T = 1`: weights proportional to true-class probability (or normalised quality score)
- `T → 0`: all weight on the single most-confident (or lowest-entropy / highest-KL) patch

The `entropy` and `kl_div` methods are label-agnostic (do not use `true_label`), making
them applicable when labels are unavailable — notably for the absence-of-evidence class
when `--aoe-handling entropy` is set.

**`kl_div` vs `entropy` on imbalanced datasets:**
When the prior is uniform (balanced dataset), `KL(Q || P_uniform) = ln(C) - H(Q)`, so
`kl_div` and `entropy` are monotonically equivalent and produce the same weights.  On
imbalanced datasets they differ: a patch that confidently predicts a rare class receives a
higher KL score than one equally confident about a common class, rewarding discrimination
against the base rates rather than mere certainty.

---

## Absence-of-evidence (AoE) class

Some datasets include a class that represents the *absence* of a finding (e.g. "no
pathology"). This creates a conceptual mismatch when pooling weights are derived from
true-class confidence: patches from such a class may look similar to positive-class
patches, so high confidence in the AoE label is a poor quality signal.

Pass `--aoe-class <index-or-name>` to flag this class. Its effect is controlled by
`--aoe-handling`:

### `filter` (default)

AoE-class images are excluded from the TabICL forward pass and Ridge fitting entirely.
The Ridge model sees only non-AoE patches, preventing the AoE label from polluting
the quality-logit training targets.

AoE-class images are **still included** in the support (mean-pooled or Ridge-pooled) and
Ridge pooling is applied to them at inference time just like any other class.

### `entropy`

AoE-class images are included in both the TabICL forward pass and Ridge fitting, but
their per-patch quality logits are computed with the `entropy` method regardless of
`--weight-method`. This lets the Ridge model learn AoE-patch quality from an
unsupervised signal, without relying on true-class confidence.

Non-AoE images continue to use the method specified by `--weight-method`.

In both modes, sampling of patch-group rows (`--use-random-subsampling`) excludes
AoE-class images to keep the quality-target distribution clean.

---

## Visualisation

Each image produces a figure with up to 6 panels:

| Panel | Content |
|-------|---------|
| 1 | Original image |
| 2 | P(true class) per patch — RdYlGn heatmap overlaid on image |
| 3 | `correct_class_prob` pooling weights |
| 4 | `entropy` pooling weights |
| 5 | `kl_div` pooling weights (uses the empirical class prior from training labels) |
| 6 | Ridge pooling weights — only shown in post-refinement figures (see below) |

The panel corresponding to the active `--weight-method` is marked with ★ in its title.

Pre-refinement figures show panels 1–5. Post-refinement figures include panel 6 with
Ridge-predicted quality weights. Post-refinement figures are produced in two situations:

- **`--post-refinement-viz`**: one post-refinement figure is saved per stage (inside
  `iter_{k}_g{group_size}_post/`), using the classifier and Ridge model from that stage.
  Pre-refinement figures for all stages are skipped.
- **Default (no `--post-refinement-viz`)**: pre-refinement figures are saved per stage as
  usual; additionally, a single post-refinement figure is saved after the *last* stage
  completes (tag `iter_{last}_g{last_group_size}_post`), giving Ridge-weight heatmaps
  without forgoing the per-stage pre-refinement views.

At each refinement stage the figures show patches grouped at that stage's `group_size`,
with the classifier fitted on the **input** support (before refinement), so the heatmaps
reflect the exact quality signal that drove the pooling weights.

### Output directory layout

```
<output_dir>/
  results.json                                   # always written (see Results section below)
  baseline/                                      # only when --n-sample > 0 and not --post-refinement-viz
    train/
      patch_quality_00_img42_Monarch.png
      patch_quality_01_img7_Swallowtail.png
      ...
      summary.png
    test/
      ...
  iter_0_g4/          # stage 0, group_size=4  (only when --refine is set and --n-sample > 0 and not --post-refinement-viz)
    train/
      ...
    test/
      ...
  iter_0_g4_post/     # only when --post-refinement-viz (per-stage post figures)
    train/
      ...
    test/
      ...
  iter_1_g1/
    ...
  iter_1_g1_post/     # always produced after last stage (Ridge weights); also per-stage when --post-refinement-viz
    ...
  ridge_quality_model_iter_0_g4.joblib   # Ridge model saved per stage when --refine is set
  ridge_quality_model_iter_1_g1.joblib
```

### Results file (`results.json`)

Written to `<output_dir>/results.json` at the end of every run (with or without
`--refine`, with or without `--n-sample`). Structure:

```json
{
  "run_timestamp": "2026-04-04T12:00:00+00:00",
  "total_time_s": 142.3,
  "args": { "weight_method": "correct_class_prob", "patch_group_sizes": [4, 1], "seed": 42, ... },
  "dataset": {
    "n_train": 5200, "n_test": 1299,
    "n_patches": 256, "embed_dim": 768, "n_classes": 75, "pca_dim": 128
  },
  "baselines": {
    "cls_token": 0.812345,   // null if CLS feature files not found
    "mean_pool": 0.793210
  },
  "stages": [
    {
      "tag": "baseline",
      "test_accuracy": 0.793210, "delta_acc": 0.0,
      "mean_prob_train": NaN, "mean_prob_test": NaN,   // NaN when --n-sample 0
      "refine_time_s": 0.0, "eval_time_s": 0.0
    },
    {
      "tag": "iter_0_g4",
      "test_accuracy": 0.821000, "delta_acc": 0.02779,
      "mean_prob_train": 0.412, "mean_prob_test": 0.398,
      "refine_time_s": 38.4, "eval_time_s": 12.1
    }
  ]
}
```

`refine_time_s` covers the full `refine_dataset_features` call (TabICL scoring + Ridge
fitting + support repooling + PCA refit). `eval_time_s` covers the final
`_compute_accuracy_from_features` call on the test set for that stage.

---

## CLI reference

```
python local_embedding_patch_quality.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--features-dir` | `extracted_features/` | Directory with `.pt` feature files |
| `--dataset-path` | `/project/.../butterfly-image-classification` | Dataset root (for CSV and images) |
| `--n-sample` | `0` | Number of images to visualise per split; `0` skips all visualisation |
| `--n-train` | `None` | Limit the support set to this many training images (random subsample); uses `--seed` |
| `--n-estimators` | `1` | TabICL ensemble size |
| `--pca-dim` | `128` | PCA output dimension; applies to both support and queries |
| `--no-pca` | `False` | Disable PCA; use full 768-D DINO features |
| `--seed` | `42` | Controls train/test split, image sampling, PCA, and TabICL |
| `--output-dir` | `patch_quality_results/` | Root output directory |
| `--patch-size` | `16` | Base patch size in pixels (must match extraction) |
| `--patch-group-sizes` | `1` | Ordered list of group sizes for iterative refinement. Each must be a perfect square (1, 4, 9, …). A single value runs one stage at that group size; multiple values (e.g. `--patch-group-sizes 4 1`) chain stages from coarse to fine. `1` = individual patches. |
| `--refine` | `False` | Run iterative multi-scale refinement |
| `--temperature` | `1.0` | Softmax temperature for pooling weights. Pass one value to use for all stages, or one per stage to vary across stages. Large → uniform/mean pooling; small → peaked on best patch. |
| `--batch-size` | `1000` | Images per TabICL call during refinement |
| `--weight-method` | `correct_class_prob` | How to derive patch pooling weights: `correct_class_prob` (log true-class probability), `entropy` (normalised entropy, label-agnostic), or `kl_div` (KL divergence from prediction to empirical class prior, normalised by maximum achievable KL; label-agnostic and sensitive to class imbalance). All methods use temperature-scaled softmax. |
| `--mix-lambda` | `1.0` | Interpolation between refined and mean-pooled embeddings (1.0 → fully refined; requires `--refine`) |
| `--ridge-alpha` | `1.0` | Ridge regularisation strength. Pass one value for all stages or one per stage. |
| `--normalize-features` | `False` | Fit a `StandardScaler` on training patches before Ridge fitting; scaler is applied at predict time too |
| `--aoe-class` | `None` | Absence-of-evidence class: patches receive special handling during Ridge fitting. May be given as a class index (integer) or class name string. The class is always included in the support and Ridge pooling is applied to it as normal. |
| `--aoe-handling` | `filter` | How to handle the AoE class during Ridge fitting (requires `--aoe-class`). `filter`: exclude AoE patches from TabICL scoring and Ridge fitting. `entropy`: include AoE patches but score them with `entropy` instead of `--weight-method`. |
| `--post-refinement-viz` | `False` | Skip pre-refinement visualisations; only produce post-refinement figures (with Ridge pooling weight panels). Requires `--refine`. |

### Example: two-stage iterative refinement

```bash
python local_embedding_patch_quality.py \
    --refine \
    --patch-group-sizes 4 1 \
    --temperature 2.0 1.0 \
    --weight-method correct_class_prob \
    --ridge-alpha 10.0 1.0 \
    --n-sample 8 \
    --post-refinement-viz \
    --output-dir results/iterative
```

This runs:
1. Baseline: mean-pooled support, original patches
2. Stage 0 (`iter_0_g4`): score 4-patch groups with baseline support (T=2.0, α=10.0)
3. Stage 1 (`iter_1_g1`): score individual patches with stage-0 support (T=1.0, α=1.0)

Post-refinement figures with Ridge weight panels are saved to `iter_0_g4_post/` and
`iter_1_g1_post/` only (pre-refinement heatmaps are skipped).

### Example: AoE class handling

```bash
python local_embedding_patch_quality.py \
    --refine \
    --aoe-class "No Finding" \
    --aoe-handling entropy \
    --weight-method correct_class_prob \
    --post-refinement-viz \
    --n-sample 8 \
    --output-dir results/aoe_entropy
```

Patches from the `"No Finding"` class are scored with the label-agnostic `entropy`
method; all other classes use `correct_class_prob`. The Ridge model is fitted on all
patches together.

---

## Key functions

**`adaptive_patch_pooling/patch_pooling.py`**

| Function | Purpose |
|----------|---------|
| `compute_patch_entropy(patch_probs)` | Per-patch Shannon entropy in nats `[P]` |
| `compute_patch_pooling_weights(dist, true_label, temperature, weight_method, class_prior)` | `correct_class_prob` / `entropy` / `kl_div` weighting → pooling weights `[P]` summing to 1; `class_prior` required for `kl_div` |
| `compute_patch_quality_logits(dist, true_label, temperature, weight_method, class_prior)` | Pre-normalisation quality logits `[P]`; used as Ridge regression targets; `class_prior` required for `kl_div` |
| `group_patches(patches, patch_group_size)` | Mean-pool spatially adjacent patches into groups; `[N, P, D]` → `[N, P', D]` |
| `_ridge_pool_weights(patches, ridge_model, feature_scaler)` | Per-patch softmax weights from a fitted Ridge model `[N, P]` |
| `_mix_and_project(repooled_raw, raw_patches, mix_lambda, pca, seed)` | Mix-lambda blend with mean-pool and re-fit PCA |
| `refine_dataset_features(train_patches, train_labels, support, pca, ..., aoe_mask, aoe_handling)` | Full refinement pass for one stage; returns `(refined [N,d], new_pca, weights_all [N,P'], ridge_model, feature_scaler, clf)` |

**`adaptive_patch_pooling/patch_visualisation.py`**

| Function | Purpose |
|----------|---------|
| `visualise_image(image, patch_probs, true_label, ..., ridge_pred_logits, class_prior, weight_method)` | Build figure with overlay panels (P(true), ccp/entropy/kl_div weights); active `weight_method` panel is marked ★; adds Ridge weight panel when `ridge_pred_logits` is provided |
| `summary_figure(results)` | Bar chart of per-image mean correct-class probability |

**`local_embedding_patch_quality.py`**

| Function | Purpose |
|----------|---------|
| `ButterflyPatchDataset` | Loads pre-extracted DINOv3 patch features from `.pt` files |
| `_get_image_paths(dataset_path, split, seed)` | Reconstruct ordered image paths + integer labels from CSV, replicating the extraction-time shuffle |
| `_compute_accuracy(support, labels, test_patches, test_labels, pca, ...)` | Test-set accuracy using mean-pooled test queries (mean-pool baseline) |
| `_compute_accuracy_from_features(support, labels, query_features, query_labels, ...)` | Test-set accuracy from pre-projected query features (CLS baseline + iterative stages) |
| `_run_visual_eval(tag, support, train_labels, split_configs, ..., ridge_model, feature_scaler, class_prior)` | Visual evaluation loop for one stage; saves per-image heatmaps and summary chart; passes Ridge model and class prior to `visualise_image`; no-op when `n_sample=0` |
| `_save_results(output_dir, run_ts, cli_args, total_time_s, ...)` | Serialise experiment record (args, dataset info, baselines, per-stage accuracy + timing) to `results.json` |
| `run_patch_quality_eval(...)` | Top-level entry point; orchestrates baseline + iterative refinement loop |
