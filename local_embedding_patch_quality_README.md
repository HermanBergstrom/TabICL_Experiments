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

**Accuracy** (`_compute_accuracy`): test images are mean-pooled across their original
patches, optionally PCA-transformed, and classified by TabICL fitted on the baseline
support. Mean accuracy over the test set is reported.

**Visual evaluation** (`_run_visual_eval`, tag `"baseline"`): a single `TabICLClassifier`
is fitted on the baseline support, then each sampled image's `P` original patches are
queried together via `clf.predict_proba`. This produces `probs [P, n_classes]` —
per-patch TabICL softmax probabilities. A figure is saved per image (see Visualisation
section) and a summary bar chart per split.

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

2. **Visual evaluation** (tag `iter_{k}_g{group_size}`): fit a classifier on
   `current_support`, query each sampled image's `P'` grouped patches, and save
   heatmaps. Because this uses `current_support` (before refinement), the visualised
   scores match exactly what will drive the pooling weights for this stage.

3. **Refine the support** (`refine_dataset_features`):
   a. Fit a `TabICLClassifier` on `current_support` (the scorer, held fixed for the pass).
   b. For each batch of images, transform grouped patches through `current_pca` and call
      `_get_patch_distributions` to get `dist [B×P', n_classes]`.
   c. Compute pooling weights via `compute_patch_pooling_weights`.
   d. Apply weights to the **raw** (pre-PCA) grouped features → `repooled_raw [N, D]`.
   e. Re-fit PCA on `repooled_raw` → `new_pca`; project → `new_support [N, d]`.
   f. If `--fit-ridge`: fit a `Ridge` model on `(grouped_patch_features, quality_logits)`
      and use it to redo the repooling (Ridge-predicted weights replace TabICL weights).
      The Ridge model is saved as `ridge_quality_model_iter_{k}_g{group_size}.joblib`.

4. **Test query pooling**: pool test images using the **same scorer** (the classifier
   fitted on `current_support` in step 3a) and the same pooling method. When Ridge is
   available it takes precedence. Pooled test features are projected through `new_pca`.

5. **Evaluate accuracy** (`_compute_accuracy_from_features`): classify the pooled test
   features against `new_support`.

6. **Advance**: set `current_support = new_support`, `current_pca = new_pca`.

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

Input: `dist [P, n_classes]`, `true_label`, `temperature`, `weight_method`, `gamma`.

The distribution `dist` comes from `predict_proba` (TabICL softmax) — `[P, n_classes]` with rows summing to 1.

```
# logit method (default)
p_i  = dist[i, true_label].clip(1e-7, 1-1e-7)   # true-class probability per patch
l_i  = ln(p_i)                                   # log-probability score
l̃_i = l_i / temperature                          # temperature scaling
w_i  = softmax(l̃_i)                              # weights summing to 1

# prob method
w_i  = p_i ^ gamma / Σ p_j ^ gamma

# entropy method
H_i  = -Σ_c dist[i,c] log dist[i,c]              # Shannon entropy
s_i  = ln(C) - H_i                               # quality score (low entropy → high)
w_i  = s_i ^ gamma / Σ s_j ^ gamma

# entropy_logit method
H_i    = -Σ_c dist[i,c] log dist[i,c]            # Shannon entropy
q_i    = 1 - H_i / ln(C)                         # normalised quality score in [0, 1]
l_i    = ln(q_i)                                  # log-score (logit), in (-∞, 0]
l̃_i  = l_i / temperature                         # temperature scaling
w_i    = softmax(l̃_i)                            # weights summing to 1

# combined method
w_i  = (w_logit_i + w_entropy_logit_i) / 2       # arithmetic mean, then renormalise
```

**Temperature behaviour (logit method):**
- `T → ∞`: log-probabilities collapse to zero → uniform weights → equivalent to mean pooling
- `T = 1`: weights proportional to true-class probability
- `T → 0`: all weight on the single most-confident patch (winner-take-all)

The `entropy_logit` method is label-agnostic (does not use `true_label`), making it
applicable even when labels are unavailable. The `combined` method blends logit and
entropy_logit weights via arithmetic averaging.

---

## Visualisation

Each image produces a figure with 6 panels:

| Panel | Content |
|-------|---------|
| 1 | Original image |
| 2 | P(true class) per patch — RdYlGn heatmap overlaid on image |
| 3 | Logit-based pooling weights |
| 4 | Prediction entropy per patch (normalised by `log(n_classes)`) — RdYlGn_r heatmap |
| 5 | Entropy-based pooling weights (`entropy` or `entropy_logit` depending on `--weight-method`) |
| 6 | Combined pooling weights (arithmetic mean of logit and entropy_logit) |

At each refinement stage the figures show patches grouped at that stage's `group_size`,
with the classifier fitted on the **input** support (before refinement), so the heatmaps
reflect the exact quality signal that drove the pooling weights.

### Output directory layout

```
<output_dir>/
  baseline/
    train/
      patch_quality_00_img42_Monarch.png
      patch_quality_01_img7_Swallowtail.png
      ...
      summary.png
    test/
      ...
  iter_0_g4/          # stage 0, group_size=4  (only when --refine is set)
    train/
      ...
    test/
      ...
  iter_1_g1/          # stage 1, group_size=1
    ...
  ridge_quality_model_iter_0_g4.joblib   # only when --fit-ridge
  ridge_quality_model_iter_1_g1.joblib
```

---

## CLI reference

```
python local_embedding_patch_quality.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--features-dir` | `extracted_features/` | Directory with `.pt` feature files |
| `--dataset-path` | `/project/.../butterfly-image-classification` | Dataset root (for CSV and images) |
| `--n-sample` | `8` | Number of images to visualise per split |
| `--n-train` | `None` | Limit the support set to this many training images (random subsample); uses `--seed` |
| `--n-estimators` | `1` | TabICL ensemble size |
| `--pca-dim` | `128` | PCA output dimension; applies to both support and queries |
| `--no-pca` | `False` | Disable PCA; use full 768-D DINO features |
| `--seed` | `42` | Controls train/test split, image sampling, PCA, and TabICL |
| `--output-dir` | `patch_quality_results/` | Root output directory |
| `--patch-size` | `16` | Base patch size in pixels (must match extraction) |
| `--patch-group-sizes` | `1` | Ordered list of group sizes for iterative refinement. Each must be a perfect square (1, 4, 9, …). A single value runs one stage at that group size; multiple values (e.g. `--patch-group-sizes 4 1`) chain stages from coarse to fine. `1` = individual patches. |
| `--refine` | `False` | Run iterative multi-scale refinement |
| `--temperature` | `1.0` | Softmax temperature for pooling weights (logit/entropy_logit/combined methods). Pass one value to use for all stages, or one per stage to vary across stages. Large → uniform/mean pooling; small → peaked on best patch. |
| `--batch-size` | `100` | Images per TabICL call during refinement |
| `--weight-method` | `logit` | How to derive pooling weights: `logit`, `prob`, `entropy`, `entropy_logit`, or `combined` |
| `--gamma` | `1.0` | Power exponent for the `prob` and `entropy` weight methods |
| `--mix-lambda` | `1.0` | Interpolation between refined and mean-pooled embeddings (1.0 → fully refined; requires `--refine`) |
| `--fit-ridge` | `False` | At each stage, fit a Ridge model on (grouped-patch-features, quality-logit targets) and use it for pooling instead of TabICL weights. Model saved per stage. |
| `--ridge-alpha` | `1.0` | Ridge regularisation strength. Pass one value for all stages or one per stage. |
| `--normalize-features` | `False` | Fit a `StandardScaler` on training patches before Ridge fitting; scaler is applied at predict time too |

### Example: two-stage iterative refinement

```bash
python local_embedding_patch_quality.py \
    --refine \
    --patch-group-sizes 4 1 \
    --temperature 2.0 1.0 \
    --weight-method logit \
    --fit-ridge \
    --ridge-alpha 10.0 1.0 \
    --n-sample 8 \
    --output-dir results/iterative
```

This runs:
1. Baseline: mean-pooled support, original patches
2. Stage 0 (`iter_0_g4`): score 4-patch groups with baseline support (T=2.0, α=10.0)
3. Stage 1 (`iter_1_g1`): score individual patches with stage-0 support (T=1.0, α=1.0)

---

## Key functions

**`adaptive_patch_pooling/patch_pooling.py`**

| Function | Purpose |
|----------|---------|
| `compute_patch_entropy(patch_probs)` | Per-patch Shannon entropy in nats `[P]` |
| `compute_patch_pooling_weights(dist, true_label, temperature, weight_method, gamma)` | Logit / prob / entropy weighting → pooling weights `[P]` summing to 1 |
| `compute_patch_quality_logits(dist, true_label, temperature, weight_method, gamma)` | Pre-normalisation quality logits `[P]`; used as Ridge regression targets |
| `group_patches(patches, patch_group_size)` | Mean-pool spatially adjacent patches into groups; `[N, P, D]` → `[N, P', D]` |
| `_ridge_pool_weights(patches, ridge_model, feature_scaler)` | Per-patch softmax weights from a fitted Ridge model `[N, P]` |
| `_mix_and_project(repooled_raw, raw_patches, mix_lambda, pca, seed)` | Mix-lambda blend with mean-pool and re-fit PCA |
| `_pool_features_with_clf(grouped_patches, labels, clf, scoring_pca, ...)` | Pool grouped patches using quality weights from a **pre-fitted** classifier; scoring in PCA space, pooling in raw DINO space |
| `refine_dataset_features(train_patches, train_labels, support, pca, ...)` | Full refinement pass for one stage; returns `(refined [N,d], new_pca, weights_all [N,P'], ridge_model, feature_scaler, clf)` |

**`adaptive_patch_pooling/patch_visualisation.py`**

| Function | Purpose |
|----------|---------|
| `visualise_image(image, patch_probs, true_label, ...)` | Build figure with softmax-based overlay panels via the inner `_dist_panels` closure |
| `summary_figure(results)` | Bar chart of per-image mean correct-class probability |

**`local_embedding_patch_quality.py`**

| Function | Purpose |
|----------|---------|
| `ButterflyPatchDataset` | Loads pre-extracted DINOv3 patch features from `.pt` files |
| `_get_image_paths(dataset_path, split, seed)` | Reconstruct ordered image paths + integer labels from CSV, replicating the extraction-time shuffle |
| `_compute_accuracy(support, labels, test_patches, test_labels, pca, ...)` | Test-set accuracy using mean-pooled test queries (baseline) |
| `_compute_accuracy_from_features(support, labels, query_features, query_labels, ...)` | Test-set accuracy from pre-projected query features (iterative stages) |
| `_run_visual_eval(tag, support, train_labels, split_configs, ...)` | Visual evaluation loop for one stage; saves per-image heatmaps and summary chart |
| `run_patch_quality_eval(...)` | Top-level entry point; orchestrates baseline + iterative refinement loop |
