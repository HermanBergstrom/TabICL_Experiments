# local_embedding_patch_quality.py

Evaluates whether individual DINOv2 patch embeddings are discriminative, and optionally
refines the support set by replacing mean-pooled image features with quality-weighted
patch pooling.

---

## Prerequisites

### Python dependencies

- `tabicl` — `TabICLClassifier`
- `sklearn` — `PCA`
- `torch`, `PIL`, `matplotlib`, `numpy`, `tqdm`

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
`test_patches [N_test, P, D]`. Image paths and class names are reconstructed from the CSV
using `_get_image_paths`, which replicates the exact shuffle/split logic used during
feature extraction (controlled by `--seed`).

If `--n-train` is set, a random subsample of that many training images is drawn (using
`--seed`) and used as the support set. The subsample indices are sorted to preserve the
original dataset order, which keeps alignment with `train_image_paths`.

A fixed random sample of `--n-sample` image indices is drawn once per split and reused
for both baseline and refined visual evaluations, ensuring the same images are always
compared.

### 2. Baseline support set

Each training image is mean-pooled across patches: `baseline_support [N, D]`.

If `--pca-dim` is set (default 128), PCA is fitted on `baseline_support` and applied
to reduce it to `[N, d]`. The same PCA object is used to transform all query features.

### 3. Baseline evaluation

**Accuracy** (`_compute_accuracy`): test images are mean-pooled across patches,
optionally PCA-transformed with the same PCA, and classified by TabICL fitted on the
baseline support. Mean accuracy over the test set is reported.

**Visual evaluation** (`_run_visual_eval`, tag `"baseline"`): a single `TabICLClassifier`
is fitted once on the support set, then each sampled image's `P` patches are queried
together via `clf.predict_proba`. This produces:
- `probs [P, n_classes]` — per-patch TabICL softmax probabilities

If `--visualize-attention` is set, a second forward pass (`clf.predict(..., return_attn=True)`)
is made per image to obtain the avg-heads attention-based class distribution
`attn_avg_scores [P, n_classes]` via `_get_patch_distributions`.

A figure is saved per image (see Visualisation section) and a summary bar chart per split.

### 4. Dataset refinement (optional, `--refine`)

`refine_dataset_features` replaces the mean-pooled baseline support with a
quality-weighted repooling of the raw DINO features.

**Algorithm** (one pass over training images, in batches of `--batch-size`):

1. Fit a single `TabICLClassifier` on `baseline_support` (the support is held fixed for
   the entire refinement pass — no chicken-and-egg issue).
2. For each batch of `B` images, flatten their patches to `[B×P, D]`, apply the
   baseline PCA, and call `_get_patch_distributions` once to get `dist [B×P, n_classes]`.
   The distribution source is controlled by `--distribution-source`:
   - `"softmax"` (default): calls `clf.predict_proba` — standard TabICL output.
   - `"attention"`: calls `clf.predict(..., return_attn=True)` and aggregates avg-heads
     attention into per-class scores via `_attn_class_scores`.
3. Reshape to `[B, P, n_classes]` and compute pooling weights per image via
   `compute_patch_pooling_weights` (see algorithm below).
4. Apply weights to the **raw DINO** features (not PCA space):
   `repooled_raw[i] = Σ_j w_j · patch_j`  →  `[N, D]`
5. After all batches, re-fit PCA on `repooled_raw` (same `n_components` as baseline PCA)
   and return `(refined [N, d], new_pca, weights_all [N, P])`.

**Why raw-feature pooling?** Pooling in PCA space would distort distances because PCA
whitens the feature space non-uniformly. Pooling in the original DINO space and then
re-fitting PCA preserves the geometric structure of the learned features.

### 5. Refined evaluation

The same accuracy and visual evaluation steps as baseline are repeated with
`refined_support` and `refined_pca`. Because `split_configs` is shared and the sample
indices are pre-drawn, the visualisation covers identical images.

### 6. Comparison summary

A table is printed to stdout comparing test accuracy and mean patch prediction quality
(mean P(true class) across patches) for baseline vs. refined, with deltas.

---

## Pooling weight algorithm (`compute_patch_pooling_weights`)

Input: `dist [P, n_classes]`, `true_label`, `temperature`, `weight_method`, `gamma`.

The distribution `dist` can come from either `predict_proba` (softmax) or from
avg-heads attention scores — both are `[P, n_classes]` with rows summing to 1.

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

**Temperature behaviour (entropy_logit method):**
- `T → ∞`: log-scores collapse to zero → uniform weights → equivalent to mean pooling
- `T = 1`: weights proportional to the log-normalised quality score
- `T → 0`: all weight on the single lowest-entropy patch

The `entropy_logit` method is label-agnostic (does not use `true_label`), making it
applicable even when labels are unavailable. It captures overall prediction confidence
rather than confidence in a specific class. The `combined` method blends this with the
class-discriminative `logit` weights via arithmetic averaging.

---

## Visualisation

Each image produces a figure with **1 row** by default, expanding to **2 rows** when
`--visualize-attention` is set. Each row corresponds to one distribution source and
contains 6 panels:

| Panel | Content |
|-------|---------|
| 1 | Original image (labelled with distribution source) |
| 2 | P(true class) per patch — RdYlGn heatmap overlaid on image |
| 3 | Logit-based pooling weights — derived from the distribution via the logit/softmax algorithm |
| 4 | Prediction entropy per patch (normalised by `log(n_classes)`) — RdYlGn_r heatmap (red = high entropy / uncertain) |
| 5 | Entropy-based pooling weights — `entropy` weights normally; `entropy_logit` weights when `--weight-method` is `entropy_logit` or `combined` |
| 6 | Combined pooling weights — arithmetic mean of logit and entropy_logit weights (always shown) |

When `--visualize-attention` is set:
- **Row 1**: panels computed from the softmax (`predict_proba`) distribution
- **Row 2**: same 6 panels computed from the avg-heads attention class distribution

This layout allows direct visual comparison of the two distribution sources on identical patches.

Pooling weight heatmaps are shown at their absolute scale (min/max from the actual weight
values). Entropy is normalised to `[0, 1]` by dividing by `log(n_classes)`.

Summary figures (one per split per tag) show a bar chart of per-image mean P(true class),
derived from the softmax distribution.

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
  refined/          # only when --refine is set
    train/
      ...
    test/
      ...
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
| `--patch-size` | `16` | Patch size in pixels (must match extraction) |
| `--refine` | `False` | Run dataset refinement and compare before/after |
| `--temperature` | `1.0` | Temperature for patch pooling weights (applies to `logit`, `entropy_logit`, and `combined`; large → uniform/mean pooling, small → peaked on best patch) |
| `--batch-size` | `100` | Images per TabICL call during refinement |
| `--weight-method` | `logit` | How to derive pooling weights: `logit`, `prob`, `entropy`, `entropy_logit`, or `combined` (refinement only; visual eval always shows logit, entropy/entropy-logit, and combined panels) |
| `--gamma` | `1.0` | Power exponent for the `prob` and `entropy` weight methods |
| `--mix-lambda` | `1.0` | Interpolation between refined and mean-pooled embeddings (1.0 → fully refined; requires `--refine`) |
| `--distribution-source` | `softmax` | Distribution used to derive pooling weights: `softmax` (TabICL `predict_proba`) or `attention` (avg-heads attention class scores) |
| `--visualize-attention` | `False` | Add a second row of panels to each figure using the attention-based class distribution |

---

## Key functions

| Function | Purpose |
|----------|---------|
| `_get_image_paths(dataset_path, split, seed)` | Reconstruct ordered image paths + integer labels from CSV, replicating the extraction-time shuffle |
| `_attn_class_scores(attn_weights, labels, n_classes, n_train, head, block)` | Aggregate attention from one block/head into per-patch, per-class scores `[P, n_classes]` |
| `_get_patch_distributions(clf, query_features, train_labels, n_classes, n_train, distribution_source)` | Return `[Q, n_classes]` from either `predict_proba` or avg-heads attention; single dispatch point used by both refinement and visual eval |
| `compute_patch_entropy(patch_probs)` | Per-patch Shannon entropy in nats `[P]` |
| `compute_patch_pooling_weights(dist, true_label, temperature, weight_method, gamma)` | Logit / prob / entropy weighting → pooling weights `[P]` summing to 1; works on any `[P, n_classes]` distribution |
| `refine_dataset_features(train_patches, train_labels, support, pca, ...)` | Full refinement pass; returns `(refined [N,d], new_pca, weights_all [N,P])` |
| `_compute_accuracy(support, labels, test_patches, test_labels, pca)` | Test-set accuracy using mean-pooled test queries |
| `_run_visual_eval(tag, support, train_labels, split_configs, ...)` | Visual evaluation loop for one support set variant; fits classifier once, saves figures |
| `_visualise_image(image, patch_probs, true_label, ..., attn_avg_scores)` | Build a 1- or 2-row figure; each row is generated by the inner `_dist_panels(dist, label)` closure |
| `run_patch_quality_eval(...)` | Top-level entry point; orchestrates everything |
