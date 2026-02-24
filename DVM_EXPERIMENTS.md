# DVM Experiments — Script Guide

`dvm_experiments.py` benchmarks classification models on the **DVM car dataset**
using tabular features, DINOv2 image embeddings, or a concatenation of both.  It
supports three broad modes of operation:

| Mode | What it does |
|---|---|
| **Standard** | Runs TabICL, Decision Tree, Random Forest, and XGBoost on the chosen feature set. |
| **Feature suite** (`--feature-suite`) | Sweeps all 9 combinations of feature mode × image reducer in a single run. |
| **Probing** (`--tabicl-features-dir` or `--methods linear_probe mlp`) | Replaces raw tabular features with pre-extracted TabICL representations and trains lightweight PyTorch heads. |

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Code Organisation](#code-organisation)
3. [CLI Reference](#cli-reference)
4. [Feature Modes & Image Reducers](#feature-modes--image-reducers)
5. [Feature Suite](#feature-suite)
6. [Probing Mode](#probing-mode)
7. [Output Format](#output-format)
8. [Companion Scripts](#companion-scripts)

---

## Quick Start

```bash
# Standard run – tabular only, all four models, 20 k train rows
python dvm_experiments.py

# Image-only features with PCA-128 reduction, XGBoost only
python dvm_experiments.py \
    --feature-mode image --image-reducer pca --image-reducer-dim 128 \
    --methods xgboost

# Full 9-config feature comparison suite
python dvm_experiments.py --feature-suite --suite-reducer-dim 64

# Probing mode with pre-extracted TabICL features
python dvm_experiments.py --methods linear_probe mlp

# Probing + image concat suite
python dvm_experiments.py \
    --methods linear_probe mlp --feature-suite \
    --tabicl-features-dir /path/to/tabiclv2_features
```

---

## Code Organisation

The script is split into **10 numbered sections**, each responsible for one
concern.  Helper functions are prefixed with `_` and documented in-line.

| # | Section | Key functions |
|---|---------|---------------|
| 1 | **Data loading** | `_import_load_dvm_dataset`, `_extract_modalities_from_dataset`, `_load_data` |
| 2 | **Sampling** | `_stratified_indices`, `_maybe_index` |
| 3 | **TabICL representations** | `_load_representations`, `_load_all_representations` |
| 4 | **Image reduction** | `_fit_image_reducer`, `_apply_reducer` |
| 5 | **Feature assembly** | `_build_features` |
| 6 | **Experiment configs** | `_resolve_feature_experiments` |
| 7 | **Model construction** | `_build_standard_models` |
| 8 | **Train & eval helpers** | `_resolve_device`, `_fit_sklearn_model`, `_evaluate_clf`, `_run_probing_head`, `_print_model_results` |
| 9 | **Main loop** | `run_experiment` |
| 10 | **CLI** | `parse_args`, `main` |

### Data flow

```
CLI args
  │
  ▼
_load_data()          ──  load DVM splits as numpy arrays
  │
  ├─ (optional) _load_all_representations()   ──  pre-extracted TabICL features
  │
  ▼
_resolve_feature_experiments()   ──  list of {mode, reducer, dim} dicts
  │
  ▼
for each experiment config:
  │
  ├── for each train size:
  │     │
  │     ├── _stratified_indices()    ──  subsample train rows
  │     ├── _build_features()        ──  assemble X_train / X_val / X_test
  │     │     └── _fit_image_reducer() + _apply_reducer()
  │     │
  │     └── for each model:
  │           ├── standard  →  _fit_sklearn_model() + _evaluate_clf()
  │           └── probing   →  _run_probing_head()
  │
  ▼
results JSON  →  saved to --results-path
```

---

## CLI Reference

### Paths

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `…/DVM_Dataset` | Root of the DVM dataset |
| `--dvm-module-path` | `…/dvm_dataset_with_dinov2.py` | Path to the Python module that defines `load_dvm_dataset` |
| `--results-path` | `results/dvm_experiments_results.json` | Where to save the JSON results |
| `--tabicl-features-dir` | *(auto-resolved)* | Directory with `{split}_representations.pt` files.  Setting this enables probing mode. |

### Seed & sizing

| Flag | Default | Description |
|---|---|---|
| `--seed` | `0` | Global random seed |
| `--n-estimators` | `1` | Number of TabICL estimators |
| `--max-train-samples` | `20000` | Cap on training rows (≤ 0 → use all) |
| `--max-eval-samples` | `20000` | Cap on val / test rows (≤ 0 → use all) |
| `--train-sizes` | `None` | Space-separated train sizes to sweep, e.g. `500 2000 10000` |

### Methods

| Flag | Default | Description |
|---|---|---|
| `--methods` | *(all standard or all probing)* | Subset of: `tabicl`, `decision_tree`, `random_forest`, `xgboost`, `linear_probe`, `mlp` |

### Feature configuration

| Flag | Default | Description |
|---|---|---|
| `--feature-mode` | `tabular` | One of `tabular`, `image`, `concat` |
| `--image-reducer` | `none` | One of `none`, `pca`, `ica`, `random_projection` |
| `--image-reducer-dim` | `128` | Target dimensionality for the selected reducer |
| `--feature-suite` | off | Run the full 9-config sweep |
| `--suite-reducer-dim` | `64` | Reducer dimensionality used inside the suite (legacy alias: `--suite-pca-dim`) |

### XGBoost / Device

| Flag | Default | Description |
|---|---|---|
| `--xgb-progress` | off | Print XGBoost training progress |
| `--xgb-verbose-every` | `10` | Print every *N* boosting rounds |
| `--device` | `auto` | One of `auto`, `cpu`, `cuda` |

---

## Feature Modes & Image Reducers

**Feature modes** control *which* columns are fed to the models:

- `tabular` — only the preprocessed tabular columns (or TabICL representations in probing mode).
- `image` — only the DINOv2 image embeddings (768-d by default).
- `concat` — column-wise concatenation of tabular + image.

**Image reducers** optionally compress/transform the 768-d image embeddings
before they are used:

| Reducer | Method |
|---|---|
| `none` | Use raw embeddings |
| `pca` | `sklearn.decomposition.PCA` |
| `ica` | `sklearn.decomposition.FastICA` |
| `random_projection` | `sklearn.random_projection.GaussianRandomProjection` |

PCA and ICA automatically cap `n_components` at `min(n_samples, n_features)` to
avoid numerical issues.

---

## Feature Suite

When `--feature-suite` is passed, the script ignores `--feature-mode` and
`--image-reducer` and instead runs all 9 combinations:

| Label | Mode | Reducer |
|---|---|---|
| `tabular_only` | tabular | none |
| `image_only` | image | none |
| `image_pca{d}` | image | pca |
| `image_ica{d}` | image | ica |
| `image_rp{d}` | image | random_projection |
| `concat` | concat | none |
| `concat_pca{d}` | concat | pca |
| `concat_ica{d}` | concat | ica |
| `concat_rp{d}` | concat | random_projection |

where *d* = `--suite-reducer-dim` (default 64).

Results for every config × train size × model are collected into a single JSON.

---

## Probing Mode

Probing mode is activated when `--tabicl-features-dir` is set (or auto-resolved
from the default when `--methods` contains `linear_probe` or `mlp`).

In this mode:

1. Pre-extracted TabICL representations (`{split}_representations.pt`) are
   loaded and mean-pooled across estimators, yielding one vector per row.
2. These representations **replace** the raw tabular features in every feature
   config.  Image features are still used normally in `image` / `concat` modes.
3. Only `linear_probe` and `mlp` methods are available.  Standard sklearn/TabICL
   models are skipped.

### Probing heads

| Head | Architecture | LR | Implementation |
|---|---|---|---|
| `linear_probe` | Single `nn.Linear` | 1e-3 | `baseline_heads.LinearProbe` |
| `mlp` | Linear → ReLU → Dropout → Linear | 1e-4 | `baseline_heads.MLPHead` |

Both are trained with Adam, CrossEntropyLoss, and early stopping (patience 10),
using `train_head()` / `predict_head()` from `baseline_heads.py`.

### Generating representations

Use `extract_tabicl_features.py` to produce the `.pt` files:

```bash
python extract_tabicl_features.py \
    --output-dir /path/to/features \
    --n-estimators 16 \
    --max-train-samples 20000
```

---

## Output Format

Results are saved as JSON with this structure:

```jsonc
{
  "metadata": {
    "data_root": "...",
    "seed": 0,
    "device": "auto",
    "feature_suite": true,
    "feature_experiments": [ /* list of config dicts */ ],
    "train_sizes": [500, 2000, 10000],
    "methods": ["tabicl", "decision_tree", "random_forest", "xgboost"],
    "max_eval_samples": 20000
  },
  "experiments": [
    {
      "label": "tabular_only",
      "feature_mode": "tabular",
      "image_reducer": "none",
      "image_reducer_dim": 64,
      "runs": [
        {
          "train_size_requested": 500,
          "train_size_actual": 500,
          "models": {
            "tabicl": {
              "fit": { "fit_seconds": 12.3 },
              "val": { "accuracy": 0.81, "f1_macro": 0.79, "eval_seconds": 0.5 },
              "test": { "accuracy": 0.80, "f1_macro": 0.78, "eval_seconds": 0.5 }
            }
            // ...other models
          }
        }
        // ...other train sizes
      ]
    }
    // ...other feature experiments
  ],
  // When only one experiment, a top-level "runs" shortcut is added:
  "runs": [ /* same as experiments[0].runs */ ]
}
```

---

## Companion Scripts

| Script | Purpose |
|---|---|
| `extract_tabicl_features.py` | Pre-extract TabICL row representations for probing mode |
| `dvm_plots.py` | Generate accuracy / F1 plots from the JSON results |
| `baseline_heads.py` | PyTorch `LinearProbe` / `MLPHead` + `train_head` / `predict_head` |
