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
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLClassifier
from torch.utils.data import Dataset
from tqdm import tqdm

from adaptive_patch_pooling.patch_pooling import (
    _ridge_pool_weights,
    group_patches,
    refine_dataset_features,
)
from adaptive_patch_pooling.patch_visualisation import summary_figure, visualise_image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUTTERFLY_DATASET_PATH = Path("/project/aip-rahulgk/hermanb/datasets/butterfly-image-classification")
RSNA_DATASET_PATH      = Path("/project/aip-rahulgk/hermanb/datasets/rsna-pneumonia")
DATASET_PATH           = BUTTERFLY_DATASET_PATH   # kept for backward compat
FEATURES_DIR           = Path("/scratch/hermanb/temp_datasets/extracted_features")


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
# RSNA helpers
# ---------------------------------------------------------------------------

def _dicom_to_pil(path: Path) -> Image.Image:
    """Load a DICOM file and return an 8-bit grayscale-as-RGB PIL Image."""
    import pydicom
    dcm    = pydicom.dcmread(str(path))
    pixels = dcm.pixel_array.astype(np.float32)
    pi     = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    if pi.strip() == "MONOCHROME1":
        pixels = pixels.max() - pixels
    pmin, pmax = pixels.min(), pixels.max()
    if pmax > pmin:
        pixels = (pixels - pmin) / (pmax - pmin) * 255.0
    return Image.fromarray(pixels.astype(np.uint8)).convert("RGB")


def _get_rsna_image_paths(
    dataset_path: Path,
    features_dir: Path,
    split:        str,
    backbone:     str = "rad-dino",
) -> tuple[list[Path], list[int], dict[int, str]]:
    """Return (paths, labels, idx_to_class) for the RSNA split.

    Paths and labels are read directly from the saved .pt metadata so that
    index i here corresponds exactly to row i in the patch-features tensor.
    """
    backbone_tag = backbone.replace("-", "_")
    pt_path = Path(features_dir) / f"rsna_{split}_{backbone_tag}_features.pt"
    ckpt    = torch.load(pt_path, map_location="cpu", weights_only=False)

    patient_ids  = ckpt["patient_ids"]          # list of strings
    labels_t     = ckpt["labels"]               # [N] int tensor
    class_to_idx = ckpt["class_to_idx"]         # {"Normal": 0, "Pneumonia": 1}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    image_dir = Path(dataset_path) / "stage_2_train_images"
    paths  = [image_dir / f"{pid}.dcm" for pid in patient_ids]
    labels = labels_t.tolist()
    return paths, labels, idx_to_class


# ---------------------------------------------------------------------------
# Feature loading dispatcher
# ---------------------------------------------------------------------------

def _load_features(
    dataset:      str,
    features_dir: Path,
    n_train:      Optional[int],
    seed:         int,
    dtype:        torch.dtype = torch.float32,
    backbone:     str = "rad-dino",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           Optional[np.ndarray], Optional[np.ndarray], dict[int, str],
           Optional[np.ndarray]]:
    """Load patch and CLS features for the requested dataset.

    Returns:
        train_patches:   [N_train, P, D]  float32 numpy
        train_labels:    [N_train]         int64   numpy
        test_patches:    [N_test,  P, D]  float32 numpy
        test_labels:     [N_test]          int64   numpy
        cls_train:       [N_train, D] or None
        cls_test:        [N_test,  D] or None
        idx_to_class:    {int → class_name}
        train_sub_idx:   [N_train] int64 or None — indices into the original full training set
                         that were selected; None when no subsampling was applied.  Callers
                         that maintain a parallel list of training image paths must apply
                         the same selection to keep alignment with train_patches.
    """
    features_dir = Path(features_dir)
    train_sub_idx: Optional[np.ndarray] = None

    if dataset == "butterfly":
        train_ds = ButterflyPatchDataset(features_dir, split="train", dtype=dtype)
        test_ds  = ButterflyPatchDataset(features_dir, split="test",  dtype=dtype)
        train_patches = train_ds.features.numpy()
        train_labels  = train_ds.labels.numpy()
        test_patches  = test_ds.features.numpy()
        test_labels   = test_ds.labels.numpy()

        # idx_to_class from feature labels (class indices 0..C-1)
        n_cls = int(train_labels.max()) + 1
        idx_to_class: dict[int, str] = {i: str(i) for i in range(n_cls)}

        # CLS features (optional)
        cls_train: Optional[np.ndarray] = None
        cls_test:  Optional[np.ndarray] = None
        cls_pt_train = features_dir / "butterfly_train_dinov3_features.pt"
        cls_pt_test  = features_dir / "butterfly_test_dinov3_features.pt"
        if cls_pt_train.exists() and cls_pt_test.exists():
            _ct = torch.load(cls_pt_train, map_location="cpu", weights_only=False)
            _cv = torch.load(cls_pt_test,  map_location="cpu", weights_only=False)
            cls_train = _ct["features"].float().numpy()
            cls_test  = _cv["features"].float().numpy()
        else:
            print(f"[warn] CLS feature files not found; skipping CLS baseline "
                  f"(expected {cls_pt_train} and {cls_pt_test})")

    elif dataset == "rsna":
        backbone_tag = backbone.replace("-", "_")
        pt_train = features_dir / f"rsna_train_{backbone_tag}_features.pt"
        pt_test  = features_dir / f"rsna_test_{backbone_tag}_features.pt"
        for p in (pt_train, pt_test):
            if not p.exists():
                raise FileNotFoundError(
                    f"RSNA feature file not found: {p}. "
                    "Run feature_extraction/extract_rsna_features.py first."
                )

        # Use memory-mapped loading (PyTorch >= 2.1) so only accessed pages are
        # paged in from disk — critical when n_train << N since we slice before
        # materialising the full patch tensor.
        _load_kwargs: dict = dict(map_location="cpu", weights_only=False)
        try:
            ck_train = torch.load(pt_train, mmap=True,  **_load_kwargs)
            ck_test  = torch.load(pt_test,  mmap=True,  **_load_kwargs)
        except TypeError:
            ck_train = torch.load(pt_train, **_load_kwargs)
            ck_test  = torch.load(pt_test,  **_load_kwargs)

        # Read lightweight tensors (labels, CLS) first — these are always small.
        train_labels  = ck_train["labels"].numpy().astype(np.int64)
        test_labels   = ck_test ["labels"].numpy().astype(np.int64)
        class_to_idx  = ck_train["class_to_idx"]
        idx_to_class  = {v: k for k, v in class_to_idx.items()}

        # Determine n_train subsample indices BEFORE touching the patch tensor so
        # that with mmap loading we only page in the selected rows.
        if n_train is not None and n_train < len(train_labels):
            n_orig  = len(train_labels)
            rng     = np.random.RandomState(seed)
            sub_idx = rng.choice(n_orig, size=n_train, replace=False)
            sub_idx.sort()
            train_sub_idx = sub_idx
            train_labels  = train_labels[sub_idx]
            # Slice the torch tensor before .numpy() — avoids materialising the
            # full [N, P, D] array in RAM (critical for large datasets).
            train_patches = ck_train["patch_features"][sub_idx].numpy()   # float16
            cls_train     = ck_train["cls_features"][sub_idx].float().numpy()
            print(f"[info] RSNA (train): N={n_train}/{n_orig} (subsampled)  "
                  f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        else:
            # Load full training set — keep as float16 to halve RAM vs float32.
            train_patches = ck_train["patch_features"].numpy()            # float16
            cls_train     = ck_train["cls_features"].float().numpy()
            print(f"[info] RSNA (train): N={len(train_labels)}  "
                  f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")

        del ck_train   # release checkpoint; patch tensor is now a numpy array

        test_patches = ck_test["patch_features"].numpy()                  # float16
        cls_test     = ck_test["cls_features"].float().numpy()
        del ck_test
        print(f"[info] RSNA (test):  N={len(test_labels)}")

        # n_train subsampling already applied above — skip the general block below.
        n_train = None

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choices: butterfly, rsna")

    # --- Optional n_train subsampling for non-RSNA datasets ---
    if n_train is not None and n_train < len(train_labels):
        n_orig  = len(train_labels)
        rng     = np.random.RandomState(seed)
        sub_idx = rng.choice(n_orig, size=n_train, replace=False)
        sub_idx.sort()
        train_sub_idx = sub_idx
        train_patches = train_patches[sub_idx]
        train_labels  = train_labels[sub_idx]
        if cls_train is not None:
            cls_train = cls_train[sub_idx]
        print(f"[info] Training set subsampled: {n_train} / {n_orig} images")

    return train_patches, train_labels, test_patches, test_labels, cls_train, cls_test, idx_to_class, train_sub_idx


def _balance_classes(
    patches:   np.ndarray,           # [N, P, D]
    labels:    np.ndarray,           # [N]
    cls_feats: Optional[np.ndarray], # [N, D] or None
    rng:       np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Undersample majority classes so every class has the same number of examples.

    Samples are drawn without replacement.  The returned arrays are in a
    random (shuffled) order so the caller does not need to shuffle again.

    Returns (patches, labels, cls_feats, keep_idx) where keep_idx are the
    indices into the input arrays that were selected, in the returned order.
    Callers that maintain parallel lists (e.g. image paths) must apply the
    same keep_idx to stay aligned.
    """
    classes, counts = np.unique(labels, return_counts=True)
    n_min = int(counts.min())
    keep: list[np.ndarray] = []
    for cls in classes:
        idx = np.where(labels == cls)[0]
        keep.append(rng.choice(idx, size=n_min, replace=False))
    keep_idx = np.concatenate(keep)
    rng.shuffle(keep_idx)   # mix classes so support set isn't class-sorted

    bal_patches = patches[keep_idx]
    bal_labels  = labels[keep_idx]
    bal_cls     = cls_feats[keep_idx] if cls_feats is not None else None

    orig_counts_str = "  ".join(f"cls{c}:{n}" for c, n in zip(classes, counts))
    print(f"[balance] {len(labels)} → {len(bal_labels)} samples  "
          f"({n_min} per class)  was: {orig_counts_str}")
    return bal_patches, bal_labels, bal_cls, keep_idx


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
) -> tuple[float, float]:
    """Classify pre-projected query features against a support set.

    Returns:
        (accuracy, auroc)  — auroc is NaN if it cannot be computed (e.g. single class in test set).
    """
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(support_features, support_labels)
    proba = clf.predict_proba(query_features)   # [N_test, n_classes]
    acc   = float((np.argmax(proba, axis=1) == query_labels).mean())
    try:
        if proba.shape[1] == 2:
            auroc = float(roc_auc_score(query_labels, proba[:, 1]))
        else:
            auroc = float(roc_auc_score(query_labels, proba, multi_class="ovr", average="macro"))
    except ValueError:
        auroc = float("nan")
    return acc, auroc


def _compute_accuracy(
    support_features: np.ndarray,   # [N_train, d]
    support_labels:   np.ndarray,   # [N_train]
    test_patches:     np.ndarray,   # [N_test, P, D]
    test_labels:      np.ndarray,   # [N_test]
    pca:              Optional[PCA],
    n_estimators:     int = 1,
    seed:             int = 42,
) -> tuple[float, float]:
    """Accuracy and AUROC of TabICL on the held-out test set using mean-pooled test queries."""
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
    ridge_model:      Optional[Ridge] = None,
    feature_scaler:   Optional[StandardScaler] = None,
    open_image:       Optional[Callable[[Path], Image.Image]] = None,
    class_prior:      Optional[np.ndarray] = None,   # [n_classes] empirical class frequencies
    weight_method:    str   = "correct_class_prob",
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

            _opener = open_image or (lambda p: Image.open(p).convert("RGB"))
            img = _opener(image_paths[img_idx])
            fig = visualise_image(
                img, probs, true_label, idx_to_class,
                n_classes=n_classes,
                patch_size=patch_size,
                temperature=temperature,
                ridge_pred_logits=ridge_pred_logits,
                class_prior=class_prior,
                weight_method=weight_method,
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

def _save_results(
    output_dir:    Path,
    run_ts:        str,
    cli_args:      Optional[dict],
    total_time_s:  float,
    train_patches: np.ndarray,
    test_labels:   np.ndarray,
    D:             int,
    n_classes:     int,
    pca:           Optional[PCA],
    cls_acc:       Optional[float],
    cls_auroc:     Optional[float],
    baseline_acc:  float,
    baseline_auroc: float,
    all_results:   list,
    attn_result:   Optional[dict] = None,
) -> None:
    """Serialise experiment results to output_dir/results.json."""
    def _fmt(v: float) -> Optional[float]:
        return round(v, 6) if not np.isnan(v) else None

    record: dict = {
        "run_timestamp": run_ts,
        "total_time_s":  round(total_time_s, 2),
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in (cli_args or {}).items()
        },
        "dataset": {
            "n_train":   int(train_patches.shape[0]),
            "n_test":    int(len(test_labels)),
            "n_patches": int(train_patches.shape[1]),
            "embed_dim": int(D),
            "n_classes": int(n_classes),
            "pca_dim":   int(pca.n_components_) if pca is not None else None,
        },
        "baselines": {
            "cls_token":    round(float(cls_acc), 6) if cls_acc is not None else None,
            "cls_token_auroc": _fmt(cls_auroc) if cls_auroc is not None else None,
            "mean_pool":    round(float(baseline_acc), 6),
            "mean_pool_auroc": _fmt(baseline_auroc),
            "attn_pool":    attn_result,
        },
        "stages": [
            {
                "tag":             stage_name,
                "test_accuracy":   round(float(acc), 6),
                "test_auroc":      _fmt(auroc),
                "delta_acc":       round(float(acc - baseline_acc), 6),
                "delta_auroc":     _fmt(auroc - baseline_auroc) if not np.isnan(auroc) and not np.isnan(baseline_auroc) else None,
                "mean_prob_train": round(float(mean_probs.get("train", float("nan"))), 6),
                "mean_prob_test":  round(float(mean_probs.get("test",  float("nan"))), 6),
                "fit_time_s":      round(fit_s,    2),
                "pool_time_s":     round(pool_s,   2),
                "refine_time_s":   round(refine_s, 2),
                "eval_time_s":     round(eval_s,   2),
            }
            for stage_name, acc, auroc, mean_probs, refine_s, eval_s, fit_s, pool_s in all_results
        ],
    }
    results_path = output_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[results] Saved → {results_path}")


def _run_attn_only(
    train_patches:        np.ndarray,
    train_labels:         np.ndarray,
    test_patches:         np.ndarray,
    test_labels:          np.ndarray,
    D:                    int,
    output_dir:           Path,
    attn_steps:           int,
    attn_lr:              float,
    attn_max_step_samples: int,
    attn_num_queries:     int,
    attn_num_heads:       int,
    device:               str,
    seed:                 int,
    pca_dim:              Optional[int],
    n_estimators:         int,
    cli_args:             Optional[dict],
) -> None:
    """Train the attention pooling head and save results to attn_pool_results.json.

    Checkpoint selection uses the best validation accuracy from the training loop
    (full 768-dim features via the frozen TabICL backbone).  The reported test
    accuracy is evaluated post-hoc: pool train with the best head, fit PCA if
    pca_dim is set, then call TabICLClassifier on the PCA-reduced features —
    matching the dimensionality used by all other baselines.
    """
    import torch as _torch
    from attention_pooling_experiments import train_attention_pooling_head, _pool_with_head
    from finetune_projection_head import ProjectionTrainingConfig

    if device == "auto":
        _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    else:
        _device = _torch.device(device)

    attn_cfg = ProjectionTrainingConfig(
        num_steps=attn_steps,
        learning_rate=attn_lr,
        max_step_samples=attn_max_step_samples,
        seed=seed,
        log_every=max(1, attn_steps // 10),
    )
    print(f"\n[attn-pool]  Training attention head  "
          f"(steps={attn_steps}  lr={attn_lr}  device={_device}  "
          f"n_queries={attn_num_queries}  n_heads={attn_num_heads}  "
          f"n_train={len(train_labels)})")

    t_start = time.perf_counter()
    head, attn_history = train_attention_pooling_head(
        train_patches=_torch.from_numpy(train_patches),
        y_train=train_labels,
        val_patches=_torch.from_numpy(test_patches),
        y_val=test_labels,
        embed_dim=D,
        out_dim=None,
        num_queries=attn_num_queries,
        num_heads=attn_num_heads,
        device=_device,
        config=attn_cfg,
    )
    total_time_s = time.perf_counter() - t_start

    best_val_acc_raw = max(attn_history["val_accuracy"]) if attn_history["val_accuracy"] else float("nan")
    time_to_best_s   = attn_history.get("time_to_best_s", float("nan"))
    best_val_step    = attn_history.get("best_val_step", 0)

    # Post-hoc evaluation: pool with best checkpoint → PCA (if used) → TabICLClassifier
    print(f"[attn-pool]  Evaluating best checkpoint (step {best_val_step}) with PCA={pca_dim} ...")
    train_pooled = _pool_with_head(head, _torch.from_numpy(train_patches), _device)
    test_pooled  = _pool_with_head(head, _torch.from_numpy(test_patches),  _device)
    if pca_dim is not None:
        n_comp_attn = min(pca_dim, len(train_labels), train_pooled.shape[1])
        attn_pca    = PCA(n_components=n_comp_attn, random_state=seed)
        train_pooled = attn_pca.fit_transform(train_pooled).astype(np.float32)
        test_pooled  = attn_pca.transform(test_pooled).astype(np.float32)
    test_acc, test_auroc = _compute_accuracy_from_features(
        train_pooled, train_labels, test_pooled, test_labels,
        n_estimators=n_estimators, seed=seed,
    )

    attn_result = {
        "test_acc":           round(test_acc, 6),
        "test_auroc":         round(test_auroc, 6) if not np.isnan(test_auroc) else None,
        "best_val_acc_raw":   round(best_val_acc_raw, 6),
        "best_val_step":      best_val_step,
        "time_to_best_s":     time_to_best_s,
        "total_train_time_s": round(total_time_s, 2),
    }
    print(f"[attn-pool]  test acc (PCA={pca_dim}): {test_acc:.4f}  auroc: {test_auroc:.4f}  "
          f"(best train val: {best_val_acc_raw:.4f}  "
          f"step {best_val_step}/{attn_steps}  time_to_best={time_to_best_s:.1f}s)")

    record = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in (cli_args or {}).items()},
        "dataset": {"n_train": int(len(train_labels)), "n_test": int(len(test_labels)), "embed_dim": int(D)},
        "attn_pool": attn_result,
    }
    attn_path = output_dir / "attn_pool_results.json"
    with attn_path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"[attn-pool]  Saved → {attn_path}")


def _merge_attn_into_results(output_dir: Path) -> bool:
    """Patch baselines.attn_pool in results.json from attn_pool_results.json.

    Returns True if both files exist and the merge succeeded.
    """
    attn_path    = output_dir / "attn_pool_results.json"
    results_path = output_dir / "results.json"
    if not attn_path.exists() or not results_path.exists():
        return False
    with attn_path.open() as f:
        attn_data = json.load(f)
    with results_path.open() as f:
        results_data = json.load(f)
    results_data.setdefault("baselines", {})["attn_pool"] = attn_data.get("attn_pool")
    with results_path.open("w") as f:
        json.dump(results_data, f, indent=2)
    print(f"[merge] attn_pool → {results_path}")
    return True


def run_patch_quality_eval(
    features_dir:      Path          = FEATURES_DIR,
    dataset_path:      Path          = DATASET_PATH,
    dataset:           str           = "butterfly",
    backbone:          str           = "rad-dino",
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
    weight_method:     str           = "correct_class_prob",
    mix_lambda:        float         = 1.0,
    ridge_alpha:       float         = 1.0,
    normalize_features: bool         = False,
    max_query_rows:         Optional[int] = None,
    use_random_subsampling: bool          = False,
    balance_train:          bool          = False,
    balance_test:           bool          = False,
    # Attention pooling upper-bound baseline
    attn_pool:              bool          = False,
    attn_pool_only:         bool          = False,   # skip all other stages, only train attn
    attn_steps:             int           = 500,
    attn_lr:                float         = 1e-3,
    attn_max_step_samples:  int           = 512,
    attn_num_queries:       int           = 1,
    attn_num_heads:         int           = 8,
    device:                 str           = "auto",
    post_refinement_viz:    bool          = False,   # skip pre-refinement viz; show only post-refinement Ridge panels
    aoe_class:              Optional[str] = None,    # absence-of-evidence class (int index or class name)
    aoe_handling:           str           = "filter",  # how to handle AoE class: "filter" | "entropy"
    gpu_ridge:              bool          = False,   # solve Ridge on GPU (requires PyTorch + CUDA)
    _cli_args:              Optional[dict] = None,   # raw CLI args for provenance logging
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_start = time.perf_counter()
    run_ts = datetime.now(timezone.utc).isoformat()

    # --- Load pre-extracted patch (and CLS) features ---
    (train_patches, train_labels,
     test_patches,  test_labels,
     cls_train_feats, cls_test_feats,
     idx_to_class, train_sub_idx) = _load_features(
        dataset=dataset,
        features_dir=features_dir,
        n_train=n_train,
        seed=seed,
        backbone=backbone,
    )

    bal_rng = np.random.RandomState(seed + 1)   # separate RNG so balancing doesn't shift other draws
    bal_train_keep_idx: Optional[np.ndarray] = None
    if balance_train:
        train_patches, train_labels, cls_train_feats, bal_train_keep_idx = _balance_classes(
            train_patches, train_labels, cls_train_feats, bal_rng
        )
    if balance_test:
        test_patches, test_labels, cls_test_feats, _ = _balance_classes(
            test_patches, test_labels, cls_test_feats, bal_rng
        )

    N_train, _P, D = train_patches.shape
    n_classes      = int(train_labels.max()) + 1

    # Empirical class prior (used by kl_div weight method and visualisation panels).
    _counts      = np.bincount(train_labels.astype(np.int64), minlength=n_classes)
    class_prior  = (_counts / _counts.sum()).astype(np.float32)

    # --- Resolve absence-of-evidence class ---
    aoe_mask: Optional[np.ndarray] = None
    if aoe_class is not None:
        class_to_idx = {v: k for k, v in idx_to_class.items()}
        try:
            aoe_class_idx = int(aoe_class)
        except (ValueError, TypeError):
            aoe_class_idx = None
        if aoe_class_idx is not None:
            if aoe_class_idx not in idx_to_class:
                raise ValueError(f"aoe_class index {aoe_class_idx} not in [0, {n_classes - 1}]")
        else:
            name = str(aoe_class)
            if name not in class_to_idx:
                raise ValueError(
                    f"aoe_class '{name}' not found in class names. "
                    f"Available: {sorted(class_to_idx)}"
                )
            aoe_class_idx = class_to_idx[name]
        aoe_class_name = idx_to_class[aoe_class_idx]
        aoe_mask = (train_labels == aoe_class_idx)
        print(f"[aoe] Absence-of-evidence class: '{aoe_class_name}' (index {aoe_class_idx}), "
              f"{int(aoe_mask.sum())} training images excluded from Ridge fitting")

    # --- Attn-pool-only fast path: skip all feature refinement stages ---
    if attn_pool_only:
        _run_attn_only(
            train_patches=train_patches, train_labels=train_labels,
            test_patches=test_patches,   test_labels=test_labels,
            D=D, output_dir=output_dir,
            attn_steps=attn_steps, attn_lr=attn_lr,
            attn_max_step_samples=attn_max_step_samples,
            attn_num_queries=attn_num_queries, attn_num_heads=attn_num_heads,
            device=device, seed=seed, pca_dim=pca_dim,
            n_estimators=n_estimators, cli_args=_cli_args,
        )
        _merge_attn_into_results(output_dir)
        return

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
    # Cast to float32 before mean to avoid float16 accumulation errors.
    baseline_support_raw = train_patches.astype(np.float32).mean(axis=1)   # [N_train, D]
    pca: Optional[PCA] = None
    if pca_dim is not None:
        n_comp = min(pca_dim, N_train, D)
        pca    = PCA(n_components=n_comp, random_state=seed)
        baseline_support = pca.fit_transform(baseline_support_raw).astype(np.float32)
        print(f"[info] PCA: {D}D → {n_comp}D")
    else:
        baseline_support = baseline_support_raw

    # --- CLS token baseline ---
    cls_acc:   Optional[float] = None
    cls_auroc: Optional[float] = None
    if cls_train_feats is not None and cls_test_feats is not None:
        cls_pca: Optional[PCA] = None
        if pca_dim is not None:
            n_comp_cls  = min(pca_dim, len(cls_train_feats), cls_train_feats.shape[1])
            cls_pca     = PCA(n_components=n_comp_cls, random_state=seed)
            cls_support = cls_pca.fit_transform(cls_train_feats).astype(np.float32)
            cls_test_q  = cls_pca.transform(cls_test_feats).astype(np.float32)
        else:
            cls_support = cls_train_feats
            cls_test_q  = cls_test_feats
        cls_acc, cls_auroc = _compute_accuracy_from_features(
            cls_support, train_labels, cls_test_q, test_labels,
            n_estimators=n_estimators, seed=seed,
        )

    # --- Image paths + opener for visualisation (only loaded when needed) ---
    train_image_paths: list = []
    test_image_paths:  list = []
    open_image: Optional[Callable] = None
    if n_sample > 0:
        if dataset == "butterfly":
            train_image_paths, _, idx_to_class = _get_image_paths(dataset_path, split="train", seed=seed)
            test_image_paths,  _, _            = _get_image_paths(dataset_path, split="test",  seed=seed)
        elif dataset == "rsna":
            train_image_paths, _, _ = _get_rsna_image_paths(dataset_path, features_dir, split="train", backbone=backbone)
            test_image_paths,  _, _ = _get_rsna_image_paths(dataset_path, features_dir, split="test",  backbone=backbone)
            open_image = _dicom_to_pil

        # Keep train_image_paths aligned with train_patches by applying the same
        # index selections that _load_features and _balance_classes applied.
        if train_sub_idx is not None:
            train_image_paths = [train_image_paths[i] for i in train_sub_idx]
        if bal_train_keep_idx is not None:
            train_image_paths = [train_image_paths[i] for i in bal_train_keep_idx]

    rng              = np.random.RandomState(seed)
    train_sample_idx = rng.choice(len(train_labels), size=min(n_sample, len(train_labels)), replace=False)
    test_sample_idx  = rng.choice(len(test_labels),  size=min(n_sample, len(test_labels)),  replace=False)

    # --- Baseline: accuracy + visual eval at original patch resolution ---
    baseline_acc, baseline_auroc = _compute_accuracy(
        baseline_support, train_labels, test_patches, test_labels,
        pca=pca, n_estimators=n_estimators, seed=seed,
    )
    if cls_acc is not None:
        print(f"\n[cls-token]  test accuracy: {cls_acc:.4f}  auroc: {cls_auroc:.4f}")
    else:
        print("\n[cls-token]  test accuracy: N/A (files not found)")
    print(f"[mean-pool]  test accuracy: {baseline_acc:.4f}  auroc: {baseline_auroc:.4f}")

    # --- Attention pooling upper-bound baseline ---
    attn_result: Optional[dict] = None
    if attn_pool:
        import torch as _torch
        from attention_pooling_experiments import train_attention_pooling_head, _pool_with_head
        from finetune_projection_head import ProjectionTrainingConfig

        if device == "auto":
            _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        else:
            _device = _torch.device(device)

        attn_cfg = ProjectionTrainingConfig(
            num_steps=attn_steps,
            learning_rate=attn_lr,
            max_step_samples=attn_max_step_samples,
            seed=seed,
            log_every=max(1, attn_steps // 10),
        )
        print(f"\n[attn-pool]  Training attention head  "
              f"(steps={attn_steps}  lr={attn_lr}  device={_device}  "
              f"n_queries={attn_num_queries}  n_heads={attn_num_heads})")

        t_attn_start = time.perf_counter()
        attn_head, attn_history = train_attention_pooling_head(
            train_patches=_torch.from_numpy(train_patches),
            y_train=train_labels,
            val_patches=_torch.from_numpy(test_patches),
            y_val=test_labels,
            embed_dim=D,
            out_dim=None,
            num_queries=attn_num_queries,
            num_heads=attn_num_heads,
            device=_device,
            config=attn_cfg,
        )
        attn_total_time_s = time.perf_counter() - t_attn_start

        attn_best_val_acc_raw = max(attn_history["val_accuracy"]) if attn_history["val_accuracy"] else float("nan")
        attn_time_to_best     = attn_history.get("time_to_best_s", float("nan"))
        attn_best_step        = attn_history.get("best_val_step", 0)

        # Post-hoc evaluation: pool with best checkpoint → PCA → TabICLClassifier
        from attention_pooling_experiments import _pool_with_head as _attn_pool_fn
        print(f"[attn-pool]  Evaluating best checkpoint (step {attn_best_step}) with PCA={pca_dim} ...")
        attn_train_pooled = _attn_pool_fn(attn_head, _torch.from_numpy(train_patches), _device)
        attn_test_pooled  = _attn_pool_fn(attn_head, _torch.from_numpy(test_patches),  _device)
        if pca_dim is not None:
            n_comp_attn       = min(pca_dim, len(train_labels), attn_train_pooled.shape[1])
            attn_pca          = PCA(n_components=n_comp_attn, random_state=seed)
            attn_train_pooled = attn_pca.fit_transform(attn_train_pooled).astype(np.float32)
            attn_test_pooled  = attn_pca.transform(attn_test_pooled).astype(np.float32)
        attn_test_acc, attn_test_auroc = _compute_accuracy_from_features(
            attn_train_pooled, train_labels, attn_test_pooled, test_labels,
            n_estimators=n_estimators, seed=seed,
        )

        attn_result = {
            "test_acc":           round(attn_test_acc, 6),
            "test_auroc":         round(attn_test_auroc, 6) if not np.isnan(attn_test_auroc) else None,
            "best_val_acc_raw":   round(attn_best_val_acc_raw, 6),
            "best_val_step":      attn_best_step,
            "time_to_best_s":     attn_time_to_best,
            "total_train_time_s": round(attn_total_time_s, 2),
        }
        print(f"[attn-pool]  test acc (PCA={pca_dim}): {attn_test_acc:.4f}  auroc: {attn_test_auroc:.4f}  "
              f"(best train val: {attn_best_val_acc_raw:.4f}  "
              f"step {attn_best_step}/{attn_steps}  time_to_best={attn_time_to_best:.1f}s)")

    if n_sample > 0 and not post_refinement_viz:
        split_configs_orig = [
            ("train", train_patches, train_labels, train_image_paths, train_sample_idx),
            ("test",  test_patches,  test_labels,  test_image_paths,  test_sample_idx),
        ]
        baseline_mean_probs = _run_visual_eval(
            "baseline", baseline_support, train_labels, split_configs_orig, idx_to_class,
            pca=pca, n_estimators=n_estimators, patch_size=patch_size,
            seed=seed, output_dir=output_dir,
            temperature=temperatures[0],
            ridge_model=None, feature_scaler=None, open_image=open_image,
            class_prior=class_prior, weight_method=weight_method,
        )
    else:
        baseline_mean_probs = {}

    if not refine:
        _save_results(
            output_dir=output_dir, run_ts=run_ts, cli_args=_cli_args,
            total_time_s=time.perf_counter() - experiment_start,
            train_patches=train_patches, test_labels=test_labels, D=D,
            n_classes=n_classes, pca=pca,
            cls_acc=cls_acc, cls_auroc=cls_auroc,
            baseline_acc=baseline_acc, baseline_auroc=baseline_auroc,
            all_results=[("baseline", baseline_acc, baseline_auroc, baseline_mean_probs, 0.0, 0.0, 0.0, 0.0)],
            attn_result=attn_result,
        )
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
    all_results: list[tuple[str, float, float, dict, float, float]] = [
        ("baseline", baseline_acc, baseline_auroc, baseline_mean_probs, 0.0, 0.0, 0.0, 0.0)
    ]

    for stage_idx, group_size in enumerate(patch_group_sizes):
        stage_temp  = temperatures[stage_idx]
        stage_alpha = ridge_alphas[stage_idx]
        group_side   = int(round(group_size ** 0.5))
        eff_patch_sz = patch_size * group_side
        tag          = f"iter_{stage_idx}_g{group_size}"

        print(f"\n[{tag}] group_size={group_size}  ({group_side}×{group_side} patches per group)  "
              f"T={stage_temp}  ridge_alpha={stage_alpha}")

        train_grouped = group_patches(train_patches, group_size)   # [N, P', D]
        test_grouped  = group_patches(test_patches,  group_size)   # [N_test, P', D]
        P_grouped     = train_grouped.shape[1]

        # -- Visualise patch quality under the *input* support (before refinement) --
        # This matches exactly what will drive the pooling weights this stage.
        if n_sample > 0 and not post_refinement_viz:
            split_configs_iter = [
                ("train", train_grouped, train_labels, train_image_paths, train_sample_idx),
                ("test",  test_grouped,  test_labels,  test_image_paths,  test_sample_idx),
            ]
            iter_mean_probs = _run_visual_eval(
                tag, current_support, train_labels, split_configs_iter, idx_to_class,
                pca=current_pca, n_estimators=n_estimators, patch_size=eff_patch_sz,
                seed=seed, output_dir=output_dir,
                temperature=stage_temp,
                ridge_model=None, feature_scaler=None, open_image=open_image,
                class_prior=class_prior, weight_method=weight_method,
            )
        else:
            iter_mean_probs = {}

        # -- Refine support (clf fitted on current_support internally) --
        # Save pre-refinement state so the final post-all-refinement viz can use it.
        pre_refine_support = current_support
        pre_refine_pca     = current_pca
        print(f"[{tag}] Refining support "
              f"(method={weight_method}, T={stage_temp}) ...")
        new_support, new_pca, _weights, ridge_model, feature_scaler, scoring_clf, \
            fit_time_s, pool_time_s = \
            refine_dataset_features(
                train_grouped, train_labels, current_support, current_pca,
                n_estimators=n_estimators, temperature=stage_temp, seed=seed,
                batch_size=batch_size, weight_method=weight_method,
                mix_lambda=mix_lambda, ridge_alpha=stage_alpha,
                normalize_features=normalize_features,
                max_query_rows=max_query_rows,
                use_random_subsampling=use_random_subsampling,
                aoe_mask=aoe_mask,
                aoe_handling=aoe_handling,
                use_gpu_ridge=gpu_ridge,
                gpu_ridge_device="cuda" if device == "auto" else device,
            )
        refine_time_s = fit_time_s + pool_time_s

        if ridge_model is not None:
            ridge_path = output_dir / f"ridge_quality_model_{tag}.joblib"
            joblib.dump(ridge_model, ridge_path)
            print(f"[ridge] Model saved → {ridge_path}")

        # -- Post-refinement visualisation with Ridge pooling weights --
        if n_sample > 0 and post_refinement_viz and ridge_model is not None:
            split_configs_post = [
                ("train", train_grouped, train_labels, train_image_paths, train_sample_idx),
                ("test",  test_grouped,  test_labels,  test_image_paths,  test_sample_idx),
            ]
            iter_mean_probs = _run_visual_eval(
                f"{tag}_post", current_support, train_labels, split_configs_post, idx_to_class,
                pca=current_pca, n_estimators=n_estimators, patch_size=eff_patch_sz,
                seed=seed, output_dir=output_dir,
                temperature=stage_temp,
                ridge_model=ridge_model, feature_scaler=feature_scaler, open_image=open_image,
                class_prior=class_prior, weight_method=weight_method,
            )

        # -- Pool test queries with Ridge (full images, same model as training) --
        w_ridge       = _ridge_pool_weights(test_grouped, ridge_model, feature_scaler)
        test_repooled = (w_ridge[:, :, None] * test_grouped).sum(axis=1)   # [N_test, D]

        test_query = (
            new_pca.transform(test_repooled).astype(np.float32)
            if new_pca is not None else test_repooled
        )

        # -- Evaluate accuracy with quality-pooled test queries --
        t_eval_start = time.perf_counter()
        iter_acc, iter_auroc = _compute_accuracy_from_features(
            new_support, train_labels, test_query, test_labels,
            n_estimators=n_estimators, seed=seed,
        )
        eval_time_s = time.perf_counter() - t_eval_start
        print(f"[{tag}] test accuracy (quality-pooled queries): {iter_acc:.4f}  auroc: {iter_auroc:.4f}  "
              f"(fit {fit_time_s:.1f}s, pool {pool_time_s:.1f}s, eval {eval_time_s:.1f}s)")

        all_results.append((tag, iter_acc, iter_auroc, iter_mean_probs, refine_time_s, eval_time_s, fit_time_s, pool_time_s))
        current_support = new_support
        current_pca     = new_pca

    # -- Final post-all-refinement visualisation (only when --post-refinement-viz is off) --
    # Produces Ridge-weight figures for the last refinement stage, giving you the quality
    # heatmaps even when per-stage post-refinement viz was skipped.
    if n_sample > 0 and not post_refinement_viz and refine and ridge_model is not None:
        split_configs_final = [
            ("train", train_grouped, train_labels, train_image_paths, train_sample_idx),
            ("test",  test_grouped,  test_labels,  test_image_paths,  test_sample_idx),
        ]
        _run_visual_eval(
            f"{tag}_post", pre_refine_support, train_labels, split_configs_final, idx_to_class,
            pca=pre_refine_pca, n_estimators=n_estimators, patch_size=eff_patch_sz,
            seed=seed, output_dir=output_dir,
            temperature=stage_temp,
            ridge_model=ridge_model, feature_scaler=feature_scaler, open_image=open_image,
            class_prior=class_prior, weight_method=weight_method,
        )

    total_time_s = time.perf_counter() - experiment_start

    # --- Summary table ---
    col_w = max(len(r[0]) for r in all_results) + 2
    print("\n" + "=" * (col_w + 78))
    print("ITERATIVE REFINEMENT SUMMARY")
    print("=" * (col_w + 70))
    print(f"  {'Stage':<{col_w}}  {'Test Acc':>10}  {'AUROC':>8}  {'Δ Acc':>8}  "
          f"{'P(true)/train':>14}  {'P(true)/test':>13}  {'Fit(s)':>8}  {'Pool(s)':>8}  {'Eval(s)':>8}")
    print("-" * (col_w + 78))
    for stage_name, acc, auroc, mean_probs, refine_s, eval_s, fit_s, pool_s in all_results:
        delta_str = "" if stage_name == "baseline" else f"{acc - baseline_acc:+.4f}"
        fit_str   = "-" if stage_name == "baseline" else f"{fit_s:.1f}"
        pool_str  = "-" if stage_name == "baseline" else f"{pool_s:.1f}"
        eval_str  = "-" if stage_name == "baseline" else f"{eval_s:.1f}"
        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "  N/A"
        print(
            f"  {stage_name:<{col_w}}  {acc:>10.4f}  {auroc_str:>8}  {delta_str:>8}"
            f"  {mean_probs.get('train', float('nan')):>14.3f}"
            f"  {mean_probs.get('test', float('nan')):>13.3f}"
            f"  {fit_str:>8}  {pool_str:>8}  {eval_str:>8}"
        )
    print("=" * (col_w + 78))
    print(f"  Total wall time: {total_time_s:.1f}s")

    _save_results(
        output_dir=output_dir, run_ts=run_ts, cli_args=_cli_args,
        total_time_s=total_time_s,
        train_patches=train_patches, test_labels=test_labels, D=D,
        n_classes=n_classes, pca=pca,
        cls_acc=cls_acc, cls_auroc=cls_auroc,
        baseline_acc=baseline_acc, baseline_auroc=baseline_auroc,
        all_results=all_results,
        attn_result=attn_result,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_n_train_sweep(
    n_train_values: list[int],
    base_output_dir: Path,
    **kwargs,
) -> None:
    """Run patch-quality eval for each value in n_train_values.

    Each run is saved under ``base_output_dir/n_train_{value}/``.
    A consolidated ``sweep_results.json`` is written to ``base_output_dir``.

    Args:
        n_train_values: Ordered list of support-set sizes to evaluate.
        base_output_dir: Root directory; per-run sub-dirs are created here.
        **kwargs: Forwarded verbatim to ``run_patch_quality_eval`` (except
            ``n_train`` and ``output_dir``, which are set per-run).
    """
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    sweep_start = time.perf_counter()
    sweep_ts    = datetime.now(timezone.utc).isoformat()

    sweep_runs: list[dict] = []

    for n_train in n_train_values:
        run_dir = base_output_dir / f"n_train_{n_train}"
        print(f"\n{'='*60}")
        print(f"  SWEEP  n_train={n_train}  →  {run_dir}")
        print(f"{'='*60}")

        # Override n_train / output_dir; record in cli_args for provenance
        run_cli_args = dict(kwargs.get("_cli_args") or {})
        run_cli_args["n_train"]   = n_train
        run_cli_args["output_dir"] = str(run_dir)

        run_patch_quality_eval(
            n_train=n_train,
            output_dir=run_dir,
            **{k: v for k, v in kwargs.items() if k != "_cli_args"},
            _cli_args=run_cli_args,
        )

        # Read back results.json (may have been created or merged into)
        results_path  = run_dir / "results.json"
        attn_path     = run_dir / "attn_pool_results.json"
        run_summary: dict = {"n_train": n_train, "output_dir": str(run_dir)}
        if results_path.exists():
            with results_path.open() as f:
                run_data = json.load(f)
            run_summary["baselines"]    = run_data.get("baselines", {})
            run_summary["stages"]       = run_data.get("stages", [])
            run_summary["total_time_s"] = run_data.get("total_time_s")
        elif attn_path.exists():
            # attn-only run with no prior results.json
            with attn_path.open() as f:
                attn_data = json.load(f)
            run_summary["baselines"] = {"attn_pool": attn_data.get("attn_pool")}
        sweep_runs.append(run_summary)

    sweep_total = time.perf_counter() - sweep_start

    sweep_record = {
        "sweep_timestamp": sweep_ts,
        "n_train_values":  n_train_values,
        "total_sweep_time_s": round(sweep_total, 2),
        "runs": sweep_runs,
    }
    sweep_path = base_output_dir / "sweep_results.json"
    with sweep_path.open("w") as f:
        json.dump(sweep_record, f, indent=2)
    print(f"\n[sweep] Done — {len(n_train_values)} runs in {sweep_total:.1f}s")
    print(f"[sweep] Results → {sweep_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch quality evaluation with TabICL")
    p.add_argument("--dataset",       type=str,   default="butterfly",
                   choices=["butterfly", "rsna"],
                   help="Which dataset to run on (default: butterfly)")
    p.add_argument("--backbone",      type=str,   default="rad-dino",
                   choices=["rad-dino", "dinov3"],
                   help="Which backbone's features to load for RSNA "
                        "(default: rad-dino; ignored for butterfly)")
    p.add_argument("--features-dir",  type=Path,  default=FEATURES_DIR)
    p.add_argument("--dataset-path",  type=Path,  default=None,
                   help="Root path of the raw dataset (images + labels). "
                        "Defaults: butterfly → butterfly-image-classification, "
                        "rsna → rsna-pneumonia")
    p.add_argument("--n-sample",      type=int,   default=0)
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
    p.add_argument("--batch-size",     type=int,   default=1000,
                   help="Number of images per TabICL call during refinement")
    p.add_argument("--weight-method",  type=str,   default="correct_class_prob",
                   choices=["correct_class_prob", "entropy", "kl_div"],
                   help="How to derive patch pooling weights from TabICL probabilities: "
                        "'correct_class_prob' (default) takes log(p_true) then applies "
                        "temperature-scaled softmax; "
                        "'entropy' maps normalised entropy to [0,1], applies log, "
                        "then temperature-scaled softmax; "
                        "'kl_div' computes KL(Q||P_prior) where P_prior is the empirical "
                        "class frequency vector, normalises by the maximum achievable KL "
                        "(-log(p_min)), applies log, then temperature-scaled softmax — "
                        "rewards patches whose predictions deviate from the base rates, "
                        "making it sensitive to class imbalance. "
                        "All methods are controlled via --temperature.")
    p.add_argument("--mix-lambda",     type=float, default=1.0,
                   help="Interpolation weight between refined and mean-pooled embeddings "
                        "(1.0 → fully refined, 0.0 → fully mean-pooled; requires --refine)")
    p.add_argument("--ridge-alpha",  type=float, nargs="+",  default=[1.0],
                   help="Regularisation strength for the Ridge quality model. "
                        "Pass one value to use it for all stages, or one value per entry in "
                        "--patch-group-sizes to set a different alpha per stage.")
    p.add_argument("--normalize-features", action="store_true",
                   help="Fit a StandardScaler on training patches before Ridge fitting "
                        "(normalises each feature dimension across all N×P patches; "
                        "recommended for Ridge regression); scaler is applied at predict time too")
    p.add_argument("--max-query-rows", type=int, default=None,
                   help="Cap on the total number of patch-group rows forwarded through TabICL "
                        "per refinement stage.  When --use-random-subsampling is set and N*P' "
                        "exceeds this limit, rows are sampled uniformly at random (without "
                        "replacement) and forwarded in a single pass; otherwise all rows are "
                        "forwarded in the usual batched loop.  Only used when --refine is set.")
    p.add_argument("--use-random-subsampling", action="store_true",
                   help="Enable random subsampling of patch-group rows for Ridge fitting. "
                        "When set and N*P' exceeds --max-query-rows, a random subset of rows "
                        "is drawn and forwarded in one pass.  When not set, the full batched "
                        "loop is always used regardless of --max-query-rows.")
    p.add_argument("--balance-train", action="store_true",
                   help="Undersample majority classes in the training set so every class has "
                        "the same number of examples (equal to the minority-class count). "
                        "Applied after --n-train subsampling.")
    p.add_argument("--balance-test", action="store_true",
                   help="Undersample majority classes in the test set (same strategy as "
                        "--balance-train).  Useful when you want balanced AUROC/accuracy "
                        "estimates, but note it changes the evaluation distribution.")
    # Attention pooling upper-bound baseline
    p.add_argument("--attn-pool", action="store_true",
                   help="Train an attention pooling head (upper-bound baseline). "
                        "Reports best val accuracy reached during training.")
    p.add_argument("--attn-pool-only", action="store_true",
                   help="Skip all feature-refinement stages and only train the attention "
                        "pooling head.  Saves attn_pool_results.json and, if results.json "
                        "already exists in the output dir, merges into it.  Useful for "
                        "running the upper-bound baseline as a separate job.")
    p.add_argument("--attn-steps",            type=int,   default=500,
                   help="Training steps for the attention pooling head (default: 500)")
    p.add_argument("--attn-lr",               type=float, default=1e-3,
                   help="AdamW learning rate for attention pooling (default: 1e-3)")
    p.add_argument("--attn-max-step-samples", type=int,   default=512,
                   help="Max training rows forwarded per step (default: 512)")
    p.add_argument("--attn-num-queries",      type=int,   default=1,
                   help="Learnable query vectors (1 = CLS-like; default: 1)")
    p.add_argument("--attn-num-heads",        type=int,   default=8,
                   help="Attention heads (must divide embed_dim=768; default: 8)")
    p.add_argument("--device", type=str, default="auto",
                   help="Torch device for attention pooling training: 'auto', 'cuda', 'cpu' "
                        "(default: auto)")
    p.add_argument("--aoe-class", type=str, default=None,
                   help="Absence-of-evidence class: patches from this class receive special "
                        "handling during Ridge fitting (see --aoe-handling). May be given as "
                        "a class index (integer) or class name string. The class is always "
                        "included in the support and Ridge pooling is applied to it as normal.")
    p.add_argument("--aoe-handling", type=str, default="filter",
                   choices=["filter", "entropy"],
                   help="How to handle the AoE class during Ridge fitting (requires --aoe-class). "
                        "'filter': exclude AoE patches entirely from the TabICL forward pass "
                        "and Ridge fitting. "
                        "'entropy': include AoE patches but score them with entropy_logit "
                        "instead of --weight-method, so labels are not required for them. "
                        "(default: filter)")
    p.add_argument("--gpu-ridge", action="store_true",
                   help="Solve Ridge regression on the GPU (requires PyTorch + CUDA). "
                        "Same results as CPU Ridge; fastest speedup for --patch-group-size 1.")
    p.add_argument("--post-refinement-viz", action="store_true",
                   help="Skip pre-refinement visualisations; only produce post-refinement "
                        "figures with Ridge pooling weight panels. Requires --refine.")
    p.add_argument("--n-train-sweep", type=int, nargs="+", default=None,
                   metavar="N",
                   help="Run one experiment per value and collect results into a single "
                        "sweep_results.json.  Each run is saved under "
                        "output-dir/n_train_<N>/.  Mutually exclusive with --n-train.")
    return p.parse_args()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    args = _parse_args()

    if args.n_train_sweep is not None and args.n_train is not None:
        raise SystemExit("error: --n-train-sweep and --n-train are mutually exclusive")

    # Resolve default dataset_path based on chosen dataset
    _dataset_defaults = {"butterfly": BUTTERFLY_DATASET_PATH, "rsna": RSNA_DATASET_PATH}
    dataset_path = args.dataset_path or _dataset_defaults[args.dataset]

    # Shared kwargs forwarded to run_patch_quality_eval in both paths
    shared_kwargs = dict(
        dataset=args.dataset,
        backbone=args.backbone,
        features_dir=args.features_dir,
        dataset_path=dataset_path,
        n_sample=args.n_sample,
        n_estimators=args.n_estimators,
        pca_dim=None if args.no_pca else args.pca_dim,
        seed=args.seed,
        patch_size=args.patch_size,
        patch_group_sizes=args.patch_group_sizes,
        refine=args.refine,
        temperature=args.temperature,
        batch_size=args.batch_size,
        weight_method=args.weight_method,
        mix_lambda=args.mix_lambda,
        ridge_alpha=args.ridge_alpha,
        normalize_features=args.normalize_features,
        max_query_rows=args.max_query_rows,
        use_random_subsampling=args.use_random_subsampling,
        balance_train=args.balance_train,
        balance_test=args.balance_test,
        attn_pool=args.attn_pool or args.attn_pool_only,
        attn_pool_only=args.attn_pool_only,
        attn_steps=args.attn_steps,
        attn_lr=args.attn_lr,
        attn_max_step_samples=args.attn_max_step_samples,
        attn_num_queries=args.attn_num_queries,
        attn_num_heads=args.attn_num_heads,
        device=args.device,
        post_refinement_viz=args.post_refinement_viz,
        aoe_class=args.aoe_class,
        aoe_handling=args.aoe_handling,
        gpu_ridge=args.gpu_ridge,
    )

    if args.n_train_sweep is not None:
        run_n_train_sweep(
            n_train_values=args.n_train_sweep,
            base_output_dir=args.output_dir,
            _cli_args=vars(args),
            **shared_kwargs,
        )
    else:
        run_patch_quality_eval(
            n_train=args.n_train,
            output_dir=args.output_dir,
            _cli_args=vars(args),
            **shared_kwargs,
        )
