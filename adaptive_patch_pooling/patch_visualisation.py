"""Patch-quality figure generation.

All functions here are pure matplotlib — no dataset loading, no TabICL.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from adaptive_patch_pooling.patch_pooling import compute_patch_entropy, compute_patch_pooling_weights


# ---------------------------------------------------------------------------
# Low-level helpers
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


# ---------------------------------------------------------------------------
# Per-image figure
# ---------------------------------------------------------------------------

def visualise_image(
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
    ridge_pred_logits: Optional[np.ndarray] = None,   # [P]  Ridge-predicted quality logits
) -> plt.Figure:
    """Figure with overlay panels showing per-patch softmax quality scores.
    Panels: original | P(true) | logit weights | entropy | entropy/entropy-logit weights
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


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

def summary_figure(results: list[dict]) -> plt.Figure:
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
