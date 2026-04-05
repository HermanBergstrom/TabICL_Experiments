"""Plot accuracy vs. training-set size from a sweep_results.json file.

Produces a figure with five lines:
  - CLS token          (from baselines.cls_token)
  - Mean pooling       (from stages[0] / baselines.mean_pool)
  - 1 iteration        (from stages[1])
  - 2 iterations       (from stages[2])
  - 3 iterations       (from stages[3])

X-axis is log-scaled; Y-axis is classification accuracy (0–1).

Usage:
    python adaptive_patch_pooling/plot_n_train_sweep.py sweep_results.json
    python adaptive_patch_pooling/plot_n_train_sweep.py sweep_results.json --output my_plot.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Series definitions
# ---------------------------------------------------------------------------

# Each entry: (label, extractor_fn)
# extractor_fn receives a single run dict and returns float | None

def _cls_token(run: dict) -> float | None:
    return run.get("baselines", {}).get("cls_token")


def _mean_pool(run: dict) -> float | None:
    return run.get("baselines", {}).get("mean_pool")


def _stage_acc(run: dict, stage_index: int) -> float | None:
    stages = run.get("stages", [])
    if stage_index < len(stages):
        v = stages[stage_index].get("test_accuracy")
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
    return None


def _attn_pool(run: dict) -> float | None:
    v = run.get("baselines", {}).get("attn_pool")
    if v is None:
        return None
    return v.get("test_acc")


# ---------------------------------------------------------------------------
# Time extractors  (return seconds for a single run, or None if missing)
# ---------------------------------------------------------------------------

def _stage_time_cumulative(run: dict, up_to_stage: int) -> float | None:
    """Sum of (refine_time_s + eval_time_s) for stages 1..up_to_stage."""
    stages = run.get("stages", [])
    total = 0.0
    for i in range(1, up_to_stage + 1):
        if i >= len(stages):
            return None
        s = stages[i]
        rt = s.get("refine_time_s")
        et = s.get("eval_time_s")
        if rt is None or et is None:
            return None
        total += rt + et
    return total


def _attn_time(run: dict) -> float | None:
    v = run.get("baselines", {}).get("attn_pool")
    if v is None:
        return None
    return v.get("time_to_best_s")


def _avg_time(runs: list[dict], time_extractor) -> str:
    """Return a human-readable average time string, or '' if no data."""
    times = [t for r in runs if (t := time_extractor(r)) is not None]
    if not times:
        return ""
    avg = np.mean(times)
    if avg < 60:
        return f"{avg:.1f}s"
    return f"{avg / 60:.1f}min"


SERIES: list[tuple[str, object, str, str, bool, object]] = [
    # (label,                extractor,                    color,      marker, is_attn, time_extractor)
    ("CLS token",            _cls_token,                   "#e07b39",  "D",    False,   None),
    ("Mean pooling",         _mean_pool,                   "#4c72b0",  "o",    False,   None),
    ("1 iteration",          lambda r: _stage_acc(r, 1),   "#55a868",  "s",    False,   lambda r: _stage_time_cumulative(r, 1)),
    ("2 iterations",         lambda r: _stage_acc(r, 2),   "#c44e52",  "^",    False,   lambda r: _stage_time_cumulative(r, 2)),
    ("3 iterations",         lambda r: _stage_acc(r, 3),   "#8172b2",  "P",    False,   lambda r: _stage_time_cumulative(r, 3)),
    ("Attn. pooling (UB)",   _attn_pool,                   "#937860",  "*",    True,    _attn_time),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_sweep(
    sweep_path: Path,
    output_path: Path | None = None,
    exclude_cls: bool = False,
    exclude_attn: bool = False,
) -> None:
    with sweep_path.open() as f:
        data = json.load(f)

    runs: list[dict] = data["runs"]
    # Sort by n_train so lines are drawn left-to-right
    runs = sorted(runs, key=lambda r: r["n_train"])
    x = [r["n_train"] for r in runs]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, extractor, color, marker, is_attn, time_extractor in SERIES:
        if exclude_cls and label == "CLS token":
            continue
        if exclude_attn and is_attn:
            continue
        y = [extractor(r) for r in runs]
        # Only plot points where data exists
        xs = [xi for xi, yi in zip(x, y) if yi is not None]
        ys = [yi for yi in y if yi is not None]
        if not xs:
            continue
        if time_extractor is not None:
            t_str = _avg_time(runs, time_extractor)
            legend_label = f"{label} (~{t_str})" if t_str else label
        else:
            legend_label = label
        ax.plot(
            xs, ys,
            label=legend_label,
            color=color,
            marker=marker,
            markersize=8 if is_attn else 6,
            linewidth=1.8,
            linestyle="--" if is_attn else "-",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training set size", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("Accuracy vs. training set size", fontsize=13)

    # Show exact n_train ticks instead of default log-scale powers of 10
    ax.set_xticks(x)
    ax.set_xticklabels([str(xi) for xi in x], fontsize=9)
    ax.xaxis.set_minor_locator(plt.NullLocator())

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0] - 0.02))
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout()

    if output_path is None:
        output_path = sweep_path.parent / "n_train_sweep_plot.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot accuracy vs. n_train from sweep_results.json")
    p.add_argument("sweep_json", type=Path,
                   help="Path to sweep_results.json produced by run_n_train_sweep")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output path for the figure (default: <sweep_dir>/n_train_sweep_plot.png)")
    p.add_argument("--no-cls", action="store_true",
                   help="Exclude the CLS token baseline from the plot")
    p.add_argument("--no-attn", action="store_true",
                   help="Exclude the attention pooling upper-bound line from the plot")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_sweep(args.sweep_json, args.output, exclude_cls=args.no_cls, exclude_attn=args.no_attn)
