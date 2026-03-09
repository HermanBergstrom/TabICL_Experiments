import argparse
import math
import os
import pickle
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot per-dataset AUROC histograms for v2 attention results")
    parser.add_argument(
        "--use-std-error",
        action="store_true",
        help="Use standard error instead of standard deviation for error bars",
    )
    parser.add_argument(
        "--stratified-subsampling",
        action="store_true",
        help="Read from stratified_subsampling results folder",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root directory",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively",
    )
    return parser.parse_args()


def aggregate_auroc(aurocs: List[List[float]], ddof: int = 1) -> Tuple[float, float]:
    """Aggregate AUROC scores across folds and repeats."""

    scores = np.asarray(aurocs)
    if scores.ndim != 2:
        raise ValueError("Input must be a list of lists: [repeat][fold]")

    repeat_means = scores.mean(axis=1)
    mean_auroc = repeat_means.mean()
    std_auroc = repeat_means.std(ddof=ddof)
    return float(mean_auroc), float(std_auroc)


def _flatten_refit_stats(refit_nested):
    flattened = []
    if not isinstance(refit_nested, list):
        return flattened
    for repeat_entries in refit_nested:
        if not isinstance(repeat_entries, list):
            continue
        for item in repeat_entries:
            if isinstance(item, dict):
                flattened.append(item)
    return flattened


def _extract_mean_reduced_ratio(results_dict) -> Optional[float]:
    k_fraction = results_dict.get("k_fraction", {})
    refit_nested = k_fraction.get("refit_on_hit_subset", [])
    refit_stats = _flatten_refit_stats(refit_nested)

    if not refit_stats:
        return None

    ratios = []
    for stat in refit_stats:
        ratio = stat.get("compression_ratio")
        if ratio is None:
            continue
        if np.isfinite(ratio):
            ratios.append(float(ratio))

    if not ratios:
        return None
    return float(np.mean(ratios))


def _extract_kept_total(results_dict) -> Optional[Tuple[int, int]]:
    """Return mean (kept, total) counts across refit splits."""
    k_fraction = results_dict.get("k_fraction", {}) if isinstance(results_dict, dict) else {}
    refit_nested = k_fraction.get("refit_on_hit_subset", [])
    refit_stats = _flatten_refit_stats(refit_nested)
    if not refit_stats:
        return None

    kept_values = []
    total_values = []
    for stat in refit_stats:
        kept = stat.get("kept_samples")
        total = stat.get("total_samples")
        if kept is None or total is None:
            continue
        kept_values.append(float(kept))
        total_values.append(float(total))

    if not kept_values or not total_values:
        return None
    mean_kept = int(round(float(np.mean(kept_values))))
    mean_total = int(round(float(np.mean(total_values))))
    if mean_total <= 0:
        return None
    return mean_kept, mean_total


def _is_refit_on_hit_subset(results_dict) -> bool:
    """Infer refit mode from stored results, without relying on CLI flags."""
    k_fraction = results_dict.get("k_fraction", {}) if isinstance(results_dict, dict) else {}
    refit_nested = k_fraction.get("refit_on_hit_subset", [])
    refit_stats = _flatten_refit_stats(refit_nested)
    return len(refit_stats) > 0


def _safe_aggregate(metric_nested, ddof=1):
    if not isinstance(metric_nested, list) or len(metric_nested) == 0:
        return np.nan, np.nan, 0
    if not isinstance(metric_nested[0], list) or len(metric_nested[0]) == 0:
        return np.nan, np.nan, 0

    mean_val, std_val = aggregate_auroc(metric_nested, ddof=ddof)
    n_repeats = len(metric_nested)
    return mean_val, std_val, n_repeats


def _load_results(results_base_dir):
    all_results = {}
    if not os.path.isdir(results_base_dir):
        return all_results

    for dataset_name in sorted(os.listdir(results_base_dir)):
        result_path = os.path.join(results_base_dir, dataset_name, "results.pkl")
        if not os.path.exists(result_path):
            continue
        with open(result_path, "rb") as f:
            all_results[dataset_name] = pickle.load(f)
    return all_results


def _dataset_subtitle(results_dict, refit_on_hit_subset):
    metadata = results_dict.get("metadata", {}) if isinstance(results_dict, dict) else {}
    qualities = metadata.get("dataset_qualities", {}) if isinstance(metadata, dict) else {}

    majority_size = qualities.get("majority_class_size")
    minority_size = qualities.get("minority_class_size")

    parts = []
    if majority_size is not None and minority_size is not None:
        denom = majority_size + minority_size
        if denom > 0:
            majority_fraction = majority_size / denom
            minority_fraction = minority_size / denom
            parts.append(
                f"balance: maj={majority_fraction * 100:.1f}%, min={minority_fraction * 100:.1f}%"
            )

    folds = metadata.get("folds")
    repeats = metadata.get("tabarena_repeats")
    if folds is not None:
        parts.append(f"folds={folds}")
    if repeats is not None:
        parts.append(f"repeats={repeats}")

    if refit_on_hit_subset:
        kept_total = _extract_kept_total(results_dict)
        if kept_total is not None:
            kept, total = kept_total
            parts.append(f"reduced kept={kept}/{total}")

    return " | ".join(parts)


def main():
    args = parse_args()

    results_rel = (
        "results/v2_attention_results/stratified_subsampling"
        if args.stratified_subsampling
        else "results/v2_attention_results/standard_subsampling"
    )
    results_base_dir = os.path.join(args.root_dir, results_rel)
    all_results = _load_results(results_base_dir)

    if not all_results:
        raise FileNotFoundError(f"No results.pkl files found under: {results_base_dir}")

    num_datasets = len(all_results)
    num_cols = 3
    num_rows = math.ceil(num_datasets / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5.5 * num_rows))
    axes = np.atleast_1d(axes).flatten()

    refit_mode_flags = []

    for idx, (dataset_name, results_dict) in enumerate(all_results.items()):
        ax = axes[idx]
        ddof = 1
        refit_on_hit_subset = _is_refit_on_hit_subset(results_dict)
        refit_mode_flags.append(refit_on_hit_subset)

        full_mean, full_std, full_n = _safe_aggregate(results_dict.get("full_model", {}).get("roc_aucs", []), ddof=ddof)
        k_mean, k_std, k_n = _safe_aggregate(results_dict.get("k_fraction", {}).get("roc_aucs", []), ddof=ddof)
        rand_mean, rand_std, rand_n = _safe_aggregate(
            results_dict.get("k_fraction_random_baseline", {}).get("roc_aucs", []), ddof=ddof
        )

        means = [full_mean, k_mean, rand_mean]
        stds = [full_std, k_std, rand_std]
        ns = [full_n, k_n, rand_n]

        if args.use_std_error:
            errors = [
                (std / np.sqrt(n) if (n > 0 and np.isfinite(std)) else np.nan)
                for std, n in zip(stds, ns)
            ]
        else:
            errors = stds

        kept_total = _extract_kept_total(results_dict) if refit_on_hit_subset else None
        k_label = "reduced" if refit_on_hit_subset else "k_fraction"
        if kept_total is not None:
            kept, total = kept_total
            k_label = f"{k_label}\n({kept}/{total} kept)"

        labels = ["full", k_label, "random_baseline"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        x_pos = np.arange(3)

        bars = ax.bar(
            x_pos,
            means,
            yerr=errors,
            capsize=5,
            alpha=0.88,
            color=colors,
            edgecolor="black",
            linewidth=1.0,
        )

        if np.isfinite(full_mean):
            ax.axhline(full_mean, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.6)

        # Highlight deltas against the full model for k_fraction/reduced and random baseline.
        for i in [1, 2]:
            if np.isfinite(means[i]) and np.isfinite(full_mean):
                delta = means[i] - full_mean
                delta_color = "#0a7f2e" if delta >= 0 else "#b22222"
                y_base = means[i] + (errors[i] if np.isfinite(errors[i]) else 0.0)
                ax.text(
                    x_pos[i],
                    y_base + 0.032,
                    f"Δ {delta:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=delta_color,
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
                )

        for i, bar in enumerate(bars):
            mean_val = means[i]
            if np.isfinite(mean_val):
                y_base = bar.get_height() + (errors[i] if np.isfinite(errors[i]) else 0.0)
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_base + 0.006,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
                )

        subtitle = _dataset_subtitle(results_dict, refit_on_hit_subset)
        wrapped_dataset_name = textwrap.fill(dataset_name, width=36, break_long_words=False)
        ax.set_title(wrapped_dataset_name, fontsize=11, fontweight="bold", pad=18)
        if subtitle:
            ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("AUROC", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

        finite_means = [m for m in means if np.isfinite(m)]
        finite_errs = [e for e in errors if np.isfinite(e)]
        if finite_means:
            ymax = max(finite_means) + (max(finite_errs) if finite_errs else 0.0) + 0.08
            ymin = 0.5
            ax.set_ylim(ymin, min(1.02, ymax))

    for idx in range(num_datasets, len(axes)):
        fig.delaxes(axes[idx])

    summary = "std error" if args.use_std_error else "std"
    mode = "stratified" if args.stratified_subsampling else "standard"

    all_refit = all(refit_mode_flags)
    any_refit = any(refit_mode_flags)
    if all_refit:
        title = f"Per-dataset AUROC comparison ({mode}, reduced vs full, error={summary})"
    elif any_refit:
        title = f"Per-dataset AUROC comparison ({mode}, mixed reduced/k_fraction vs full, error={summary})"
    else:
        title = f"Per-dataset AUROC comparison ({mode}, k_fraction vs full, error={summary})"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    output_name = (
        "v2_attention_histograms_stratified_refit.pdf"
        if args.stratified_subsampling and all_refit
        else "v2_attention_histograms_standard_refit.pdf"
        if (not args.stratified_subsampling and all_refit)
        else "v2_attention_histograms_stratified.pdf"
        if args.stratified_subsampling
        else "v2_attention_histograms_standard.pdf"
    )
    output_path = os.path.join(args.root_dir, output_name)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()