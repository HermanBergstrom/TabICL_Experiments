"""Plot results from multimodal training experiments.

This script loads all JSON result files from a directory and creates visualizations
that compare multiple experiments.

Example:
    python multimodal_plots.py results/
    python multimodal_plots.py results/ --output plots/comparison.png
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_results(results_dir: Path | str) -> Dict[str, Tuple[Dict, Dict]]:
    """Load results and config from all JSON files in a directory.
    
    Args:
        results_dir: Path to the directory containing JSON results files.
    
    Returns:
        Dictionary with structure {filename: (config, results)}
    """
    results_dir = Path(results_dir)
    
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {results_dir}")
    
    json_files = sorted(results_dir.glob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {results_dir}")
    
    results_data = {}
    
    for json_file in json_files:
        print(f"Loading results from {json_file.name}...")
        with open(json_file, "r") as f:
            data = json.load(f)
        
        config = data["config"]
        results = data["results"]
        
        results_data[json_file.stem] = (config, results)
    
    print(f"Loaded {len(results_data)} results file(s)")
    return results_data


def compute_metric_stats(
    results: Dict,
    metric: str = "auroc",
) -> Dict[str, Tuple[List[int], List[float], List[float]]]:
    """Compute mean and std of a specific metric for each modality across train sizes and seeds.
    
    Args:
        results: Results dictionary with structure {train_size: {modality: [metrics, ...]}}
        metric: Metric to extract (e.g., "auroc", "accuracy", "f1", "precision", "recall")
    
    Returns:
        Dictionary with structure {modality: ([train_sizes], [mean_values], [std_values])}
    """
    modalities = ["tabular", "image", "concat"]
    stats_by_modality = {mod: ([], [], []) for mod in modalities}
    
    # Sort train sizes numerically
    train_sizes = sorted(
        [int(k) if k != "all" else float('inf') for k in results.keys()]
    )
    
    for train_size in train_sizes:
        key = "all" if train_size == float('inf') else str(train_size)
        size_results = results[key]
        
        for modality in modalities:
            metrics_list = size_results.get(modality, [])
            
            if metrics_list:  # Only process if there are results
                values = [m.get(metric) for m in metrics_list if m.get(metric) is not None]
                if values:
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    stats_by_modality[modality][0].append(train_size if train_size != float('inf') else int(key))
                    stats_by_modality[modality][1].append(mean_value)
                    stats_by_modality[modality][2].append(std_value)
    
    return stats_by_modality


def plot_results(
    metric_data: Dict[str, Dict[str, Tuple[List[int], List[float], List[float]]]],
    configs: Dict[str, Dict],
    metric: str = "auroc",
    output_file: Path | str | None = None,
    show: bool = True,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
) -> None:
    """Plot mean metric vs training set size for multiple experiments with error bands.
    
    Args:
        metric_data: Dictionary with structure {filename: {modality: ([sizes], [mean_values], [std_values])}}
        configs: Dictionary with structure {filename: config}
        metric: The metric being plotted (for title)
        output_file: Optional path to save the plot.
        show: Whether to call plt.show() to display the plot.
        xlim: Optional tuple of (min, max) for x-axis limits.
        ylim: Optional tuple of (min, max) for y-axis limits.
    """
    plt.figure(figsize=(14, 8))
    
    # Define colors for different files and markers for modalities
    colors_list = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    markers = {"tabular": "o", "image": "s", "concat": "^"}
    
    legend_elements = []
    
    for file_idx, (filename, stats_by_modality) in enumerate(sorted(metric_data.items())):
        color = colors_list[file_idx % len(colors_list)]
        classifier = configs[filename].get("classifier", filename)
        
        for modality, (train_sizes, mean_values, std_values) in stats_by_modality.items():
            if train_sizes:  # Only plot if there are data points
                # Plot shaded region for standard deviation
                lower_bound = np.array(mean_values) - np.array(std_values)
                upper_bound = np.array(mean_values) + np.array(std_values)
                plt.fill_between(
                    train_sizes,
                    lower_bound,
                    upper_bound,
                    color=color,
                    alpha=0.1,
                )
                
                # Plot mean line
                line = plt.plot(
                    train_sizes,
                    mean_values,
                    marker=markers[modality],
                    label=classifier,
                    color=color,
                    linestyle="-" if modality == "tabular" else ("--" if modality == "image" else "-."),
                    linewidth=2,
                    markersize=8,
                    alpha=0.7,
                )
    
    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel(f"Mean {metric.upper()}", fontsize=12)
    plt.title(f"{metric.upper()} vs Training Set Size - Multiple Experiments", fontsize=14)
    plt.legend(fontsize=9, loc="best")
    plt.grid(True, alpha=0.3)
    
    # Check if we should use log scale
    all_train_sizes = []
    for stats_by_modality in metric_data.values():
        for train_sizes, _, _ in stats_by_modality.values():
            all_train_sizes.extend(train_sizes)
    
    if all_train_sizes and len(set(all_train_sizes)) > 1:
        plt.xscale("log")
    
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")
    
    if show:
        plt.show()


def plot_individual_seeds(
    results_data: Dict[str, Tuple[Dict, Dict]],
    metric: str = "auroc",
    output_file: Path | str | None = None,
    show: bool = True,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
) -> None:
    """Plot each seed as a separate subplot showing line plots for all experiments and modalities.
    
    Args:
        results_data: Dictionary with structure {filename: (config, results)}
        metric: The metric to plot (e.g., "auroc", "accuracy", "f1")
        output_file: Optional path to save the plot.
        show: Whether to call plt.show() to display the plot.
        xlim: Optional tuple of (min, max) for x-axis limits.
        ylim: Optional tuple of (min, max) for y-axis limits.
    """
    # Get data from first file to determine number of seeds
    first_filename = next(iter(results_data.keys()))
    first_config, first_results = results_data[first_filename]
    
    # Determine number of seeds
    train_size_keys = list(first_results.keys())
    first_size_results = first_results[train_size_keys[0]]
    modalities = ["tabular", "image", "concat"]
    num_seeds = 0
    for modality in modalities:
        num_seeds = len(first_size_results.get(modality, []))
        if num_seeds > 0:
            break
    
    if num_seeds == 0:
        print("Warning: No seed data found in results. Skipping individual seed plot.")
        return
    
    # Create subplots grid (one per seed, stacked vertically)
    fig, axes = plt.subplots(num_seeds, 1, figsize=(12, 4 * num_seeds))
    
    # Handle case where there's only one seed
    if num_seeds == 1:
        axes = np.array([axes])
    
    # Define colors for different experiments (classifiers)
    colors_list = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    linestyles = {"tabular": "-", "image": "--", "concat": "-."}
    
    # Sort train sizes numerically (same for all experiments)
    train_sizes_sorted = sorted(
        [(int(k) if k != "all" else float('inf'), k) for k in train_size_keys],
        key=lambda x: x[0]
    )
    
    # Plot each seed
    for seed_idx in range(num_seeds):
        ax = axes[seed_idx]
        
        # Loop through all experiments
        for exp_idx, (filename, (config, results)) in enumerate(sorted(results_data.items())):
            classifier = config.get("classifier", filename)
            color = colors_list[exp_idx % len(colors_list)]
            
            # Extract values for each modality across all train sizes
            for modality in modalities:
                values = []
                valid_train_sizes = []
                
                for _, size_key in train_sizes_sorted:
                    size_results = results[size_key]
                    metrics_list = size_results.get(modality, [])
                    
                    if metrics_list and seed_idx < len(metrics_list):
                        value = metrics_list[seed_idx].get(metric)
                        if value is not None:
                            values.append(value)
                            # Extract numeric train size
                            train_size_numeric = int(size_key) if size_key.isdigit() else float('inf')
                            if train_size_numeric == float('inf'):
                                # Find the max numeric train size for "all"
                                numeric_keys = [int(k) for k in train_size_keys if k.isdigit()]
                                train_size_numeric = max(numeric_keys) * 2 if numeric_keys else 1
                            valid_train_sizes.append(train_size_numeric)
                
                if values:
                    label = f"{classifier} ({modality})"
                    ax.plot(
                        valid_train_sizes,
                        values,
                        marker="o",
                        label=label,
                        color=color,
                        linestyle=linestyles[modality],
                        linewidth=2,
                        markersize=6,
                        alpha=0.7,
                    )
        
        ax.set_title(f"Seed {seed_idx}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Set Size", fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        
        # Set axis limits
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim([0, 1])
        
        if xlim is not None:
            ax.set_xlim(xlim)
        
        # Use log scale if there are multiple train sizes
        if len(train_sizes_sorted) > 1:
            ax.set_xscale("log")
    
    fig.suptitle(f"{metric.upper()} for Each Seed (Individual Line Plots)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Individual seed plot saved to: {output_file}")
    
    if show:
        plt.show()


def main(
    results_dir: Path | str,
    metric: str = "auroc",
    show_seeds: bool = False,
    output_file: Path | str | None = None,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
) -> None:
    """Main function to load and plot results from multiple JSON files.
    
    Args:
        results_dir: Path to the directory containing JSON results files.
        metric: Metric to plot (e.g., "auroc", "accuracy", "f1", "precision", "recall")
        show_seeds: Whether to create individual seed subplots
        output_file: Optional path to save the plot.
        xlim: Optional tuple of (min, max) for x-axis limits.
        ylim: Optional tuple of (min, max) for y-axis limits.
    """
    results_data = load_results(results_dir)
    
    # Plot individual seeds if requested
    if show_seeds:
        print(f"\nCreating individual seed plot for {metric}...")
        seed_output = None
        if output_file:
            output_path = Path(output_file)
            seed_output = output_path.parent / f"{output_path.stem}_seeds{output_path.suffix}"
        plot_individual_seeds(results_data, metric=metric, output_file=seed_output, show=False, xlim=xlim, ylim=ylim)
    
    # Compute mean metric for each file
    metric_data = {}
    configs = {}
    for filename, (config, results) in results_data.items():
        metric_data[filename] = compute_metric_stats(results, metric=metric)
        configs[filename] = config
    
    # Plot aggregated results
    print(f"\nCreating aggregated plot for {metric}...")
    plot_results(metric_data, configs, metric=metric, output_file=output_file, show=not show_seeds, xlim=xlim, ylim=ylim)
    
    # Show all plots at once if seeds were created
    if show_seeds:
        plt.show()
    
    # Show all plots at once if seeds were created
    if show_seeds:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot multimodal training results from all JSON files in a directory."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to the directory containing JSON results files",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["auroc", "accuracy", "precision", "recall", "f1"],
        default="auroc",
        help="Metric to plot (default: auroc)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the plot (e.g., plots/results.png). If not specified, plot is only displayed.",
    )
    parser.add_argument(
        "--show-seeds",
        action="store_true",
        help="Create additional plot showing each individual seed as a subplot",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="X-axis limits (e.g., --xlim 10 1000)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Y-axis limits (e.g., --ylim 0.5 1.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    xlim = tuple(args.xlim) if args.xlim else None
    ylim = tuple(args.ylim) if args.ylim else None
    main(args.results_dir, metric=args.metric, show_seeds=args.show_seeds, output_file=args.output, xlim=xlim, ylim=ylim)
