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


def compute_mean_auroc(results: Dict) -> Dict[str, Tuple[List[int], List[float], List[float]]]:
    """Compute mean and std AUROC for each modality across train sizes and seeds.
    
    Args:
        results: Results dictionary with structure {train_size: {modality: [metrics, ...]}}
    
    Returns:
        Dictionary with structure {modality: ([train_sizes], [mean_aurocs], [std_aurocs])}
    """
    modalities = ["tabular", "image", "concat"]
    auroc_by_modality = {mod: ([], [], []) for mod in modalities}
    
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
                aurocs = [m.get("auroc") for m in metrics_list if m.get("auroc") is not None]
                if aurocs:
                    mean_auroc = np.mean(aurocs)
                    std_auroc = np.std(aurocs)
                    auroc_by_modality[modality][0].append(train_size if train_size != float('inf') else int(key))
                    auroc_by_modality[modality][1].append(mean_auroc)
                    auroc_by_modality[modality][2].append(std_auroc)
    
    return auroc_by_modality


def plot_results(
    auroc_data: Dict[str, Dict[str, Tuple[List[int], List[float], List[float]]]],
    configs: Dict[str, Dict],
    output_file: Path | str | None = None,
) -> None:
    """Plot mean AUROC vs training set size for multiple experiments with error bands.
    
    Args:
        auroc_data: Dictionary with structure {filename: {modality: ([sizes], [mean_aurocs], [std_aurocs])}}
        configs: Dictionary with structure {filename: config}
        output_file: Optional path to save the plot.
    """
    plt.figure(figsize=(14, 8))
    
    # Define colors for different files and markers for modalities
    colors_list = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    markers = {"tabular": "o", "image": "s", "concat": "^"}
    
    legend_elements = []
    
    for file_idx, (filename, auroc_by_modality) in enumerate(sorted(auroc_data.items())):
        color = colors_list[file_idx % len(colors_list)]
        classifier = configs[filename].get("classifier", filename)
        
        for modality, (train_sizes, mean_aurocs, std_aurocs) in auroc_by_modality.items():
            if train_sizes:  # Only plot if there are data points
                # Plot shaded region for standard deviation
                lower_bound = np.array(mean_aurocs) - np.array(std_aurocs)
                upper_bound = np.array(mean_aurocs) + np.array(std_aurocs)
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
                    mean_aurocs,
                    marker=markers[modality],
                    label=classifier,
                    color=color,
                    linestyle="-" if modality == "tabular" else ("--" if modality == "image" else "-."),
                    linewidth=2,
                    markersize=8,
                    alpha=0.7,
                )
    
    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel("Mean AUROC", fontsize=12)
    plt.title("AUROC vs Training Set Size - Multiple Experiments", fontsize=14)
    plt.legend(fontsize=9, loc="best")
    plt.grid(True, alpha=0.3)
    
    # Check if we should use log scale
    all_train_sizes = []
    for auroc_by_modality in auroc_data.values():
        for train_sizes, _, _ in auroc_by_modality.values():
            all_train_sizes.extend(train_sizes)
    
    if all_train_sizes and len(set(all_train_sizes)) > 1:
        plt.xscale("log")
    
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")
    
    plt.show()


def main(results_dir: Path | str, output_file: Path | str | None = None) -> None:
    """Main function to load and plot results from multiple JSON files.
    
    Args:
        results_dir: Path to the directory containing JSON results files.
        output_file: Optional path to save the plot.
    """
    results_data = load_results(results_dir)
    
    # Compute mean AUROC for each file
    auroc_data = {}
    configs = {}
    for filename, (config, results) in results_data.items():
        auroc_data[filename] = compute_mean_auroc(results)
        configs[filename] = config
    
    plot_results(auroc_data, configs, output_file)


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
        "--output",
        type=Path,
        default=None,
        help="Path to save the plot (e.g., plots/results.png). If not specified, plot is only displayed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.results_dir, args.output)
