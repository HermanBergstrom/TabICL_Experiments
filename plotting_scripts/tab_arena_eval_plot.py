"""Plot TabArena evaluation results.

Reads results.pkl files produced by tab_arena_eval.py and creates one line plot per dataset.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot TabArena evaluation results")
	parser.add_argument(
		"--results_dir",
		default="tabarena_results/stratified_subsampling",
		help="Base directory containing dataset subfolders with results.pkl",
	)
	parser.add_argument(
		"--output_dir",
		default="plots/tab_arena_eval",
		help="Directory to save plots",
	)
	parser.add_argument(
		"--include_k_fraction",
		action="store_true",
		help="Include k_fraction results if no n_samples_* entries exist",
	)
	return parser.parse_args()


def _collect_n_sample_keys(results: dict, include_k_fraction: bool) -> list[tuple[str, int | None]]:
	keys = []
	for key in results.keys():
		if key.startswith("n_samples_"):
			try:
				keys.append((key, int(key.split("n_samples_")[-1])))
			except ValueError:
				continue
	if not keys and include_k_fraction and "k_fraction" in results:
		keys.append(("k_fraction", None))
	return sorted(keys, key=lambda item: (item[1] is None, item[1] or 0))

def aggregate_auroc(
    aurocs: List[List[float]], ddof: int = 1
) -> Tuple[float, float]:
    """
    Aggregate AUROC scores across folds and repeats.
    """

    scores = np.asarray(aurocs)

    if scores.ndim != 2:
        raise ValueError("Input must be a list of lists: [repeat][fold]")

    # Step 1: average over folds (per repeat)
    repeat_means = scores.mean(axis=1)

    # Step 2: average over repeats
    mean_auroc = repeat_means.mean()

    # Uncertainty: variability across repeats
    std_auroc = repeat_means.std(ddof=ddof)

    return mean_auroc, std_auroc


def extract_dataset_results(results_path: str, include_k_fraction: bool) -> Tuple[str, dict] | None:
	"""Extract plot data from a results.pkl file."""
	with open(results_path, "rb") as f:
		results = pickle.load(f)

	metadata = results.get("metadata", {})
	dataset_name = metadata.get("dataset_name", os.path.basename(os.path.dirname(results_path)))

	n_sample_keys = _collect_n_sample_keys(results, include_k_fraction)
	if not n_sample_keys:
		return None

	methods = ["tab_icl", "tab_pfn"]
	x_values: list[int] = []
	y_values = {method: [] for method in methods}
	y_errors = {method: [] for method in methods}

	for key, n_samples in n_sample_keys:
		if n_samples is None:
			continue
		x_values.append(n_samples)

		for method in methods:
			roc_aucs = results[key][method]["roc_aucs"]
			mean_auc, std_auc = aggregate_auroc(roc_aucs)
			y_values[method].append(mean_auc)
			y_errors[method].append(std_auc)

	if not x_values:
		return None

	return dataset_name, {
		"x": x_values,
		"y": y_values,
		"errors": y_errors,
	}


def main() -> None:
	args = parse_args()

	if not os.path.exists(args.results_dir):
		raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

	dataset_dirs = [
		os.path.join(args.results_dir, name)
		for name in os.listdir(args.results_dir)
		if os.path.isdir(os.path.join(args.results_dir, name))
	]

	if not dataset_dirs:
		print(f"No dataset directories found in {args.results_dir}")
		return

	# Collect results from all datasets
	all_results = {}
	for dataset_dir in dataset_dirs:
		results_path = os.path.join(dataset_dir, "results.pkl")
		if not os.path.exists(results_path):
			continue
		result = extract_dataset_results(results_path, args.include_k_fraction)
		if result is not None:
			dataset_name, data = result
			all_results[dataset_name] = data

	if not all_results:
		print("No valid results found to plot.")
		return

	# Create subplots
	n_datasets = len(all_results)
	n_cols = 3
	n_rows = (n_datasets + n_cols - 1) // n_cols
	
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
	axes_flat = axes.flat if isinstance(axes, np.ndarray) else [axes]

	methods = ["tab_icl", "tab_pfn"]
	
	for idx, (dataset_name, data) in enumerate(sorted(all_results.items())):
		ax = axes_flat[idx]
		
		x_values = data["x"]
		y_values = data["y"]
		y_errors = data["errors"]
		
		for method in methods:
			y = y_values[method]
			err = y_errors[method]
			
			# Plot line with error shading
			ax.plot(x_values, y, marker="o", label=method, linewidth=2)
			ax.fill_between(
				x_values,
				np.array(y) - np.array(err),
				np.array(y) + np.array(err),
				alpha=0.2
			)
		
		ax.set_xscale("log")
		ax.set_xlabel("Training set size (log scale)")
		ax.set_ylabel("Mean AUROC")
		ax.set_title(dataset_name)
		ax.legend()
		ax.grid(True, alpha=0.3)
	
	# Hide unused subplots
	for idx in range(n_datasets, len(axes_flat)):
		axes_flat[idx].set_visible(False)
	
	plt.tight_layout()
	
	os.makedirs(args.output_dir, exist_ok=True)
	out_path = os.path.join(args.output_dir, "all_datasets.pdf")
	plt.savefig(out_path, dpi=200, bbox_inches="tight")
	print(f"Saved plot to {out_path}")
	plt.close()


if __name__ == "__main__":
	main()
