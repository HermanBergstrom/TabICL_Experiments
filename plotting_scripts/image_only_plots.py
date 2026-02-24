"""Plot mean AUROC vs training set size for image-only experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import math


def load_results(
	results_path: Path,
) -> dict:
	with open(results_path, "r") as f:
		data = json.load(f)

	results = data["results"]
	train_sizes = sorted(int(k) for k in results.keys())

	output = {
		"train_sizes": train_sizes,
		"linear_probe": {"means": [], "sems": []},
		"tabpfn": {"means": [], "sems": []},
		"tabicl": {"means": [], "sems": []},
		"knn": {"means": [], "sems": []},
	}
	
	has_knn = False
	has_linear_cheat = False
	has_knn_cheat = False
	has_tabicl_v2 = False
	
	for size in train_sizes:
		key = str(size)
		linear_runs = results[key].get("linear_probe", [])
		tabpfn_runs = results[key].get("foundation_model", [])
		tabicl_runs = results[key].get("tabicl", [])

		linear_aurocs = [run["auroc"] for run in linear_runs if run.get("auroc") is not None]
		tabpfn_aurocs = [run["auroc"] for run in tabpfn_runs if run.get("auroc") is not None]
		tabicl_aurocs = [run["auroc"] for run in tabicl_runs if run.get("auroc") is not None]

		output["linear_probe"]["means"].append(float(np.mean(linear_aurocs)) if linear_aurocs else float("nan"))
		output["tabpfn"]["means"].append(float(np.mean(tabpfn_aurocs)) if tabpfn_aurocs else float("nan"))
		output["tabicl"]["means"].append(float(np.mean(tabicl_aurocs)) if tabicl_aurocs else float("nan"))
		output["linear_probe"]["sems"].append(
			float(np.std(linear_aurocs, ddof=1) / np.sqrt(len(linear_aurocs)))
			if len(linear_aurocs) > 1
			else 0.0
		)
		output["tabpfn"]["sems"].append(
			float(np.std(tabpfn_aurocs, ddof=1) / np.sqrt(len(tabpfn_aurocs)))
			if len(tabpfn_aurocs) > 1
			else 0.0
		)
		output["tabicl"]["sems"].append(
			float(np.std(tabicl_aurocs, ddof=1) / np.sqrt(len(tabicl_aurocs)))
			if len(tabicl_aurocs) > 1
			else 0.0
		)
		
		# Load TabICL v2 data if available
		if "tabicl_v2" in results[key]:
			has_tabicl_v2 = True
			if "tabicl_v2" not in output:
				output["tabicl_v2"] = {"means": [], "sems": []}
			tabicl_v2_runs = results[key]["tabicl_v2"]
			tabicl_v2_aurocs = [run["auroc"] for run in tabicl_v2_runs if run.get("auroc") is not None]
			output["tabicl_v2"]["means"].append(float(np.mean(tabicl_v2_aurocs)) if tabicl_v2_aurocs else float("nan"))
			output["tabicl_v2"]["sems"].append(
				float(np.std(tabicl_v2_aurocs, ddof=1) / np.sqrt(len(tabicl_v2_aurocs)))
				if len(tabicl_v2_aurocs) > 1
				else 0.0
			)
		
		# Load KNN data if available
		if "knn" in results[key]:
			has_knn = True
			knn_runs = results[key]["knn"]
			knn_aurocs = [run["auroc"] for run in knn_runs if run.get("auroc") is not None]
			output["knn"]["means"].append(float(np.mean(knn_aurocs)) if knn_aurocs else float("nan"))
			output["knn"]["sems"].append(
				float(np.std(knn_aurocs, ddof=1) / np.sqrt(len(knn_aurocs)))
				if len(knn_aurocs) > 1
				else 0.0
			)
		
		# Load cheat versions if available
		if "linear_probe_cheat" in results[key]:
			has_linear_cheat = True
			if "linear_probe_cheat" not in output:
				output["linear_probe_cheat"] = {"means": [], "sems": []}
			linear_cheat_runs = results[key]["linear_probe_cheat"]
			linear_cheat_aurocs = [run["auroc"] for run in linear_cheat_runs if run.get("auroc") is not None]
			output["linear_probe_cheat"]["means"].append(float(np.mean(linear_cheat_aurocs)) if linear_cheat_aurocs else float("nan"))
			output["linear_probe_cheat"]["sems"].append(
				float(np.std(linear_cheat_aurocs, ddof=1) / np.sqrt(len(linear_cheat_aurocs)))
				if len(linear_cheat_aurocs) > 1
				else 0.0
			)
		
		if "knn_cheat" in results[key]:
			has_knn_cheat = True
			if "knn_cheat" not in output:
				output["knn_cheat"] = {"means": [], "sems": []}
			knn_cheat_runs = results[key]["knn_cheat"]
			knn_cheat_aurocs = [run["auroc"] for run in knn_cheat_runs if run.get("auroc") is not None]
			output["knn_cheat"]["means"].append(float(np.mean(knn_cheat_aurocs)) if knn_cheat_aurocs else float("nan"))
			output["knn_cheat"]["sems"].append(
				float(np.std(knn_cheat_aurocs, ddof=1) / np.sqrt(len(knn_cheat_aurocs)))
				if len(knn_cheat_aurocs) > 1
				else 0.0
			)

	if not has_knn:
		output["knn"] = None
	if not has_linear_cheat:
		output["linear_probe_cheat"] = None
	if not has_knn_cheat:
		output["knn_cheat"] = None
	if not has_tabicl_v2:
		output["tabicl_v2"] = None
	
	return output


def plot_auroc_vs_train_size(
	data: dict,
	output_path: Path | None = None,
	ax: plt.Axes | None = None,
	title: str | None = None,
) -> None:
	train_sizes = data["train_sizes"]
	
	# Hard-coded color mapping
	color_map = {
		"linear_probe": "C0",  # blue
		"tabpfn": "C1",  # orange
		"tabicl": "C2",  # green
		"tabicl_v2": "C2",  # green
		"knn": "C3",  # red
	}
	
	# Create figure if no axis provided
	if ax is None:
		plt.figure(figsize=(7, 4))
		ax = plt.gca()
	
	# Plot linear probe (solid) and cheat version (dashed) in same color
	if data.get("linear_probe") is not None:
		color_lp = color_map["linear_probe"]
		line_lp = ax.plot(train_sizes, data["linear_probe"]["means"], marker="o", label="Linear Probing", linestyle="-", color=color_lp)
		ax.fill_between(
			train_sizes,
			np.array(data["linear_probe"]["means"]) - np.array(data["linear_probe"]["sems"]),
			np.array(data["linear_probe"]["means"]) + np.array(data["linear_probe"]["sems"]),
			alpha=0.2,
			color=color_lp,
		)
		
		if data.get("linear_probe_cheat") is not None:
			ax.plot(train_sizes, data["linear_probe_cheat"]["means"], marker="o", label="Linear Probing (cheat)", linestyle="--", color=color_lp)
			ax.fill_between(
				train_sizes,
				np.array(data["linear_probe_cheat"]["means"]) - np.array(data["linear_probe_cheat"]["sems"]),
				np.array(data["linear_probe_cheat"]["means"]) + np.array(data["linear_probe_cheat"]["sems"]),
				alpha=0.1,
				color=color_lp,
			)
	
	# Plot TabPFN
	if data.get("tabpfn") is not None:
		color_tabpfn = color_map["tabpfn"]
		line_tabpfn = ax.plot(train_sizes, data["tabpfn"]["means"], marker="o", label="TabPFN", linestyle="-", color=color_tabpfn)
		ax.fill_between(
			train_sizes,
			np.array(data["tabpfn"]["means"]) - np.array(data["tabpfn"]["sems"]),
			np.array(data["tabpfn"]["means"]) + np.array(data["tabpfn"]["sems"]),
			alpha=0.2,
			color=color_tabpfn,
		)
	
	# Plot TabICL
	if data.get("tabicl") is not None:
		color_tabicl = color_map["tabicl"]
		line_tabicl = ax.plot(train_sizes, data["tabicl"]["means"], marker="o", label="TabICL", linestyle="-", color=color_tabicl)
		ax.fill_between(
			train_sizes,
			np.array(data["tabicl"]["means"]) - np.array(data["tabicl"]["sems"]),
			np.array(data["tabicl"]["means"]) + np.array(data["tabicl"]["sems"]),
			alpha=0.2,
			color=color_tabicl,
		)
	
	# Plot TabICL v2
	if data.get("tabicl_v2") is not None:
		color_tabicl_v2 = color_map["tabicl_v2"]
		line_tabicl_v2 = ax.plot(train_sizes, data["tabicl_v2"]["means"], marker="s", label="TabICL v2", linestyle="-", color=color_tabicl_v2)
		ax.fill_between(
			train_sizes,
			np.array(data["tabicl_v2"]["means"]) - np.array(data["tabicl_v2"]["sems"]),
			np.array(data["tabicl_v2"]["means"]) + np.array(data["tabicl_v2"]["sems"]),
			alpha=0.2,
			color=color_tabicl_v2,
		)
	
	# Plot KNN (solid) and cheat version (dashed) in same color
	if data.get("knn") is not None:
		color_knn = color_map["knn"]
		line_knn = ax.plot(train_sizes, data["knn"]["means"], marker="o", label="KNN", linestyle="-", color=color_knn)
		ax.fill_between(
			train_sizes,
			np.array(data["knn"]["means"]) - np.array(data["knn"]["sems"]),
			np.array(data["knn"]["means"]) + np.array(data["knn"]["sems"]),
			alpha=0.2,
			color=color_knn,
		)
		
		if data.get("knn_cheat") is not None:
			ax.plot(train_sizes, data["knn_cheat"]["means"], marker="o", label="KNN (cheat)", linestyle="--", color=color_knn)
			ax.fill_between(
				train_sizes,
				np.array(data["knn_cheat"]["means"]) - np.array(data["knn_cheat"]["sems"]),
				np.array(data["knn_cheat"]["means"]) + np.array(data["knn_cheat"]["sems"]),
				alpha=0.1,
				color=color_knn,
			)
	
	ax.set_xlabel("Training set size")
	ax.set_ylabel("Mean AUROC")
	if title:
		ax.set_title(title)
	else:
		ax.set_title("Mean AUROC vs Training Set Size")
	ax.set_xscale("log")
	ax.grid(True, alpha=0.3)
	ax.legend()

	# Only save/show if we created our own figure (ax was None)
	if ax is None:
		if output_path is not None:
			output_path.parent.mkdir(parents=True, exist_ok=True)
			plt.savefig(output_path, dpi=200, bbox_inches="tight")
		else:
			plt.show()


def plot_directory(
	results_dir: Path,
	output_path: Path | None = None,
) -> None:
	"""Create a grid of subplots, one for each JSON file in the directory."""
	# Find all JSON files in directory
	json_files = sorted(results_dir.glob("*.json"))
	
	if not json_files:
		print(f"No JSON files found in {results_dir}")
		return
	
	print(f"Found {len(json_files)} JSON files")
	
	def _dataset_sort_key(path: Path) -> tuple[str, str]:
		try:
			with open(path, "r") as f:
				json_data = json.load(f)
			dataset_name = json_data.get("config", {}).get("dataset", "Unknown")
		except Exception:
			dataset_name = "Unknown"
		return (str(dataset_name), path.name)
	
	# Sort by dataset name, then filename
	json_files = sorted(json_files, key=_dataset_sort_key)
	
	# Determine grid dimensions
	n_files = len(json_files)
	ncols = min(2, n_files)  # Max 2 columns
	nrows = math.ceil(n_files / ncols)
	
	# Create figure with subplots
	fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
	
	# Handle case where axes is not an array
	if n_files == 1:
		axes = np.array([axes])
	else:
		axes = axes.flatten()
	
	# Plot each dataset
	for idx, json_file in enumerate(json_files):
		print(f"Processing {json_file.name}...")
		
		# Load results
		data = load_results(json_file)
		
		# Get dataset name from config
		with open(json_file, "r") as f:
			json_data = json.load(f)
		config = json_data.get("config", {})
		dataset_name = config.get("dataset", "Unknown")
		pca_dims = config.get("pca_dims")
		rp_dims = config.get("rp_dims")
		
		# Build title with dimensionality reduction info if present
		title = dataset_name
		if pca_dims is not None:
			title += f" (PCA={pca_dims})"
		if rp_dims is not None:
			title += f" (RP={rp_dims})"
		
		# Plot on corresponding subplot
		plot_auroc_vs_train_size(data, ax=axes[idx], title=title)
	
	# Hide unused subplots
	for idx in range(n_files, len(axes)):
		axes[idx].axis('off')
	
	plt.tight_layout()
	
	# Save to directory
	if output_path is None:
		output_path = results_dir / "combined_auroc_plot.pdf"
	
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=200, bbox_inches="tight")
	print(f"\nSaved combined plot to {output_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot mean AUROC lineplot.")
	parser.add_argument(
		"--results-path",
		type=Path,
		#default=Path("results/image_only_results/skin-cancer_linear_vs_fm_20260210_161244.json"),
		help="Path to results JSON file or directory containing JSON files.",
	)
	parser.add_argument(
		"--output-path",
		type=Path,
		default=None,
		#default=Path("plots/skin-cancer_linear_vs_fm_auroc.png"),
		help="Path to save the plot. For directories, defaults to combined_auroc_plot.pdf in the results directory. Use --output-path '' to show instead (single file only).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	
	# Check if results_path is a directory or file
	if args.results_path.is_dir():
		# Directory mode: create grid of subplots
		plot_directory(args.results_path, args.output_path)
	else:
		# Single file mode
		output_path = None if str(args.output_path) == "" else args.output_path
		data = load_results(args.results_path)
		plot_auroc_vs_train_size(data, output_path)


if __name__ == "__main__":
	main()
