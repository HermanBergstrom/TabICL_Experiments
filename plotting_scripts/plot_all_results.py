import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


LEGEND_FONT_SIZE = 13

MODEL_DISPLAY_NAMES = {
    "decision_tree": "Decision Tree",
    "logistic_regression_ur": "Logistic Reg. (unregularized)",
    "logistic_regression_l1": "Logistic Reg. (L1)",
    "small_mlp": "MLP",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "tabpfn": "TabPFN",
    "tabicl": "TabICL",
}

INJECTION_PREFIXES = {
    "noise": ["noise_"],
    "low_rank": ["low_rank_concat_", "low_rank_proj_"],
    "sparse_low_rank": ["sparse_low_rank_", "sparse_low_rank_proj_"],
    "low_signal": ["low_signal_", "low_signal_proj_"],
}

INJECTION_TITLES = {
    "noise": "Pure Noise Features",
    "low_rank": "Low-Rank Features",
    "sparse_low_rank": "Sparse Low-Rank Features",
    "low_signal": "Low-Signal Features",
}

REDUCTION_METHODS = {
    "no_reduction": "No Reduction",
    "pca": "PCA",
    "rp": "Random Projection",
}


def parse_dataset_name(dataset_name, method_hint=None):
    """
    Extract injection type and n_added from dataset name.
    For reduced datasets: pca_<injection>_<n> or rp_<injection>_<n>
    For original datasets: <injection>_<n>
    """
    # Check for reduction prefix
    if dataset_name.startswith("pca_"):
        remainder = dataset_name[4:]  # Remove "pca_"
        method = "pca"
    elif dataset_name.startswith("rp_"):
        remainder = dataset_name[3:]  # Remove "rp_"
        method = "rp"
    else:
        remainder = dataset_name
        method = "no_reduction"

    if remainder == "base":
        return method, "base", 0

    # Parse injection type and count
    for injection_type, prefixes in INJECTION_PREFIXES.items():
        for prefix in prefixes:
            if remainder.startswith(prefix):
                n_added = int(remainder.removeprefix(prefix))
                return method, injection_type, n_added

    return None, None, None


def discover_seed_files(data_dir):
    """
    Discover all seed-suffixed result files in the data directory.
    Returns a dict: {"no_reduction": [paths], "pca": [paths], "rp": [paths]}
    """
    data_dir = Path(data_dir)
    seed_files = {
        "no_reduction": [],
        "pca": [],
        "rp": [],
    }
    
    # Find all files ending with _seed*.csv
    for csv_file in sorted(data_dir.glob("*_seed*.csv")):
        filename = csv_file.name
        
        # Determine method from filename
        if filename.startswith("feature_quality_pca_"):
            seed_files["pca"].append(csv_file)
        elif filename.startswith("feature_quality_rp_"):
            seed_files["rp"].append(csv_file)
        elif filename.startswith("feature_quality_experiment_"):
            seed_files["no_reduction"].append(csv_file)
    
    return seed_files


def load_all_results(csv_normal, csv_pca, csv_rp):
    """
    Load results from all three CSV files and organize by method and injection type.
    Supports aggregation across multiple seed files.
    """
    all_injection_types = list(INJECTION_PREFIXES.keys())
    data = {
        method: {inj: defaultdict(list) for inj in all_injection_types}
        for method in ("no_reduction", "pca", "rp")
    }

    base_points = {
        "no_reduction": defaultdict(list),
        "pca": defaultdict(list),
        "rp": defaultdict(list),
    }

    # Aggregate data across seed files
    for csv_paths, method in [
        ([csv_normal] if isinstance(csv_normal, Path) else csv_normal, "no_reduction"),
        ([csv_pca] if isinstance(csv_pca, Path) else csv_pca, "pca"),
        ([csv_rp] if isinstance(csv_rp, Path) else csv_rp, "rp"),
    ]:
        for csv_path in csv_paths:
            if not csv_path.exists():
                continue

            with csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("error", "").strip():
                        continue

                    dataset_name = row["dataset"]
                    model_name = row["model"]
                    
                    # Parse dataset to extract injection type and n_added
                    parsed_method, injection_type, n_added = parse_dataset_name(dataset_name)
                    
                    if injection_type is None:
                        continue

                    target_method = parsed_method if parsed_method in data else method

                    accuracy = float(row["accuracy"])
                    if injection_type == "base":
                        base_points[target_method][model_name].append(accuracy)
                        continue

                    data[target_method][injection_type][model_name].append((n_added, accuracy))

    # Average base points across seeds
    for method in base_points:
        for model_name, accuracies in base_points[method].items():
            if accuracies:
                base_points[method][model_name] = mean(accuracies)

    # Add x=0 base points to every injection curve, and average across seeds
    for method in data:
        for injection_type in data[method]:
            for model_name in list(data[method][injection_type].keys()):
                points = data[method][injection_type][model_name]
                
                # Group points by n_added and average accuracies
                averaged_points = defaultdict(list)
                for n_added, accuracy in points:
                    averaged_points[n_added].append(accuracy)
                
                averaged_data = [(n, mean(accs)) for n, accs in averaged_points.items()]
                data[method][injection_type][model_name] = averaged_data
                
                # Add base point (x=0) if available
                if model_name in base_points[method]:
                    base_acc = base_points[method][model_name]
                    if not any(x == 0 for x, _ in averaged_data):
                        data[method][injection_type][model_name].append((0, base_acc))

    # Sort points by n_added for each model
    for method in data:
        for injection_type in data[method]:
            for model_name, points in data[method][injection_type].items():
                data[method][injection_type][model_name] = sorted(
                    points, key=lambda x: x[0]
                )

    return data


def plot_injection_type(
    ax,
    injection_type,
    model_points,
    n_informative_features,
    show_legend=True,
    show_injection_title=True,
):
    """Plot a single injection type."""
    raw_x_values = sorted({p[0] for points in model_points.values() for p in points})
    positive_x_values = [x for x in raw_x_values if x > 0]
    has_zero = 0 in raw_x_values

    # Plot x=0 at a small positive location so we can keep true log scaling.
    if positive_x_values:
        zero_plot_x = positive_x_values[0] / 3.0
    else:
        zero_plot_x = 1e-2

    for model_name, points in sorted(model_points.items()):
        x = [zero_plot_x if p[0] == 0 else p[0] for p in points]
        y = [p[1] for p in points]
        if not x:
            continue
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        ax.plot(x, y, marker="o", linewidth=2, label=display_name)

    # Reference line
    ax.axvline(
        n_informative_features,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"added = informative ({n_informative_features})",
    )

    if show_injection_title:
        ax.set_title(INJECTION_TITLES[injection_type])
    else:
        ax.set_title("")
    ax.set_xlabel("Number of Added Features")
    ax.set_ylabel("Accuracy")
    ax.set_xscale("log")
    if positive_x_values:
        xticks = ([zero_plot_x] if has_zero else []) + positive_x_values
        xticklabels = (["0"] if has_zero else []) + [str(x) for x in positive_x_values]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(left=zero_plot_x * 0.85, right=max(positive_x_values) * 1.1)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=max(2, min(len(labels), 4)),
            fontsize=LEGEND_FONT_SIZE,
            handlelength=2.0,
            handletextpad=0.5,
            columnspacing=1.2,
            borderpad=0.5,
            frameon=True,
            fancybox=True,
            framealpha=0.92,
            edgecolor="#cccccc",
        )


def make_per_method_plots(data, output_dir, n_informative_features):
    """Create one 3-panel figure for each reduction method."""
    injection_types = [
        inj for inj in INJECTION_PREFIXES
        if any(data[m][inj] for m in data)
    ]

    for method in ["no_reduction", "pca", "rp"]:
        n_inj = len(injection_types)
        fig, axes = plt.subplots(1, n_inj, figsize=(7 * n_inj, 5), sharey=True)
        if n_inj == 1:
            axes = [axes]

        for ax, injection_type in zip(axes, injection_types):
            plot_injection_type(
                ax,
                injection_type,
                data[method][injection_type],
                n_informative_features,
                show_legend=False,
            )

        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=max(2, min(len(labels), 6)),
            fontsize=LEGEND_FONT_SIZE,
            handlelength=2.2,
            handletextpad=0.5,
            columnspacing=1.4,
            borderpad=0.5,
            frameon=True,
            fancybox=True,
            framealpha=0.92,
            edgecolor="#cccccc",
            bbox_to_anchor=(0.5, 0.94),
        )

        method_title = REDUCTION_METHODS[method]
        fig.suptitle(
            f"Feature Injection Robustness ({method_title})",
            fontsize=14,
            y=0.975,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.91])
        
        output_path = output_dir / f"feature_quality_{method}_all_injections.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def make_joint_comparison_plot(data, output_dir, n_informative_features):
    """Create an NxM joint plot: rows=injections, cols=methods."""
    injection_types = [
        inj for inj in INJECTION_PREFIXES
        if any(data[m][inj] for m in data)
    ]
    methods = ["no_reduction", "pca", "rp"]

    n_rows, n_cols = len(injection_types), len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), sharey="row")
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[row] for row in axes]

    for row_idx, injection_type in enumerate(injection_types):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx][col_idx]
            
            plot_injection_type(
                ax,
                injection_type,
                data[method][injection_type],
                n_informative_features,
                show_legend=False,
                show_injection_title=False,
            )

            # Set column title on top row
            if row_idx == 0:
                ax.set_title(f"{REDUCTION_METHODS[method]}", fontsize=12, fontweight="bold")

            # Set row label on left side
            if col_idx == 0:
                ax.set_ylabel("Accuracy")
            else:
                ax.set_ylabel("")

    # Shared legend positioned between title and top subplot row.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(2, min(len(labels), 6)),
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.2,
        handletextpad=0.5,
        columnspacing=1.4,
        borderpad=0.5,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.95),
    )

    fig.suptitle(
        "Reduction Method Comparison: Feature Injection Robustness",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.08, 0.04, 1, 0.92])

    # Row headings aligned to the actual subplot row centers.
    fig.canvas.draw()
    for row_idx, injection_type in enumerate(injection_types):
        row_ax = axes[row_idx][0]
        bbox = row_ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2.0
        fig.text(
            0.02,
            y_center,
            INJECTION_TITLES[injection_type],
            va="center",
            ha="left",
            rotation=90,
            fontsize=12,
            fontweight="bold",
        )
    
    output_path = output_dir / "feature_quality_joint_comparison.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot feature-quality experiment results comparing all reduction methods."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results/feature_quality_experiments"),
        help="Directory containing result CSV files (will auto-discover seed files).",
    )
    parser.add_argument(
        "--normal-input",
        type=Path,
        help="(Optional) Path to normal (no reduction) results CSV. If not provided, seed files will be discovered automatically.",
    )
    parser.add_argument(
        "--pca-input",
        type=Path,
        help="(Optional) Path to PCA results CSV. If not provided, seed files will be discovered automatically.",
    )
    parser.add_argument(
        "--rp-input",
        type=Path,
        help="(Optional) Path to Random Projection results CSV. If not provided, seed files will be discovered automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/feature_quality_comparison"),
        help="Directory where plot images will be written.",
    )
    parser.add_argument(
        "--n-informative-features",
        type=int,
        default=200,
        help="Number of informative base features. A vertical reference line is drawn at this x-value.",
    )
    args = parser.parse_args()

    # Auto-discover seed files if custom inputs not provided
    if args.normal_input is None or args.pca_input is None or args.rp_input is None:
        seed_files = discover_seed_files(args.data_dir)
        normal_files = seed_files.get("no_reduction", [])
        pca_files = seed_files.get("pca", [])
        rp_files = seed_files.get("rp", [])
        
        # Fall back to aggregate files if no seed files found
        if not normal_files and args.normal_input is None:
            normal_files = [args.data_dir / "feature_quality_experiment_results.csv"]
        if not pca_files and args.pca_input is None:
            pca_files = [args.data_dir / "feature_quality_pca_results.csv"]
        if not rp_files and args.rp_input is None:
            rp_files = [args.data_dir / "feature_quality_rp_results.csv"]
        
        args.normal_input = normal_files
        args.pca_input = pca_files
        args.rp_input = rp_files
    else:
        args.normal_input = [args.normal_input]
        args.pca_input = [args.pca_input]
        args.rp_input = [args.rp_input]

    # Check that at least some files exist
    existing_files = [
        f for files in [args.normal_input, args.pca_input, args.rp_input]
        for f in files if f.exists()
    ]
    if not existing_files:
        raise FileNotFoundError(
            f"No results found in {args.data_dir}"
        )

    print(f"Loading results from {args.data_dir}")
    print(f"  no_reduction files: {len(args.normal_input)} found")
    print(f"  pca files: {len(args.pca_input)} found")
    print(f"  rp files: {len(args.rp_input)} found")
    print()

    data = load_all_results(args.normal_input, args.pca_input, args.rp_input)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create per-method plots
    make_per_method_plots(data, args.output_dir, args.n_informative_features)
    
    # Create joint comparison plot
    joint_path = make_joint_comparison_plot(
        data,
        args.output_dir,
        args.n_informative_features,
    )
    
    print(f"Saved plots to: {args.output_dir}")
    print(f"Joint comparison figure: {joint_path}")
    print()
    print("Generated files:")
    for m in ["no_reduction", "pca", "rp"]:
        print(f"  - feature_quality_{m}_all_injections.png")
    print(f"  - feature_quality_joint_comparison.png")


if __name__ == "__main__":
    main()
