import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


LEGEND_FONT_SIZE = 13

INJECTION_PREFIXES = {
    "noise": "noise_",
    "low_rank": "low_rank_concat_",
    "low_signal": "low_signal_",
}

INJECTION_TITLES = {
    "noise": "Pure Noise Features",
    "low_rank": "Low-Rank Features",
    "low_signal": "Low-Signal Features",
}

REDUCTION_METHODS = {
    "no_reduction": "No Reduction",
    "pca": "PCA",
    "rp": "Random Projection",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_variant_name(variant_name):
    """
    Extract (reduction_method, injection_type, n_added) from a variant string.

    Handles both synthetic-style dataset names and real-data variant strings:
      pca_noise_50          -> ("pca",          "noise",    50)
      rp_low_rank_concat_50 -> ("rp",           "low_rank", 50)
      low_signal_200        -> ("no_reduction", "low_signal", 200)
      base / pca_base / ...  -> (method, "base", 0)
    """
    name = variant_name.strip()

    # Strip reduction prefix
    if name.startswith("pca_"):
        method, remainder = "pca", name[4:]
    elif name.startswith("rp_"):
        method, remainder = "rp", name[3:]
    else:
        method, remainder = "no_reduction", name

    if remainder == "base":
        return method, "base", 0

    for injection_type, prefix in INJECTION_PREFIXES.items():
        if remainder.startswith(prefix):
            try:
                n_added = int(remainder[len(prefix):])
            except ValueError:
                continue
            return method, injection_type, n_added

    return None, None, None


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_seed_files(data_dir):
    """
    Discover all seed-suffixed result files in the data directory (synthetic mode).
    Returns a dict: {"no_reduction": [paths], "pca": [paths], "rp": [paths]}
    """
    data_dir = Path(data_dir)
    seed_files = {"no_reduction": [], "pca": [], "rp": []}

    for csv_file in sorted(data_dir.glob("*_seed*.csv")):
        filename = csv_file.name
        if filename.startswith("feature_quality_pca_"):
            seed_files["pca"].append(csv_file)
        elif filename.startswith("feature_quality_rp_"):
            seed_files["rp"].append(csv_file)
        elif filename.startswith("feature_quality_experiment_"):
            seed_files["no_reduction"].append(csv_file)

    return seed_files


def discover_task_csvs(task_dir):
    """
    Walk *task_dir* recursively and return every results.csv found as a flat list.
    """
    task_dir = Path(task_dir)
    direct   = task_dir / "results.csv"
    if direct.exists():
        return [direct]
    return sorted(task_dir.rglob("results.csv"))


# ---------------------------------------------------------------------------
# Data loading — synthetic (multi-file) mode
# ---------------------------------------------------------------------------

def load_all_results(csv_normal, csv_pca, csv_rp):
    """
    Load results from up to three CSV file lists (synthetic experiment format).
    The 'dataset' column encodes the injection type and method.
    """
    data = {m: {inj: defaultdict(list) for inj in INJECTION_PREFIXES}
            for m in REDUCTION_METHODS}
    base_points = {m: defaultdict(list) for m in REDUCTION_METHODS}

    file_groups = [
        (csv_normal if isinstance(csv_normal, list) else [csv_normal], "no_reduction"),
        (csv_pca    if isinstance(csv_pca,    list) else [csv_pca],    "pca"),
        (csv_rp     if isinstance(csv_rp,     list) else [csv_rp],     "rp"),
    ]

    for csv_paths, default_method in file_groups:
        for csv_path in csv_paths:
            if not Path(csv_path).exists():
                continue
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("error", "").strip():
                        continue
                    parsed_method, injection_type, n_added = parse_variant_name(
                        row["dataset"]
                    )
                    if injection_type is None:
                        continue
                    target_method = parsed_method if parsed_method in data else default_method
                    accuracy = float(row["accuracy"])
                    model_name = row["model"]
                    if injection_type == "base":
                        base_points[target_method][model_name].append(accuracy)
                    else:
                        data[target_method][injection_type][model_name].append(
                            (n_added, accuracy)
                        )

    return _finalise_data(data, base_points)



def load_real_dataset_results(csv_path):
    """
    Load a single results.csv from a real-dataset experiment.

    Expected columns: variant, model, accuracy, n_original_features,
                      normalized_effective_rank, dataset_name  (and others).

    Returns:
        data           – same nested dict as load_all_results
        metadata       – dict with keys: dataset_name, n_original_features,
                         normalized_effective_rank
    """
    data = {m: {inj: defaultdict(list) for inj in INJECTION_PREFIXES}
            for m in REDUCTION_METHODS}
    base_points = {m: defaultdict(list) for m in REDUCTION_METHODS}

    meta_dataset_name = ""
    meta_n_original   = None
    meta_eff_rank     = None

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error", "").strip():
                continue

            # Harvest metadata from the first valid row
            if not meta_dataset_name and row.get("dataset_name"):
                meta_dataset_name = row["dataset_name"].strip()
            if meta_n_original is None and row.get("n_original_features"):
                try:
                    meta_n_original = int(float(row["n_original_features"]))
                except ValueError:
                    pass
            if meta_eff_rank is None and row.get("normalized_effective_rank"):
                try:
                    meta_eff_rank = float(row["normalized_effective_rank"])
                except ValueError:
                    pass

            variant = row.get("variant", "").strip()
            if not variant:
                continue

            parsed_method, injection_type, n_added = parse_variant_name(variant)
            if injection_type is None:
                continue

            accuracy   = float(row["accuracy"])
            model_name = row["model"].strip()

            if injection_type == "base":
                base_points[parsed_method][model_name].append(accuracy)
            else:
                data[parsed_method][injection_type][model_name].append(
                    (n_added, accuracy)
                )

    finalised = _finalise_data(data, base_points)
    metadata  = {
        "dataset_name":              meta_dataset_name or Path(csv_path).parent.name,
        "n_original_features":       meta_n_original or 0,
        "normalized_effective_rank": meta_eff_rank,
    }
    return finalised, metadata


def _finalise_data(data, base_points):
    """Average across seeds, inject x=0 base points, and sort by n_added."""
    # Average base points
    for method in base_points:
        for model_name, accs in base_points[method].items():
            if accs:
                base_points[method][model_name] = mean(accs)

    for method in data:
        for injection_type in data[method]:
            for model_name in list(data[method][injection_type]):
                points = data[method][injection_type][model_name]

                # Average across seeds per n_added
                grouped = defaultdict(list)
                for n, acc in points:
                    grouped[n].append(acc)
                averaged = [(n, mean(accs)) for n, accs in grouped.items()]
                data[method][injection_type][model_name] = averaged

                # Prepend base point at x=0
                if model_name in base_points[method]:
                    base_acc = base_points[method][model_name]
                    if not any(x == 0 for x, _ in averaged):
                        data[method][injection_type][model_name].append(
                            (0, base_acc)
                        )

    # Sort by n_added
    for method in data:
        for injection_type in data[method]:
            for model_name in data[method][injection_type]:
                data[method][injection_type][model_name].sort(key=lambda x: x[0])

    return data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_injection_type(
    ax,
    injection_type,
    model_points,
    n_reference_features,
    reference_label=None,
    show_legend=True,
    show_injection_title=True,
):
    """Plot a single injection type on *ax*."""
    raw_x_values      = sorted({p[0] for pts in model_points.values() for p in pts})
    positive_x_values = [x for x in raw_x_values if x > 0]
    has_zero          = 0 in raw_x_values

    zero_plot_x = (positive_x_values[0] / 3.0) if positive_x_values else 1e-2

    for model_name, points in sorted(model_points.items()):
        x = [zero_plot_x if p[0] == 0 else p[0] for p in points]
        y = [p[1] for p in points]
        if x:
            ax.plot(x, y, marker="o", linewidth=2, label=model_name)

    ref_label = reference_label or f"added = original ({n_reference_features})"
    ax.axvline(
        n_reference_features,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=ref_label,
    )

    if show_injection_title:
        ax.set_title(INJECTION_TITLES[injection_type])
    else:
        ax.set_title("")

    ax.set_xlabel("Number of Added Features")
    ax.set_ylabel("Accuracy")
    ax.set_xscale("log")

    if positive_x_values:
        xticks      = ([zero_plot_x] if has_zero else []) + positive_x_values
        xticklabels = (["0"]         if has_zero else []) + [str(x) for x in positive_x_values]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(left=zero_plot_x * 0.85, right=max(positive_x_values) * 1.1)

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=max(2, min(len(labels), 4)),
            fontsize=LEGEND_FONT_SIZE,
            handlelength=2.0, handletextpad=0.5, columnspacing=1.2,
            borderpad=0.5, frameon=True, fancybox=True,
            framealpha=0.92, edgecolor="#cccccc",
        )


def _dataset_subtitle(metadata):
    """Build a compact subtitle string from real-dataset metadata."""
    if not metadata:
        return ""
    parts = []
    if metadata.get("dataset_name"):
        parts.append(metadata["dataset_name"])
    if metadata.get("n_original_features"):
        parts.append(f"original features = {metadata['n_original_features']}")
    if metadata.get("normalized_effective_rank") is not None:
        parts.append(f"norm. eff. rank = {metadata['normalized_effective_rank']:.3f}")
    return "  |  ".join(parts)
def make_per_method_plots(data, output_dir, n_informative_features, metadata=None):
    """Create one 3-panel figure for each reduction method."""
    injection_types = ["noise", "low_rank", "low_signal"]
    subtitle        = _dataset_subtitle(metadata)

    for method in ["no_reduction", "pca", "rp"]:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

        for ax, injection_type in zip(axes, injection_types):
            plot_injection_type(
                ax,
                injection_type,
                data[method][injection_type],
                n_informative_features,
                show_legend=False,
            )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=max(2, min(len(labels), 6)),
            fontsize=LEGEND_FONT_SIZE,
            handlelength=2.2, handletextpad=0.5, columnspacing=1.4,
            borderpad=0.5, frameon=True, fancybox=True,
            framealpha=0.92, edgecolor="#cccccc",
            bbox_to_anchor=(0.5, 0.94),
        )

        method_title = REDUCTION_METHODS[method]
        main_title   = f"Feature Injection Robustness ({method_title})"
        if subtitle:
            main_title = f"{main_title}\n{subtitle}"

        fig.suptitle(main_title, fontsize=13, y=0.985)
        fig.tight_layout(rect=[0, 0.04, 1, 0.91])

        output_path = output_dir / f"feature_quality_{method}_all_injections.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def make_joint_comparison_plot(data, output_dir, n_informative_features, metadata=None):
    """Create a 3×3 joint plot: rows = injection types, cols = reduction methods."""
    injection_types = ["noise", "low_rank", "low_signal"]
    methods         = ["no_reduction", "pca", "rp"]
    subtitle        = _dataset_subtitle(metadata)

    fig, axes = plt.subplots(3, 3, figsize=(20, 15), sharey="row")

    for row_idx, injection_type in enumerate(injection_types):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            plot_injection_type(
                ax,
                injection_type,
                data[method][injection_type],
                n_informative_features,
                show_legend=False,
                show_injection_title=False,
            )
            if row_idx == 0:
                ax.set_title(REDUCTION_METHODS[method], fontsize=12, fontweight="bold")
            if col_idx != 0:
                ax.set_ylabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=max(2, min(len(labels), 6)),
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.2, handletextpad=0.5, columnspacing=1.4,
        borderpad=0.5, frameon=True, fancybox=True,
        framealpha=0.92, edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.95),
    )

    main_title = "Reduction Method Comparison: Feature Injection Robustness"
    if subtitle:
        main_title = f"{main_title}\n{subtitle}"

    fig.suptitle(main_title, fontsize=13, fontweight="bold", y=0.988)
    fig.tight_layout(rect=[0.08, 0.04, 1, 0.92])

    # Row headings (injection type labels on the left)
    fig.canvas.draw()
    for row_idx, injection_type in enumerate(injection_types):
        row_ax = axes[row_idx, 0]
        bbox   = row_ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2.0
        fig.text(
            0.02, y_center,
            INJECTION_TITLES[injection_type],
            va="center", ha="left", rotation=90,
            fontsize=12, fontweight="bold",
        )

    output_path = output_dir / "feature_quality_joint_comparison.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def run_synthetic_mode(args):
    """Original behaviour: read from separate no-reduction / pca / rp CSVs."""
    if args.normal_input is None or args.pca_input is None or args.rp_input is None:
        seed_files   = discover_seed_files(args.data_dir)
        normal_files = seed_files.get("no_reduction", [])
        pca_files    = seed_files.get("pca", [])
        rp_files     = seed_files.get("rp", [])

        if not normal_files and args.normal_input is None:
            normal_files = [args.data_dir / "feature_quality_experiment_results.csv"]
        if not pca_files and args.pca_input is None:
            pca_files = [args.data_dir / "feature_quality_pca_results.csv"]
        if not rp_files and args.rp_input is None:
            rp_files = [args.data_dir / "feature_quality_rp_results.csv"]

        args.normal_input = normal_files
        args.pca_input    = pca_files
        args.rp_input     = rp_files
    else:
        args.normal_input = [args.normal_input]
        args.pca_input    = [args.pca_input]
        args.rp_input     = [args.rp_input]

    existing = [
        f
        for files in [args.normal_input, args.pca_input, args.rp_input]
        for f in files
        if Path(f).exists()
    ]
    if not existing:
        raise FileNotFoundError(f"No results found in {args.data_dir}")

    print(f"Loading results from {args.data_dir}")
    print(f"  no_reduction files : {len(args.normal_input)}")
    print(f"  pca files          : {len(args.pca_input)}")
    print(f"  rp files           : {len(args.rp_input)}")

    data = load_all_results(args.normal_input, args.pca_input, args.rp_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    make_per_method_plots(data, args.output_dir, args.n_informative_features)
    joint_path = make_joint_comparison_plot(
        data, args.output_dir, args.n_informative_features
    )

    print(f"\nSaved plots to: {args.output_dir}")
    print(f"Joint figure  : {joint_path}")


def run_task_mode(args):
    """
    Walk a task directory, find every results.csv, and produce one set of
    plots per file, stored under the dataset name read from the CSV itself.
    """
    csv_paths = discover_task_csvs(args.task_dir)
    if not csv_paths:
        raise FileNotFoundError(
            f"No results.csv files found under {args.task_dir}"
        )

    print(f"Found {len(csv_paths)} dataset(s) under {args.task_dir}\n")

    for csv_path in csv_paths:
        data, metadata = load_real_dataset_results(csv_path)

        # Use the dataset name from the CSV as the output directory label,
        # falling back to the parent directory name if unavailable.
        dataset_name = metadata.get("dataset_name") or csv_path.parent.name
        safe_label   = dataset_name.replace(" ", "_").replace("/", "_")

        print(f"  {dataset_name}  ({csv_path})")

        n_ref = metadata["n_original_features"] or args.n_informative_features
        if n_ref == 0:
            print(f"    WARNING: n_original_features = 0, falling back to "
                  f"--n-informative-features={args.n_informative_features}")
            n_ref = args.n_informative_features

        out_dir = args.output_dir / safe_label
        out_dir.mkdir(parents=True, exist_ok=True)

        make_per_method_plots(data, out_dir, n_ref, metadata=metadata)
        joint_path = make_joint_comparison_plot(
            data, out_dir, n_ref, metadata=metadata
        )
        print(f"    -> {joint_path}\n")

    print(f"All plots saved under: {args.output_dir}")
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot feature-quality experiment results comparing all reduction methods.\n\n"
            "Two modes:\n"
            "  Synthetic mode  (default): reads separate no-reduction / pca / rp CSVs\n"
            "                             from --data-dir.\n"
            "  Task mode (--task-dir)   : walks a task directory tree, reads every\n"
            "                             results.csv, and produces one plot set per\n"
            "                             dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- task-directory mode ----
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=None,
        help=(
            "Root of a task directory tree, e.g. 'experiments/task_363618'. "
            "Each sub-directory that contains a results.csv is treated as one dataset. "
            "When this flag is given, --data-dir / --normal-input / --pca-input / "
            "--rp-input are ignored."
        ),
    )

    # ---- synthetic mode ----
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results/feature_quality_experiments"),
        help="Directory containing synthetic result CSV files (auto-discovers seed files).",
    )
    parser.add_argument(
        "--normal-input",
        type=Path,
        help="(Optional) Path to no-reduction results CSV.",
    )
    parser.add_argument(
        "--pca-input",
        type=Path,
        help="(Optional) Path to PCA results CSV.",
    )
    parser.add_argument(
        "--rp-input",
        type=Path,
        help="(Optional) Path to Random Projection results CSV.",
    )

    # ---- shared ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/feature_quality_comparison"),
        help="Root directory where plot images will be written.",
    )
    parser.add_argument(
        "--n-informative-features",
        type=int,
        default=200,
        help=(
            "Fallback number of informative features for the vertical reference line "
            "(synthetic mode) or when n_original_features is missing (task mode)."
        ),
    )

    args = parser.parse_args()

    if args.task_dir is not None:
        run_task_mode(args)
    else:
        run_synthetic_mode(args)


if __name__ == "__main__":
    main()