from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_INPUT = Path(
    "/home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/results/dvm_results/dvm_sweep.json"
)
DEFAULT_OUTPUT = Path(
    "/home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/plots/experiments_sweep_plot.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment sweep results")
    parser.add_argument(
        "--input-json",
        type=Path,
        nargs="+",
        default=[DEFAULT_INPUT],
        help="One or more paths to experiment JSON result files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path (.png/.pdf/.svg)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Which split metrics to plot",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["accuracy", "f1_macro"],
        default="accuracy",
        help="Metric to plot on y-axis",
    )
    parser.add_argument(
        "--logx",
        action="store_true",
        help="Use log scale on the x-axis (train size)",
    )
    parser.add_argument(
        "--plot-kind",
        type=str,
        choices=["lineplot"],
        default="lineplot",
        help="Plot type to generate",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Methods to include (e.g., tabicl xgboost linear_probe)",
    )
    parser.add_argument(
        "--reducers",
        type=str,
        nargs="+",
        default=None,
        choices=["pca", "ica", "rp", "random_projection"],
        help="Reduced-image reducers to plot in a second panel (e.g., ica pca)",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _merge_results(paths: list[Path]) -> dict:
    """Load one or more JSON result files and merge into a unified structure."""
    all_experiments: dict[str, dict] = {}   # label -> experiment info + _runs dict
    metadata: dict = {}

    for path in paths:
        data = load_results(path)
        if not metadata:
            metadata = dict(data.get("metadata", {}))

        experiments = data.get("experiments", [])
        if not experiments and "runs" in data:
            fe = data.get("metadata", {}).get("feature_experiments", [{}])
            label = fe[0].get("label", "default") if fe else "default"
            experiments = [{"label": label, "runs": data["runs"]}]

        for exp in experiments:
            label = exp.get("label", "unknown")
            if label not in all_experiments:
                all_experiments[label] = {
                    k: v for k, v in exp.items() if k != "runs"
                }
                all_experiments[label]["_runs_by_size"] = {}

            for run in exp.get("runs", []):
                ts = int(run.get("train_size_actual", 0))
                bucket = all_experiments[label]["_runs_by_size"]
                if ts not in bucket:
                    bucket[ts] = {
                        "train_size_requested": run.get("train_size_requested"),
                        "train_size_actual": ts,
                        "models": {},
                    }
                bucket[ts]["models"].update(run.get("models", {}))

    # Flatten back to sorted lists.
    experiments_list = []
    for label in sorted(all_experiments):
        exp = {k: v for k, v in all_experiments[label].items() if k != "_runs_by_size"}
        exp["runs"] = sorted(
            all_experiments[label]["_runs_by_size"].values(),
            key=lambda r: r["train_size_actual"],
        )
        experiments_list.append(exp)

    result: dict = {"metadata": metadata, "experiments": experiments_list}
    if len(experiments_list) == 1:
        result["runs"] = experiments_list[0]["runs"]
    return result


def _dataset_display_name(results: dict) -> str:
    """Resolve a human-readable dataset name from results metadata.

    Backward compatibility: when the dataset key is missing, default to DVM.
    """
    raw = str(results.get("metadata", {}).get("dataset", "dvm")).strip().lower()
    pretty = {
        "dvm": "DVM",
        "petfinder": "PetFinder",
    }
    return pretty.get(raw, raw.replace("_", " ").title())


def _normalize_reducer_name(name: str) -> str:
    return "rp" if name == "random_projection" else name


def _available_methods(results: dict) -> list[str]:
    methods: set[str] = set()
    for experiment in results.get("experiments", []):
        for run in experiment.get("runs", []):
            methods.update(run.get("models", {}).keys())
    return sorted(methods)


def _feature_group(label: str) -> str:
    if label.startswith("image"):
        return "image"
    if label.startswith("concat"):
        return "concat"
    if label.startswith("tabular"):
        return "tabular"
    return label


def _reducer_style_key(label: str) -> str:
    lowered = label.lower()
    if "random_projection" in lowered or "_rp" in lowered:
        return "rp"
    if "ica" in lowered:
        return "ica"
    if "pca" in lowered:
        return "pca"
    return "none"


def _style_for_label(label: str) -> str:
    style_by_key = {
        "none": "-",
        "pca": "--",
        "ica": "-.",
        "rp": ":",
    }
    return style_by_key[_reducer_style_key(label)]


def _reducer_dims_from_labels(labels: list[str], reducer: str) -> list[int]:
    """Extract reducer dimensions from labels like image_pca64 / concat_rp128."""
    if reducer == "none":
        return []

    if reducer == "rp":
        pattern = r"(?:^|_)(?:rp|random_projection)(\d+)(?:$|_)"
    else:
        pattern = rf"(?:^|_){reducer}(\d+)(?:$|_)"

    dims: set[int] = set()
    for label in labels:
        match = re.search(pattern, label.lower())
        if match:
            dims.add(int(match.group(1)))
    return sorted(dims)


def _pick_canonical_label(labels: list[str], group: str, reducer: str) -> str | None:
    candidates = [
        label for label in labels
        if _feature_group(label) == group and _reducer_style_key(label) == reducer
    ]
    if not candidates:
        return None

    if group == "tabular" and reducer == "none":
        preferred = ["tabular_only", "tabular_none", "tabular"]
    elif group == "image" and reducer == "none":
        preferred = ["image_only", "image_none", "image"]
    elif group == "concat" and reducer == "none":
        preferred = ["concat", "concat_none"]
    else:
        preferred = []

    for candidate in preferred:
        if candidate in candidates:
            return candidate
    return sorted(candidates)[0]


def _select_base_labels(labels: list[str]) -> list[str]:
    selected: list[str] = []
    for group in ["tabular", "image", "concat"]:
        label = _pick_canonical_label(labels, group, "none")
        if label is not None:
            selected.append(label)
    return selected


def _select_reduced_labels(labels: list[str], reducers: list[str]) -> list[str]:
    selected: list[str] = []

    tabular = _pick_canonical_label(labels, "tabular", "none")
    if tabular is not None:
        selected.append(tabular)

    for reducer in reducers:
        for group in ["image", "concat"]:
            label = _pick_canonical_label(labels, group, reducer)
            if label is not None and label not in selected:
                selected.append(label)
    return selected


def collect_series_for_labels(
    results: dict,
    split: str,
    metric: str,
    methods: list[str],
    feature_labels: list[str],
    color_mode: str = "method",
) -> tuple[
    dict[str, list[tuple[int, float]]],
    list[str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    experiments = results.get("experiments", [])
    if not experiments:
        raise ValueError("No 'experiments' found in results JSON.")

    feature_set = set(feature_labels)
    series: dict[str, list[tuple[int, float]]] = {}
    colors_by_line: dict[str, str] = {}
    styles_by_line: dict[str, str] = {}
    markers_by_line: dict[str, str] = {}

    method_colors = {
        method: plt.get_cmap("tab10")(index % 10)
        for index, method in enumerate(methods)
    }
    modality_colors = {
        "tabular": plt.get_cmap("tab10")(2),
        "image": plt.get_cmap("tab10")(1),
        "concat": plt.get_cmap("tab10")(0),
    }
    marker_by_group = {
        "tabular": "o",
        "image": "s",
        "concat": "^",
    }

    for experiment in experiments:
        label = experiment.get("label", "unknown")
        if label not in feature_set:
            continue
        runs = experiment.get("runs", [])
        for run in runs:
            train_size = int(run.get("train_size_actual", 0))
            model_block = run.get("models", {})

            for method in methods:
                metrics = model_block.get(method, {})
                value = metrics.get(split, {}).get(metric)
                if value is None:
                    continue

                line_label = f"{method} | {label}"
                series.setdefault(line_label, []).append((train_size, float(value)))
                if color_mode == "modality":
                    colors_by_line[line_label] = modality_colors.get(_feature_group(label), plt.get_cmap("tab10")(3))
                else:
                    colors_by_line[line_label] = method_colors[method]
                styles_by_line[line_label] = _style_for_label(label)
                markers_by_line[line_label] = marker_by_group.get(_feature_group(label), "o")

    for line_label in series:
        series[line_label] = sorted(series[line_label], key=lambda x: x[0])

    ordered_labels = [
        f"{method} | {label}"
        for method in methods
        for label in feature_labels
        if f"{method} | {label}" in series
    ]
    return series, ordered_labels, colors_by_line, styles_by_line, markers_by_line


def _plot_on_axis(
    ax,
    series: dict[str, list[tuple[int, float]]],
    labels: list[str],
    split: str,
    metric: str,
    dataset_name: str,
    logx: bool,
    colors_by_label: dict[str, str] | None = None,
    styles_by_label: dict[str, str] | None = None,
    markers_by_label: dict[str, str] | None = None,
    title_suffix: str = "",
    split_legend: bool = False,
) -> None:

    for label in labels:
        if label not in series:
            continue
        xy = series[label]
        x_vals = [x for x, _ in xy]
        y_vals = [y for _, y in xy]
        color = None if colors_by_label is None else colors_by_label.get(label)
        linestyle = "-" if styles_by_label is None else styles_by_label.get(label, "-")
        marker = "o" if markers_by_label is None else markers_by_label.get(label, "o")
        ax.plot(x_vals, y_vals, marker=marker, linewidth=2, linestyle=linestyle, color=color, label=label)

    if logx:
        ax.set_xscale("log")

    ax.set_xlabel("Train size")
    ylabel = "Accuracy" if metric == "accuracy" else "Macro F1"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset_name} Sweep: {split.capitalize()} {ylabel}{title_suffix}")
    ax.grid(True, alpha=0.3)

    if not split_legend:
        ax.legend(fontsize=8)
        return

    plotted_feature_labels = [
        lbl.split("|", 1)[1].strip()
        for lbl in labels
        if "|" in lbl and lbl in series
    ]
    modalities_present = [m for m in ["tabular", "image", "concat"] if any(_feature_group(f) == m for f in plotted_feature_labels)]
    reducers_present = [r for r in ["none", "pca", "ica", "rp"] if any(_reducer_style_key(f) == r for f in plotted_feature_labels)]

    modality_color_map = {
        "tabular": plt.get_cmap("tab10")(2),
        "image": plt.get_cmap("tab10")(1),
        "concat": plt.get_cmap("tab10")(0),
    }
    reducer_style_map = {
        "none": "-",
        "pca": "--",
        "ica": "-.",
        "rp": ":",
    }

    modality_handles = [
        Line2D([0], [0], color=modality_color_map[m], linestyle="-", linewidth=2, label=m)
        for m in modalities_present
    ]
    reducer_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=reducer_style_map[r],
            linewidth=2,
            label=(
                f"{r}{'/'.join(str(d) for d in _reducer_dims_from_labels(plotted_feature_labels, r))}"
                if _reducer_dims_from_labels(plotted_feature_labels, r)
                else r
            ),
        )
        for r in reducers_present
    ]

    legend_modalities = ax.legend(
        handles=modality_handles,
        title="Modality (color)",
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        handlelength=3.0,
        handletextpad=0.9,
    )
    ax.add_artist(legend_modalities)
    ax.legend(
        handles=reducer_handles,
        title="Reducer (linestyle)",
        loc="lower right",
        fontsize=8,
        title_fontsize=9,
        handlelength=4.0,
        handletextpad=0.9,
    )


def make_plot(
    left_series: dict[str, list[tuple[int, float]]],
    left_labels: list[str],
    dataset_name: str,
    split: str,
    metric: str,
    output: Path,
    logx: bool,
    left_colors: dict[str, str],
    left_styles: dict[str, str],
    left_markers: dict[str, str],
    right_series: dict[str, list[tuple[int, float]]] | None = None,
    right_labels: list[str] | None = None,
    right_colors: dict[str, str] | None = None,
    right_styles: dict[str, str] | None = None,
    right_markers: dict[str, str] | None = None,
    split_legend: bool = False,
) -> None:
    if not left_series:
        raise ValueError("No plottable series found in the results file.")

    if right_series is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_on_axis(
            ax,
            left_series,
            left_labels,
            split,
            metric,
            dataset_name,
            logx,
            colors_by_label=left_colors,
            styles_by_label=left_styles,
            markers_by_label=left_markers,
            title_suffix=" | base (none)",
            split_legend=split_legend,
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        _plot_on_axis(
            axes[0],
            left_series,
            left_labels,
            split,
            metric,
            dataset_name,
            logx,
            colors_by_label=left_colors,
            styles_by_label=left_styles,
            markers_by_label=left_markers,
            title_suffix=" | base (none)",
            split_legend=False,
        )
        _plot_on_axis(
            axes[1],
            right_series,
            right_labels or [],
            split,
            metric,
            dataset_name,
            logx,
            colors_by_label=right_colors,
            styles_by_label=right_styles,
            markers_by_label=right_markers,
            title_suffix=" | reduced",
            split_legend=False,
        )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = _merge_results(args.input_json)
    dataset_name = _dataset_display_name(results)

    if args.plot_kind != "lineplot":
        raise ValueError(f"Unsupported plot kind: {args.plot_kind}")

    labels_all = [exp.get("label", "unknown") for exp in results.get("experiments", [])]
    base_labels = _select_base_labels(labels_all)
    if not base_labels:
        raise ValueError("Could not find base labels (tabular/image/concat with no reduction).")

    available_methods = _available_methods(results)
    if not available_methods:
        raise ValueError("No methods found in the provided results files.")

    selected_methods = args.methods if args.methods is not None else available_methods
    selected_methods = [m for m in selected_methods if m in available_methods]
    if not selected_methods:
        raise ValueError("No methods remain after --methods filtering.")

    left_series, left_labels, left_colors, left_styles, left_markers = collect_series_for_labels(
        results,
        split=args.split,
        metric=args.metric,
        methods=selected_methods,
        feature_labels=base_labels,
    )

    reducers = None if args.reducers is None else [_normalize_reducer_name(r) for r in args.reducers]
    if reducers:
        if len(selected_methods) == 1:
            reduced_labels = _select_reduced_labels(labels_all, reducers)
            combined_labels = list(dict.fromkeys(base_labels + reduced_labels))

            combined_series, combined_ordered_labels, combined_colors, combined_styles, combined_markers = collect_series_for_labels(
                results,
                split=args.split,
                metric=args.metric,
                methods=selected_methods,
                feature_labels=combined_labels,
                color_mode="modality",
            )

            make_plot(
                left_series=combined_series,
                left_labels=combined_ordered_labels,
                dataset_name=dataset_name,
                split=args.split,
                metric=args.metric,
                output=args.output,
                logx=args.logx,
                left_colors=combined_colors,
                left_styles=combined_styles,
                left_markers=combined_markers,
                split_legend=True,
            )
            print(f"Saved plot to: {args.output}")
            return

        reduced_labels = _select_reduced_labels(labels_all, reducers)
        if len(reduced_labels) <= 1:
            raise ValueError(
                f"No reduced image/concat labels found for reducers={reducers}."
            )

        right_series, right_labels, right_colors, right_styles, right_markers = collect_series_for_labels(
            results,
            split=args.split,
            metric=args.metric,
            methods=selected_methods,
            feature_labels=reduced_labels,
        )

        make_plot(
            left_series=left_series,
            left_labels=left_labels,
            dataset_name=dataset_name,
            split=args.split,
            metric=args.metric,
            output=args.output,
            logx=args.logx,
            left_colors=left_colors,
            left_styles=left_styles,
            left_markers=left_markers,
            right_series=right_series,
            right_labels=right_labels,
            right_colors=right_colors,
            right_styles=right_styles,
            right_markers=right_markers,
        )
    else:
        make_plot(
            left_series=left_series,
            left_labels=left_labels,
            dataset_name=dataset_name,
            split=args.split,
            metric=args.metric,
            output=args.output,
            logx=args.logx,
            left_colors=left_colors,
            left_styles=left_styles,
            left_markers=left_markers,
        )

    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
