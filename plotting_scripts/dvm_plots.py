from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(
    "/home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/results/dvm_results/dvm_sweep.json"
)
DEFAULT_OUTPUT = Path(
    "/home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/plots/dvm_sweep_plot.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DVM sweep experiment results")
    parser.add_argument(
        "--input-json",
        type=Path,
        nargs="+",
        default=[DEFAULT_INPUT],
        help="One or more paths to DVM sweep JSON result files",
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
    logx: bool,
    colors_by_label: dict[str, str] | None = None,
    styles_by_label: dict[str, str] | None = None,
    markers_by_label: dict[str, str] | None = None,
    title_suffix: str = "",
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
    ax.set_title(f"DVM Sweep: {split.capitalize()} {ylabel}{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def make_plot(
    left_series: dict[str, list[tuple[int, float]]],
    left_labels: list[str],
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
            logx,
            colors_by_label=left_colors,
            styles_by_label=left_styles,
            markers_by_label=left_markers,
            title_suffix=" | base (none)",
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        _plot_on_axis(
            axes[0],
            left_series,
            left_labels,
            split,
            metric,
            logx,
            colors_by_label=left_colors,
            styles_by_label=left_styles,
            markers_by_label=left_markers,
            title_suffix=" | base (none)",
        )
        _plot_on_axis(
            axes[1],
            right_series,
            right_labels or [],
            split,
            metric,
            logx,
            colors_by_label=right_colors,
            styles_by_label=right_styles,
            markers_by_label=right_markers,
            title_suffix=" | reduced",
        )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = _merge_results(args.input_json)

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
