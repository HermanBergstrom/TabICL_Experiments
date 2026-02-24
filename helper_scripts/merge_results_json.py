"""Merge selected baselines from a source results JSON into a target results JSON.

The target file is not modified in place; a new JSON is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _get_baselines(baselines: Iterable[str] | None, source: dict) -> list[str]:
    if baselines:
        return list(baselines)

    # Default to all baselines present in the first results entry
    results = source.get("results", {})
    if not results:
        return []
    first_key = sorted(results.keys())[0]
    first_entry = results.get(first_key, {})
    return [k for k, v in first_entry.items() if isinstance(v, list)]


def merge_baselines(
    source: dict,
    target: dict,
    baselines: Iterable[str],
    rename_map: dict[str, str] | None = None,
) -> dict:
    """Merge baselines from source to target, optionally renaming them.
    
    Args:
        source: Source results dictionary
        target: Target results dictionary
        baselines: List of baseline keys to copy from source
        rename_map: Optional dict mapping source names to target names
    """
    if rename_map is None:
        rename_map = {}
    
    source_results = source.get("results", {})
    target_results = target.get("results", {})

    for size_key, target_entry in target_results.items():
        source_entry = source_results.get(size_key, {})
        if not source_entry:
            continue
        for baseline in baselines:
            if baseline in source_entry:
                # Use renamed key if provided, otherwise use original
                target_key = rename_map.get(baseline, baseline)
                target_entry[target_key] = source_entry[baseline]

    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge selected baselines from a source JSON into a target JSON."
    )
    parser.add_argument("--source", type=Path, required=True, help="Source results JSON file.")
    parser.add_argument("--target", type=Path, required=True, help="Target results JSON file.")
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=None,
        help="Baseline keys to copy (e.g., linear_probe tabicl knn). If omitted, all baselines from source are used.",
    )
    parser.add_argument(
        "--rename",
        nargs="*",
        default=None,
        help="Rename baselines in format 'old:new' (e.g., tabicl:tabicl_v2). Can specify multiple.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <target>_merged.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _load_json(args.source)
    target = _load_json(args.target)

    baselines = _get_baselines(args.baselines, source)
    if not baselines:
        raise ValueError("No baselines found in source JSON.")

    # Parse rename mappings
    rename_map = {}
    if args.rename:
        for rename_spec in args.rename:
            if ":" not in rename_spec:
                raise ValueError(f"Invalid rename format '{rename_spec}'. Use 'old:new' format.")
            old, new = rename_spec.split(":", 1)
            rename_map[old] = new

    merged = merge_baselines(source, target, baselines, rename_map)

    output_path = args.output
    if output_path is None:
        output_path = args.target.with_name(f"{args.target.stem}_merged{args.target.suffix}")

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Wrote merged results to {output_path}")


if __name__ == "__main__":
    main()
