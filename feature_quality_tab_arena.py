from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openml
from skrub import TableVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from feature_quality_experiment import (
    ALL_STRATEGIES,
    build_experiment_datasets,
    build_reduced_datasets,
    compute_signal_imbalance,
    compute_signal_imbalance_lr,
    get_model_factories,
    print_results_table,
)


RESULT_FIELDNAMES = [
    "task_id",
    "dataset_id",
    "dataset_name",
    "repeat",
    "fold",
    "seed",
    "variant",
    "model",
    "accuracy",
    "f1",
    "auc",
    "fit_seconds",
    "predict_seconds",
    "normalized_effective_rank",
    "signal_imbalance_dt",
    "normalized_signal_imbalance_dt",
    "signal_imbalance_lr",
    "normalized_signal_imbalance_lr",
    "n_original_features",
    "train_pos_count",
    "train_neg_count",
    "test_pos_count",
    "test_neg_count",
    "train_pos_frac",
    "test_pos_frac",
    "combined_pos_frac",
    "error",
]


def sanitize_for_path(text):
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))
    return safe.strip("_") or "unknown"


def task_output_csv_path(output_dir, task_id, dataset_id, dataset_name):
    task_folder = (
        output_dir
        / f"task_{task_id}"
        / f"dataset_{dataset_id}_{sanitize_for_path(dataset_name)}"
    )
    return task_folder / "results.csv"


def save_dry_run_manifest(output_dir, dry_run_rows):
    if not dry_run_rows:
        return

    manifest_csv = output_dir / "dry_run_candidates.csv"
    task_ids_txt = output_dir / "dry_run_task_ids.txt"

    fieldnames = [
        "task_id",
        "dataset_id",
        "dataset_name",
        "successful_splits",
        "total_splits",
        "mean_normalized_effective_rank",
        "max_n_original_features",
        "mean_train_pos_frac",
        "mean_test_pos_frac",
        "mean_combined_pos_frac",
    ]

    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dry_run_rows)

    task_ids = sorted({row["task_id"] for row in dry_run_rows})
    with task_ids_txt.open("w") as f:
        for task_id in task_ids:
            f.write(f"{task_id}\n")

    print(f"Dry-run manifest saved: {manifest_csv}")
    print(f"Task queue list saved: {task_ids_txt}")


def effective_rank(X):
    X = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    return np.exp(-np.sum(p * np.log(p)))


def class_balance_stats(y_binary):
    y_binary = np.asarray(y_binary, dtype=np.int64)
    n_total = int(y_binary.shape[0])
    n_pos = int(np.sum(y_binary == 1))
    n_neg = int(n_total - n_pos)
    pos_frac = float(n_pos / n_total) if n_total > 0 else np.nan
    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_frac": pos_frac,
    }


def balanced_downsample_binary_split(X, y_binary, max_samples, random_state):
    """
    Downsamples to at most max_samples while keeping classes as balanced as possible.
    """
    y_binary = np.asarray(y_binary, dtype=np.int64)
    n_total = y_binary.shape[0]
    if n_total == 0:
        return X, y_binary

    target_n = min(max_samples, n_total)
    if target_n == n_total:
        return X.reset_index(drop=True), y_binary

    pos_idx = np.flatnonzero(y_binary == 1)
    neg_idx = np.flatnonzero(y_binary == 0)

    # If only one class exists, fall back to random subsampling.
    rng = np.random.default_rng(random_state)
    if pos_idx.size == 0 or neg_idx.size == 0:
        selected = rng.choice(n_total, size=target_n, replace=False)
        selected.sort()
        return X.iloc[selected].reset_index(drop=True), y_binary[selected]

    # Choose class counts that are closest to 50/50 while respecting availability.
    k_pos_min = max(0, target_n - neg_idx.size)
    k_pos_max = min(pos_idx.size, target_n)
    k_pos = int(np.clip(round(target_n / 2), k_pos_min, k_pos_max))
    k_neg = target_n - k_pos

    sel_pos = rng.choice(pos_idx, size=k_pos, replace=False)
    sel_neg = rng.choice(neg_idx, size=k_neg, replace=False)
    selected = np.concatenate([sel_pos, sel_neg])
    selected.sort()

    return X.iloc[selected].reset_index(drop=True), y_binary[selected]


def normalized_effective_rank(X):
    n, d = X.shape
    rank = effective_rank(X)
    return rank / min(n, d)


def singular_value_profile(X, label):
    X = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s_norm = s / s.sum()
    plt.plot(np.cumsum(s_norm), label=label)


def evaluate_models_on_fixed_split(
    X_train,
    y_train,
    X_test,
    y_test,
    model_factories,
    random_state=42,
):
    import time
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    # Keep feature ordering randomized but consistent between train and test.
    rng = np.random.default_rng(random_state)
    feature_perm = rng.permutation(X_train.shape[1])
    X_train = X_train[:, feature_perm]
    X_test = X_test[:, feature_perm]

    results = []
    for model_name, build_model in model_factories.items():
        model = build_model()
        run_result = {
            "model": model_name,
            "accuracy": np.nan,
            "f1": np.nan,
            "auc": np.nan,
            "fit_seconds": np.nan,
            "predict_seconds": np.nan,
            "error": "",
        }

        try:
            start_fit = time.perf_counter()
            model.fit(X_train, y_train)
            run_result["fit_seconds"] = time.perf_counter() - start_fit

            start_pred = time.perf_counter()
            y_pred = model.predict(X_test)
            run_result["predict_seconds"] = time.perf_counter() - start_pred

            run_result["accuracy"] = accuracy_score(y_test, y_pred)
            run_result["f1"] = f1_score(y_test, y_pred)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    run_result["auc"] = roc_auc_score(y_test, y_proba[:, 1])
                elif y_proba.ndim == 1:
                    run_result["auc"] = roc_auc_score(y_test, y_proba)
        except Exception as exc:
            run_result["error"] = str(exc)

        results.append(run_result)

    return results


def build_split_datasets(
    X_train,
    y_train,
    X_test,
    y_test,
    seed,
    reduction_only=False,
    only_projected=False,
    strategies=None,
):
    """
    Builds feature-quality variants while preserving the provided split.
    """
    X_full = np.concatenate([X_train, X_test], axis=0)
    y_full = np.concatenate([y_train, y_test], axis=0)
    n_train = X_train.shape[0]

    datasets_full = build_experiment_datasets(
        X_full,
        y_full,
        noise_levels=(100, 200, 400, 2000),
        low_rank_levels=(100, 200, 400, 1000, 2000),
        low_signal_levels=(100, 200, 400, 1000, 2000),
        signal_strength=0.01,
        only_projected=only_projected,
        strategies=strategies,
        random_state=seed,
    )
    reduced_full = build_reduced_datasets(
        datasets_full,
        n_components=X_train.shape[1],
        random_state=seed,
    )

    split_datasets = {}
    if not reduction_only:
        for dataset_name, X_data in datasets_full.items():
            split_datasets[dataset_name] = (X_data[:n_train], X_data[n_train:])

    for dataset_name, X_data in reduced_full.items():
        split_datasets[dataset_name] = (X_data[:n_train], X_data[n_train:])

    return split_datasets


def save_results_csv(all_results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)


def append_results_csv(results_batch, output_path):
    if not results_batch:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(results_batch)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="TabArena feature-quality evaluation")
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Use TabArena-Lite version (first repeat of first fold only).",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Run only a single OpenML task ID (useful for one-job-per-task scheduling).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fetch data, run split preprocessing, and compute effective rank, "
            "but skip model fitting. Writes a queue-ready manifest."
        ),
    )
    parser.add_argument(
        "--reduction-only",
        action="store_true",
        help="Only evaluate reduced datasets (PCA and random projection).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tabarena_feature_quality"),
        help="Root folder where each task gets its own subfolder and CSV.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=2000,
        help="Cap train rows per fold to keep runtime manageable.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=400,
        help="Cap test rows per fold to keep runtime manageable.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help=(
            "Number of seeds to run per task. Defaults to the total number of "
            "available (repeat, fold) splits. If larger than the available splits, "
            "splits are reused with a different subsampling seed."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Only run tasks whose dataset name contains this substring (case-insensitive).",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="+",
        default=None,
        metavar="COL",
        help="Drop these raw columns before vectorization (e.g. --drop-columns state).",
    )
    parser.add_argument(
        "--only-projected",
        action="store_true",
        help=(
            "Return only the constructed (projected/weak) features, without "
            "concatenating the original features. No effect on the 'noise' strategy."
        ),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(ALL_STRATEGIES),
        default=None,
        metavar="STRATEGY",
        help=(
            f"Subset of strategies to run. Choices: {list(ALL_STRATEGIES)}. "
            "If omitted, all strategies are run."
        ),
    )
    return parser.parse_args()


def tab_arena_test(args):
    all_results = []
    dry_run_rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_suite = openml.study.get_suite("tabarena-v0.1")
    task_ids = benchmark_suite.tasks
    if args.task_id is not None:
        task_ids = [tid for tid in task_ids if tid == args.task_id]
        if not task_ids:
            raise ValueError(f"Task ID {args.task_id} is not part of tabarena-v0.1")

    if args.dataset_name is not None:
        print(f"Filtering tasks to dataset names containing: '{args.dataset_name}'")
    if args.drop_columns:
        print(f"Dropping columns before vectorization: {args.drop_columns}")
    print("Getting data for TabArena tasks...")
    if args.lite:
        print("TabArena Lite is enabled. Using first repeat of first fold only.")
    if args.dry_run:
        print("Dry-run mode enabled: model fitting is skipped.")

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        task_results = []
        task_rank_values = []
        task_n_original_features = []
        task_train_pos_frac = []
        task_test_pos_frac = []
        task_combined_pos_frac = []
        successful_splits = 0

        task_csv_path = task_output_csv_path(
            args.output_dir,
            task_id=task.id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
        )
        if task_csv_path.exists() and not args.dry_run:
            task_csv_path.unlink()

        if args.dataset_name is not None and args.dataset_name.lower() not in dataset.name.lower():
            continue

        #Only skip regression
        if dataset.qualities["NumberOfClasses"] == 0 or dataset.qualities["NumberOfClasses"] > 100:
            print(f"Skipping task {task.id}: regression.")
            continue

        #if dataset.qualities["NumberOfClasses"] != 2:
        #    print(f"Skipping task {task.id}: regression or multi-class.")
        #    continue


        if dataset.qualities["NumberOfFeatures"] > 1000:
            print(f"Skipping task {task.id}: more than 1000 features.")
            continue

        #if dataset.qualities["NumberOfFeatures"] < 5:
        #    print(f"Skipping task {task.id}: fewer than 5 features.")
        #    continue


        minority_ratio = dataset.qualities.get("MinorityClassSize") / (
            dataset.qualities.get("MajorityClassSize")
            + dataset.qualities.get("MinorityClassSize")
        )
        if minority_ratio < 0.1:
            print(f"Skipping task {task.id}: highly imbalanced.")
            continue

        print(f"Task ID: {task.id}, Dataset ID: {dataset.id}, Dataset Name: {dataset.name}")

        if args.lite:
            folds = 1
            tabarena_repeats = 1
        else:
            _, folds, _ = task.get_split_dimensions()
            n_instances = dataset.qualities["NumberOfInstances"]
            if n_instances < 2_500:
                tabarena_repeats = 10
            elif n_instances > 250_000:
                tabarena_repeats = 1
            else:
                tabarena_repeats = 3

        total_splits = tabarena_repeats * folds
        num_seeds = args.num_seeds if args.num_seeds is not None else total_splits
        print(f"TabArena repeats: {tabarena_repeats} | folds: {folds} | num_seeds: {num_seeds}")

        for seed_idx in range(num_seeds):
            split_idx = seed_idx % total_splits
            repeat = split_idx // folds
            fold = split_idx % folds
            seed = seed_idx + 1

            x, y, _, _ = dataset.get_data(
                target=task.target_name,
                dataset_format="dataframe",
            )
            train_indices, test_indices = task.get_train_test_split_indices(
                fold=fold,
                repeat=repeat,
            )
            x_train = x.iloc[train_indices]
            y_train = y.iloc[train_indices]
            x_test = x.iloc[test_indices]
            y_test = y.iloc[test_indices]

            if args.drop_columns:
                cols_to_drop = [c for c in args.drop_columns if c in x_train.columns]
                if cols_to_drop:
                    x_train = x_train.drop(columns=cols_to_drop)
                    x_test = x_test.drop(columns=cols_to_drop)

            pos_class = task.class_labels[1]
            if pos_class == "True":
                pos_class = True

            y_train_binary_full = np.array([1 if label == pos_class else 0 for label in y_train])
            y_test_binary_full = np.array([1 if label == pos_class else 0 for label in y_test])

            x_train, y_train_binary = balanced_downsample_binary_split(
                x_train,
                y_train_binary_full,
                max_samples=args.max_train_samples,
                random_state=seed,
            )
            x_test, y_test_binary = balanced_downsample_binary_split(
                x_test,
                y_test_binary_full,
                max_samples=args.max_test_samples,
                random_state=seed,
            )

            train_balance = class_balance_stats(y_train_binary)
            test_balance = class_balance_stats(y_test_binary)
            combined_balance = class_balance_stats(
                np.concatenate([y_train_binary, y_test_binary], axis=0)
            )

            vectorizer = TableVectorizer()
            imputer = SimpleImputer(strategy="constant", fill_value=0.0)
            scaler = StandardScaler()
            X_train_vectorized = vectorizer.fit_transform(x_train)

            _debug_datasets = ()
            if any(name in dataset.name.lower() for name in _debug_datasets):
                cols = vectorizer.get_feature_names_out().tolist()
                print(
                    f"\n[DEBUG] {dataset.name} | seed={seed} repeat={repeat} fold={fold}"
                    f" | {len(cols)} columns after vectorization:"
                )
                for col in cols:
                    print(f"  {col}")

            X_train_imputed = imputer.fit_transform(X_train_vectorized)
            X_train_normalized = scaler.fit_transform(X_train_imputed)

            X_test_vectorized = vectorizer.transform(x_test)
            X_test_imputed = imputer.transform(X_test_vectorized)
            X_test_normalized = scaler.transform(X_test_imputed)

            # Guardrail: skip split if any non-finite values still remain.
            if not np.isfinite(X_train_normalized).all() or not np.isfinite(X_test_normalized).all():
                print(
                    f"Skipping task {task.id} repeat {repeat} fold {fold}: "
                    "non-finite values remain after imputation/scaling."
                )
                continue

            n_original_features = X_train_normalized.shape[1]

            split_datasets = build_split_datasets(
                X_train=X_train_normalized,
                y_train=y_train_binary,
                X_test=X_test_normalized,
                y_test=y_test_binary,
                seed=seed,
                reduction_only=args.reduction_only,
                only_projected=args.only_projected,
                strategies=args.strategies,
            )

            try:
                normalized_effective_rank_base = normalized_effective_rank(
                    np.concatenate([X_train_normalized, X_test_normalized], axis=0)
                )
            except Exception as e:
                print(f"Error computing normalized effective rank: {e}")
                continue

            try:
                signal_imbalance_dt, normalized_signal_imbalance_dt = (
                    compute_signal_imbalance(X_train_normalized, y_train_binary, random_state=seed)
                )
            except Exception as e:
                print(f"Error computing signal imbalance (DT): {e}")
                signal_imbalance_dt, normalized_signal_imbalance_dt = np.nan, np.nan

            try:
                signal_imbalance_lr, normalized_signal_imbalance_lr = (
                    compute_signal_imbalance_lr(X_train_normalized, y_train_binary, random_state=seed)
                )
            except Exception as e:
                print(f"Error computing signal imbalance (LR): {e}")
                signal_imbalance_lr, normalized_signal_imbalance_lr = np.nan, np.nan

            print(
                f"Task {task.id} | seed {seed_idx + 1}/{num_seeds} "
                f"(repeat {repeat + 1}/{tabarena_repeats}, fold {fold + 1}/{folds}) | "
                f"base normalized rank={normalized_effective_rank_base:.4f}"
            )

            successful_splits += 1
            task_rank_values.append(normalized_effective_rank_base)
            task_n_original_features.append(n_original_features)
            task_train_pos_frac.append(train_balance["pos_frac"])
            task_test_pos_frac.append(test_balance["pos_frac"])
            task_combined_pos_frac.append(combined_balance["pos_frac"])

            if args.dry_run:
                continue

            model_factories = get_model_factories(random_state=seed)
            seed_results = []
            for variant_name, (X_variant_train, X_variant_test) in tqdm(
                split_datasets.items(),
                desc=f"Task {task.id} seed {seed_idx + 1}/{num_seeds}",
                leave=False,
            ):
                variant_results = evaluate_models_on_fixed_split(
                    X_train=X_variant_train,
                    y_train=y_train_binary,
                    X_test=X_variant_test,
                    y_test=y_test_binary,
                    model_factories=model_factories,
                    random_state=seed,
                )

                for row in variant_results:
                    row.update(
                        {
                            "task_id": task.id,
                            "dataset_id": dataset.id,
                            "dataset_name": dataset.name,
                            "repeat": repeat,
                            "fold": fold,
                            "seed": seed,
                            "variant": variant_name,
                            "normalized_effective_rank": normalized_effective_rank_base,
                            "signal_imbalance_dt": signal_imbalance_dt,
                            "normalized_signal_imbalance_dt": normalized_signal_imbalance_dt,
                            "signal_imbalance_lr": signal_imbalance_lr,
                            "normalized_signal_imbalance_lr": normalized_signal_imbalance_lr,
                            "n_original_features": n_original_features,
                            "train_pos_count": train_balance["n_pos"],
                            "train_neg_count": train_balance["n_neg"],
                            "test_pos_count": test_balance["n_pos"],
                            "test_neg_count": test_balance["n_neg"],
                            "train_pos_frac": train_balance["pos_frac"],
                            "test_pos_frac": test_balance["pos_frac"],
                            "combined_pos_frac": combined_balance["pos_frac"],
                        }
                    )
                    all_results.append(row)
                    task_results.append(row)
                    seed_results.append(row)

            append_results_csv(seed_results, task_csv_path)
            print(
                f"Checkpoint saved for task {task.id} seed {seed_idx + 1}/{num_seeds}: "
                f"{len(seed_results)} rows -> {task_csv_path}"
            )

        if args.dry_run:
            if successful_splits > 0:
                dry_run_rows.append(
                    {
                        "task_id": task.id,
                        "dataset_id": dataset.id,
                        "dataset_name": dataset.name,
                        "successful_splits": successful_splits,
                        "total_splits": num_seeds,
                        "mean_normalized_effective_rank": float(np.mean(task_rank_values)),
                        "max_n_original_features": int(max(task_n_original_features)),
                        "mean_train_pos_frac": float(np.mean(task_train_pos_frac)),
                        "mean_test_pos_frac": float(np.mean(task_test_pos_frac)),
                        "mean_combined_pos_frac": float(np.mean(task_combined_pos_frac)),
                    }
                )
                print(
                    f"Dry-run candidate task {task.id}: {successful_splits}/{num_seeds} "
                    f"splits preprocessed successfully."
                )
            else:
                print(f"Dry-run rejected task {task.id}: no successful splits.")
        else:
            print(
                f"Task {task.id} complete: {len(task_results)} total rows saved to {task_csv_path}"
            )

    if args.dry_run:
        save_dry_run_manifest(args.output_dir, dry_run_rows)

    return all_results


if __name__ == "__main__":
    args = parse_args()
    all_results = tab_arena_test(args)

    if args.dry_run:
        print()
        print(f"Dry run complete. Queue files are in: {args.output_dir}")
        raise SystemExit(0)

    print("Model Benchmark Results")
    print("-" * 60)
    print_results_table(
        [
            {
                "dataset": row["variant"],
                "model": row["model"],
                "accuracy": row["accuracy"],
                "f1": row["f1"],
                "auc": row["auc"],
                "fit_seconds": row["fit_seconds"],
                "predict_seconds": row["predict_seconds"],
                "error": row["error"],
            }
            for row in all_results
        ]
    )

    save_results_csv(
        all_results,
        args.output_dir / "all_tasks_results.csv",
    )
    print()
    print(f"Saved task-separated results under: {args.output_dir}")
