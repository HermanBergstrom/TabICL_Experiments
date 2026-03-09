"""Minimal example to get the TabArena data and tasks without the TabArena framework.

To run this code, you only need to install `openml`.
    pip install openml
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import openml

from tabicl_v2_attention.analysis import build_refit_on_hit_subset_stats, compute_full_model_stats
from tabicl_v2_attention.common import to_label_array
from tabicl_v2_attention.modeling import fit_full_model_and_extract_attention, train_and_score_model_on_subset
from tabicl_v2_attention.subsampling import (
    build_classwise_clusters,
    compute_cluster_hit_statistics,
    compute_cluster_attention_summary,
    plot_cluster_hit_histogram,
    plot_cluster_hit_histogram_per_class,
    select_by_classwise_cluster_attention_weighted,
    select_by_classwise_cluster_uniform_cluster_attention_row,
    sample_rows_from_top_attention_clusters,
    select_by_classwise_cluster_hit_proportional,
    select_by_classwise_cluster_representatives,
    select_subset_indices,
)
from tabicl_v2_attention.viz import (
    plot_embedding_attention_suite,
    plot_hit_count_distribution,
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="TabArena Evaluation with TabICL")
    parser.add_argument('--lite', action='store_true', help='Use TabArena-Lite version (first repeat of first fold only)')
    parser.add_argument('--k', type=float, default=0.1, help='Subset size control: 0<k<=1 means fraction of train set, k>1 means absolute number of samples (default: 0.1)')
    parser.add_argument('--attn_batch_size', type=int, default=224, help='Batch size used when collecting attention maps on test data')
    parser.add_argument('--attn_aggregation', type=str, default='sum', choices=['mean', 'sum'], help='How to aggregate attention over training examples per class')
    parser.add_argument('--hit_count_mode', type=str, default='predicted_class', choices=['predicted_class', 'all_classes'], help="How to count attention hits: 'predicted_class' (current behavior) or 'all_classes' (one hit per class per test sample)")
    parser.add_argument(
        '--subsampling_strategy',
        type=str,
        default='topk_attn_sum',
        choices=[
            'topk_attn_sum',
            'weighted_attn_sum',
            'cluster_attn_mass',
            'cluster_class_representative',
            'cluster_class_hit_proportional',
            'cluster_class_attention_weighted',
            'cluster_class_uniform_cluster_attention_row',
        ],
        help='Attention-based subsampling strategy for selecting training indices (a random baseline of the same size is always run alongside for comparison)',
    )
    parser.add_argument('--cluster_n', type=int, default=200, help='Number of clusters used by cluster_attn_mass strategy')
    parser.add_argument('--cluster_selection_fraction', type=float, default=0.01, help='Per-class cluster fraction used by cluster_class_hit_proportional strategy (0,1]')
    parser.add_argument('--visualize_cluster_hits', action='store_true', help='Visualize cluster hit-rate histograms using class-wise clustering from full-model attention maps')
    parser.add_argument('--cluster_hit_fraction', type=float, default=0.1, help='Per-class cluster count fraction used for cluster-hit visualization (0,1]')
    parser.add_argument('--visualize_pca_hits', action='store_true', help='Create a PCA+t-SNE comparison plot of train embeddings with per-class highest-attention samples highlighted')
    parser.add_argument('--pca_top_k_per_class', type=int, default=1, help='Number of highest-attention train samples to highlight per class in PCA plot')
    parser.add_argument('--embedding_plot_subset_size', type=int, default=0, help='Optional number of train samples to include in PCA+t-SNE visualization (0 means all samples)')
    parser.add_argument('--top_attended_fraction', type=float, default=0.1, help='Top fraction of attended training points to highlight in attention visualization plots (default: 0.1)')
    parser.add_argument('--n_test_attention_plots', type=int, default=8, help='Number of test-sample attention neighborhood plots to generate (default: 4)')
    parser.add_argument('--visualize_hit_distribution', action='store_true', help='Create hit-count distribution plot showing proportion of train samples at each hit count')
    parser.add_argument('--refit_on_hit_subset', action='store_true', help='Refit a second model after removing all training items with zero hits and compare performance')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tabarena_lite = args.lite

    benchmark_suite = openml.study.get_suite("tabarena-v0.1")
    task_ids = benchmark_suite.tasks

    print("Getting Data for TabArena tasks...")
    if tabarena_lite:
        print("TabArena Lite is enabled. Getting first repeat of first fold for each task.")

    k = args.k
    if k <= 0:
        raise ValueError(f"--k must be > 0, got {k}")
    if k > 1:
        print(f"Using fixed subset size k={int(k)} samples for the subset refit model.")
    else:
        print(f"Using k={k:.4f} ({int(k * 100)}%) of samples for the subset refit model.")

    attn_batch_size = args.attn_batch_size
    attn_aggregation = args.attn_aggregation
    hit_count_mode = args.hit_count_mode
    subsampling_strategy = args.subsampling_strategy
    cluster_n = args.cluster_n
    cluster_selection_fraction = args.cluster_selection_fraction
    visualize_cluster_hits = args.visualize_cluster_hits
    cluster_hit_fraction = args.cluster_hit_fraction
    visualize_pca_hits = args.visualize_pca_hits
    pca_top_k_per_class = args.pca_top_k_per_class
    embedding_plot_subset_size = args.embedding_plot_subset_size
    top_attended_fraction = args.top_attended_fraction
    n_test_attention_plots = args.n_test_attention_plots
    visualize_hit_distribution = args.visualize_hit_distribution
    refit_on_hit_subset = args.refit_on_hit_subset

    if embedding_plot_subset_size < 0:
        raise ValueError(f"--embedding_plot_subset_size must be >= 0, got {embedding_plot_subset_size}")
    if not (0 < top_attended_fraction <= 1):
        raise ValueError(f"--top_attended_fraction must be in (0, 1], got {top_attended_fraction}")
    if n_test_attention_plots < 0:
        raise ValueError(f"--n_test_attention_plots must be >= 0, got {n_test_attention_plots}")

    print(
        "Using class-stratified subset selection; for topk/weighted strategies, attention scores are label-matched "
        "(test label == train label)."
    )
    output_base_dir = f'results/v2_attention_results/{subsampling_strategy}'

    print(f"Using subsampling strategy: {subsampling_strategy}")
    if subsampling_strategy in {
        'cluster_class_hit_proportional',
        'cluster_class_attention_weighted',
        'cluster_class_uniform_cluster_attention_row',
    }:
        print(f"Using cluster_selection_fraction: {cluster_selection_fraction}")
    print(f"Using hit_count_mode: {hit_count_mode}")

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        if dataset.qualities['NumberOfClasses'] != 2:
            print("Skipping regression or multi-class task.")
            continue

        if dataset.qualities['NumberOfFeatures'] > 500:
            print("Skipping dataset with more than 500 features.")
            continue

        if dataset.qualities.get('MinorityClassSize') / (dataset.qualities.get('MajorityClassSize') + dataset.qualities.get('MinorityClassSize')) < 0.1:
            print("Skipping highly imbalanced dataset.")
            continue

        print(f"Task ID: {task.id}, Dataset ID: {dataset.id}, Dataset Name: {dataset.name}")

        if tabarena_lite:
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

        print(f"TabArena Repeats: {tabarena_repeats} | Folds: {folds}")

        results_dict = {
            'metadata': {
                'task_id': task.id,
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'k': k,
                'tabarena_lite': tabarena_lite,
                'folds': folds,
                'tabarena_repeats': tabarena_repeats,
                'max_train_samples': 10000,
                'max_test_samples': 5000,
                'n_estimators': 1,
                'attn_batch_size': attn_batch_size,
                'attn_aggregation': attn_aggregation,
                'hit_count_mode': hit_count_mode,
                'subsampling_strategy': subsampling_strategy,
                'cluster_n': cluster_n,
                'cluster_selection_fraction': cluster_selection_fraction,
                'visualize_cluster_hits': visualize_cluster_hits,
                'cluster_hit_fraction': cluster_hit_fraction,
                'embedding_plot_subset_size': embedding_plot_subset_size,
                'top_attended_fraction': top_attended_fraction,
                'n_test_attention_plots': n_test_attention_plots,
                'dataset_qualities': {
                    'n_instances': dataset.qualities.get('NumberOfInstances'),
                    'n_features': dataset.qualities.get('NumberOfFeatures'),
                    'n_classes': dataset.qualities.get('NumberOfClasses'),
                    'majority_class_size': dataset.qualities.get('MajorityClassSize'),
                    'minority_class_size': dataset.qualities.get('MinorityClassSize'),
                },
            },
            'full_model': {
                'accuracies': [[] for _ in range(tabarena_repeats)],
                'f1s': [[] for _ in range(tabarena_repeats)],
                'roc_aucs': [[] for _ in range(tabarena_repeats)],
                'stats': [[] for _ in range(tabarena_repeats)],
            },
        }

        results_dict['k_fraction'] = {
            'accuracies': [[] for _ in range(tabarena_repeats)],
            'f1s': [[] for _ in range(tabarena_repeats)],
            'roc_aucs': [[] for _ in range(tabarena_repeats)],
            'refit_on_hit_subset': [[] for _ in range(tabarena_repeats)],
        }
        results_dict['k_fraction_random_baseline'] = {
            'accuracies': [[] for _ in range(tabarena_repeats)],
            'f1s': [[] for _ in range(tabarena_repeats)],
            'roc_aucs': [[] for _ in range(tabarena_repeats)],
        }

        output_dir = f'{output_base_dir}/{dataset.name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for repeat in range(tabarena_repeats):
            for fold in range(folds):
                seed = (fold + 1) * (repeat + 1)
                rng = np.random.RandomState(seed)

                x, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
                train_indices, test_indices = task.get_train_test_split_indices(fold=fold, repeat=repeat)
                x_train = x.iloc[train_indices]
                y_train = y.iloc[train_indices]
                x_test = x.iloc[test_indices]
                y_test = y.iloc[test_indices]

                pos_class = task.class_labels[1]
                if pos_class == 'True':
                    pos_class = True

                max_train_samples = 10000
                max_test_samples = 5000
                if x_train.shape[0] > max_train_samples:
                    train_sample_idx = x_train.sample(n=max_train_samples, random_state=seed).index
                    x_train = x_train.loc[train_sample_idx].reset_index(drop=True)
                    y_train = y_train.loc[train_sample_idx].reset_index(drop=True)

                if x_test.shape[0] > max_test_samples:
                    test_sample_idx = x_test.sample(n=max_test_samples, random_state=seed).index
                    x_test = x_test.loc[test_sample_idx].reset_index(drop=True)
                    y_test = y_test.loc[test_sample_idx].reset_index(drop=True)

                y_train_binary = np.array([1 if label == pos_class else 0 for label in y_train])
                train_labels = to_label_array(y_train)

                print(
                    f"Fitting full model | Repeat {repeat+1}/{tabarena_repeats}, "
                    f"Fold {fold+1}/{folds} ..."
                )

                train_embeddings_required = (
                    visualize_pca_hits
                    or visualize_cluster_hits
                    or subsampling_strategy in {
                        'cluster_attn_mass',
                        'cluster_class_representative',
                        'cluster_class_hit_proportional',
                        'cluster_class_attention_weighted',
                        'cluster_class_uniform_cluster_attention_row',
                    }
                )
                test_embeddings_required = visualize_pca_hits

                classifier_full, x_test_transformed_full, attn_maps_full, train_row_embeddings_full, test_row_embeddings_full = fit_full_model_and_extract_attention(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    n_estimators=1,
                    random_state=seed,
                    attn_batch_size=attn_batch_size,
                    extract_train_embeddings=train_embeddings_required,
                    extract_test_embeddings_flag=test_embeddings_required,
                )

                hit_counts, attn_sums_per_class, full_model_stats = compute_full_model_stats(
                    classifier=classifier_full,
                    x_test_transformed=x_test_transformed_full,
                    attn_maps=attn_maps_full,
                    train_labels=train_labels,
                    y_test=y_test,
                    attn_aggregation=attn_aggregation,
                    pos_label=pos_class,
                    train_row_embeddings=train_row_embeddings_full,
                    hit_count_mode=hit_count_mode,
                )

                results_dict['full_model']['accuracies'][repeat].append(full_model_stats['accuracy'])
                results_dict['full_model']['f1s'][repeat].append(full_model_stats['f1'])
                results_dict['full_model']['roc_aucs'][repeat].append(full_model_stats['roc_auc'])
                results_dict['full_model']['stats'][repeat].append(full_model_stats)

                print(
                    f"Full model | Repeat {repeat+1}/{tabarena_repeats}, Fold {fold+1}/{folds} | "
                    f"ROC AUC: {full_model_stats['roc_auc']:.4f} | F1: {full_model_stats['f1']:.4f} | Acc: {full_model_stats['accuracy']:.4f} | "
                    f"Attn agg: {attn_aggregation} | "
                    f"MeanAttn-vs-pred: {full_model_stats['mean_attn_vs_pred_acc']:.3f} | "
                    f"MeanAttn-vs-true: {full_model_stats['mean_attn_vs_true_acc']:.3f} | "
                    f"Pred-vs-true: {full_model_stats['pred_vs_true_acc']:.3f}"
                )

                outlier_stats = full_model_stats.get('attention_outlier_analysis', {})
                if outlier_stats.get('enabled'):
                    top5 = outlier_stats.get('top_slices', {}).get('top_5pct', {})
                    dens_quint = outlier_stats.get('density_quintile_attention', {})
                    outlier_q_share = dens_quint.get('top_20pct_most_isolated_attention_share', float('nan'))
                    dens_quint_pc = outlier_stats.get('density_quintile_attention_per_class', {})
                    outlier_q_share_pc_w = dens_quint_pc.get(
                        'top_20pct_most_isolated_attention_share_weighted_mean',
                        float('nan'),
                    )
                    print(
                        f"Outlier-attn | top5% knn-dist/global={top5.get('knn_distance_ratio_vs_global', float('nan')):.3f} | "
                        f"top5% density/global={top5.get('density_ratio_vs_global', float('nan')):.3f} | "
                        f"attn_share_q80_100={outlier_q_share:.3f} | "
                        f"attn_share_q80_100_per_class_w={outlier_q_share_pc_w:.3f} | "
                        f"spearman_global={outlier_stats.get('spearman_attn_vs_knn_distance', float('nan')):.3f} | "
                        f"spearman_per_class_mean={outlier_stats.get('spearman_attn_vs_knn_distance_per_class_mean', float('nan')):.3f} | "
                        f"spearman_per_class_weighted={outlier_stats.get('spearman_attn_vs_knn_distance_per_class_weighted_mean', float('nan')):.3f}"
                    )

                if visualize_pca_hits:
                    plot_embedding_attention_suite(
                        train_representations=train_row_embeddings_full,
                        train_labels=train_labels,
                        hit_counts=hit_counts,
                        attn_maps=attn_maps_full,
                        test_representations=test_row_embeddings_full,
                        test_labels=to_label_array(y_test),
                        top_k_per_class=pca_top_k_per_class,
                        top_fraction=top_attended_fraction,
                        subset_size=embedding_plot_subset_size,
                        n_test_samples=n_test_attention_plots,
                        random_state=seed,
                        comparison_output_path=(
                            f"{output_dir}/pca_hits/"
                            f"repeat{repeat+1}_fold{fold+1}.png"
                        ),
                        top_overall_output_path=(
                            f"{output_dir}/top_overall_attended/"
                            f"repeat{repeat+1}_fold{fold+1}.png"
                        ),
                        top_per_class_output_dir=(
                            f"{output_dir}/top_per_class_attended/"
                            f"repeat{repeat+1}_fold{fold+1}"
                        ),
                        test_neighborhood_output_dir=(
                            f"{output_dir}/test_sample_attended_neighborhoods/"
                            f"repeat{repeat+1}_fold{fold+1}"
                        ),
                        comparison_title=(
                            f"{dataset.name} | repeat {repeat+1}/{tabarena_repeats} | "
                            f"fold {fold+1}/{folds} | full model PCA+t-SNE"
                        ),
                        top_overall_title=(
                            f"{dataset.name} | repeat {repeat+1}/{tabarena_repeats} | "
                            f"fold {fold+1}/{folds} | top overall attended train samples"
                        ),
                        top_per_class_title=(
                            f"{dataset.name} | repeat {repeat+1}/{tabarena_repeats} | "
                            f"fold {fold+1}/{folds} | top attended train samples per class"
                        ),
                        test_title_prefix=(
                            f"{dataset.name} | repeat {repeat+1}/{tabarena_repeats} | "
                            f"fold {fold+1}/{folds}"
                        ),
                    )

                if visualize_hit_distribution:
                    plot_hit_count_distribution(
                        hit_counts=hit_counts,
                        output_path=(
                            f"{output_dir}/hit_count_distribution/"
                            f"repeat{repeat+1}_fold{fold+1}.png"
                        ),
                        title=(
                            f"{dataset.name} | repeat {repeat+1}/{tabarena_repeats} | "
                            f"fold {fold+1}/{folds} | full model hit count distribution"
                        ),
                    )

                if visualize_cluster_hits:
                    cluster_summary_vis = build_classwise_clusters(
                        train_row_embeddings=train_row_embeddings_full,
                        train_labels=train_labels,
                        cluster_fraction=cluster_hit_fraction,
                        random_state=seed,
                    )
                    cluster_hit_stats = compute_cluster_hit_statistics(
                        attn_maps=attn_maps_full,
                        cluster_summary=cluster_summary_vis,
                    )
                    plot_cluster_hit_histogram(
                        cluster_hit_stats=cluster_hit_stats,
                        output_path=(
                            f"{output_dir}/cluster_hit_histograms/"
                            f"repeat{repeat+1}_fold{fold+1}.png"
                        ),
                        title=(
                            f"{dataset.name} | repeat {repeat+1}/{tabarena_repeats} | "
                            f"fold {fold+1}/{folds} | cluster hit rates"
                        ),
                    )
                    plot_cluster_hit_histogram_per_class(
                        cluster_hit_stats=cluster_hit_stats,
                        output_path=(
                            f"{output_dir}/cluster_hit_histograms_per_class/"
                            f"repeat{repeat+1}_fold{fold+1}.png"
                        ),
                        title=(
                            f"{dataset.name} | repeat {repeat+1}/{tabarena_repeats} | "
                            f"fold {fold+1}/{folds} | cluster hit rates per class"
                        ),
                    )

                if refit_on_hit_subset:
                    refit_stats = build_refit_on_hit_subset_stats(
                        x_subset=x_train.reset_index(drop=True),
                        y_subset=y_train.reset_index(drop=True),
                        x_test=x_test,
                        y_test=y_test,
                        hit_counts=hit_counts,
                        n_estimators=1,
                        random_state=seed,
                        pos_label=pos_class,
                        full_accuracy=full_model_stats['accuracy'],
                        full_f1=full_model_stats['f1'],
                        full_roc_auc=full_model_stats['roc_auc'],
                        full_label_balance=full_model_stats['train_label_balance'],
                    )

                    accuracy = refit_stats['reduced_accuracy']
                    f1 = refit_stats['reduced_f1']
                    roc_auc = refit_stats['reduced_roc_auc']
                    random_accuracy = refit_stats['random_matched_accuracy']
                    random_f1 = refit_stats['random_matched_f1']
                    random_roc_auc = refit_stats['random_matched_roc_auc']

                    results_dict['k_fraction']['accuracies'][repeat].append(accuracy)
                    results_dict['k_fraction']['f1s'][repeat].append(f1)
                    results_dict['k_fraction']['roc_aucs'][repeat].append(roc_auc)
                    results_dict['k_fraction']['refit_on_hit_subset'][repeat].append(refit_stats)
                    results_dict['k_fraction_random_baseline']['accuracies'][repeat].append(random_accuracy)
                    results_dict['k_fraction_random_baseline']['f1s'][repeat].append(random_f1)
                    results_dict['k_fraction_random_baseline']['roc_aucs'][repeat].append(random_roc_auc)

                    print(
                        f"Subset model (hit-based override) | Repeat {repeat+1}/{tabarena_repeats}, Fold {fold+1}/{folds} | "
                        f"kept={refit_stats['kept_samples']}/{refit_stats['total_samples']} "
                        f"({refit_stats['compression_ratio']:.3f}) | "
                        f"reduced ROC AUC={roc_auc:.4f}, reduced F1={f1:.4f}, reduced Acc={accuracy:.4f} | "
                        f"random matched ROC AUC={random_roc_auc:.4f}, random matched F1={random_f1:.4f}, random matched Acc={random_accuracy:.4f} | "
                        f"Δ ROC AUC={roc_auc - random_roc_auc:+.4f}, ΔF1={f1 - random_f1:+.4f}"
                    )
                else:
                    if subsampling_strategy == 'cluster_attn_mass':
                        n_train = len(y_train_binary)
                        if k > 1:
                            n_subset = min(int(k), n_train)
                        else:
                            n_subset = max(1, int(k * n_train))

                        attn_received = np.asarray(attn_maps_full).sum(axis=0)
                        cluster_summary = compute_cluster_attention_summary(
                            train_row_embeddings=train_row_embeddings_full,
                            attention_weights=attn_received,
                            #n_clusters=cluster_n,
                            n_clusters=0.02 * n_train,  # set n_clusters as a fraction of train size to avoid too-small clusters on small datasets
                            random_state=seed,
                        )
                        subset_indices = sample_rows_from_top_attention_clusters(
                            cluster_summary=cluster_summary,
                            total_samples=n_subset,
                            rng=rng,
                        )
                    elif subsampling_strategy == 'cluster_class_representative':
                        attn_received = np.asarray(attn_maps_full).sum(axis=0)
                        subset_result = select_by_classwise_cluster_representatives(
                            train_row_embeddings=train_row_embeddings_full,
                            train_labels=train_labels,
                            attention_weights=attn_received,
                            k=k,
                            random_state=seed,
                        )
                        subset_indices = subset_result['subset_indices']
                    elif subsampling_strategy == 'cluster_class_hit_proportional':
                        attn_received = np.asarray(attn_maps_full).sum(axis=0)
                        subset_result = select_by_classwise_cluster_hit_proportional(
                            train_row_embeddings=train_row_embeddings_full,
                            train_labels=train_labels,
                            attention_weights=attn_received,
                            attn_maps=attn_maps_full,
                            k=k,
                            cluster_fraction=cluster_selection_fraction,
                            random_state=seed,
                        )
                        subset_indices = subset_result['subset_indices']
                    elif subsampling_strategy == 'cluster_class_attention_weighted':
                        subset_result = select_by_classwise_cluster_attention_weighted(
                            train_row_embeddings=train_row_embeddings_full,
                            train_labels=train_labels,
                            attn_sums_per_class=attn_sums_per_class,
                            k=k,
                            cluster_fraction=cluster_selection_fraction,
                            random_state=seed,
                        )
                        subset_indices = subset_result['subset_indices']
                    elif subsampling_strategy == 'cluster_class_uniform_cluster_attention_row':
                        subset_result = select_by_classwise_cluster_uniform_cluster_attention_row(
                            train_row_embeddings=train_row_embeddings_full,
                            train_labels=train_labels,
                            attn_sums_per_class=attn_sums_per_class,
                            k=k,
                            cluster_fraction=cluster_selection_fraction,
                            random_state=seed,
                        )
                        subset_indices = subset_result['subset_indices']
                    else:
                        subset_indices = select_subset_indices(
                            strategy=subsampling_strategy,
                            y_train_binary=y_train_binary,
                            rng=rng,
                            k=k,
                            current_n_samples=None,
                            attn_sums_per_class=attn_sums_per_class,
                            train_labels=train_labels,
                        )

                    random_baseline_rng = np.random.RandomState(seed + 1)
                    if subsampling_strategy == 'cluster_attn_mass':
                        random_subset_indices = random_baseline_rng.choice(
                            np.arange(len(y_train_binary)),
                            size=len(subset_indices),
                            replace=False,
                        )
                    else:
                        random_subset_indices = select_subset_indices(
                            strategy='random',
                            y_train_binary=y_train_binary,
                            rng=random_baseline_rng,
                            k=k,
                            current_n_samples=len(subset_indices),
                            attn_sums_per_class=None,
                            train_labels=train_labels,
                        )

                    accuracy, f1, roc_auc = train_and_score_model_on_subset(
                        x_train, y_train, x_test, y_test,
                        subset_indices=subset_indices,
                        n_estimators=1, random_state=seed, pos_label=pos_class,
                    )
                    random_accuracy, random_f1, random_roc_auc = train_and_score_model_on_subset(
                        x_train, y_train, x_test, y_test,
                        subset_indices=random_subset_indices,
                        n_estimators=1, random_state=seed, pos_label=pos_class,
                    )

                    results_dict['k_fraction']['accuracies'][repeat].append(accuracy)
                    results_dict['k_fraction']['f1s'][repeat].append(f1)
                    results_dict['k_fraction']['roc_aucs'][repeat].append(roc_auc)
                    results_dict['k_fraction_random_baseline']['accuracies'][repeat].append(random_accuracy)
                    results_dict['k_fraction_random_baseline']['f1s'][repeat].append(random_f1)
                    results_dict['k_fraction_random_baseline']['roc_aucs'][repeat].append(random_roc_auc)

                    if k > 1:
                        subset_desc = f"k={int(k)} ({len(subset_indices)} samples)"
                    else:
                        subset_desc = f"k={k:.4f} ({len(subset_indices)} samples)"

                    print(
                        f"Subset model | Repeat {repeat+1}/{tabarena_repeats}, Fold {fold+1}/{folds}, "
                        f"{subset_desc} | "
                        f"{subsampling_strategy}: ROC AUC={roc_auc:.4f}, F1={f1:.4f}, Acc={accuracy:.4f} | "
                        f"random baseline: ROC AUC={random_roc_auc:.4f}, F1={random_f1:.4f}, Acc={random_accuracy:.4f} | "
                        f"Δ ROC AUC={roc_auc - random_roc_auc:+.4f}, ΔF1={f1 - random_f1:+.4f}"
                    )

        with open(f'{output_dir}/results.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
