"""Minimal example to get the TabArena data and tasks without the TabArena framework.

To run this code, you only need to install `openml`.
    pip install openml
"""

from __future__ import annotations

import openml
from tabicl import TabICLClassifier
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import scipy.stats
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def plot_highlighted_embeddings(embeddings, labels, title, filename, top_k_indices, bottom_k_indices, k, save_directory='representation_visualizations'):
    """Helper function to plot embeddings with highlighted top/bottom k samples."""
    plt.figure(figsize=(10, 8))
    

    # Plot all points color-coded by label
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], s=20, alpha=0.5, c=labels, cmap='viridis', label='All samples')
    # Highlight top-k indices with their class colors
    plt.scatter(
        embeddings[top_k_indices, 0],
        embeddings[top_k_indices, 1],
        s=100,
        alpha=0.9,
        c=labels[top_k_indices],
        cmap=scatter.cmap,
        norm=scatter.norm,
        marker='*',
        edgecolors='black',
        linewidths=1.2,
        label=f'Top {k} attention'
    )
    # Highlight bottom-k indices with their class colors
    plt.scatter(
        embeddings[bottom_k_indices, 0],
        embeddings[bottom_k_indices, 1],
        s=100,
        alpha=0.9,
        c=labels[bottom_k_indices],
        cmap=scatter.cmap,
        norm=scatter.norm,
        marker='s',
        edgecolors='black',
        linewidths=1.2,
        label=f'Bottom {k} attention'
    )
    plt.colorbar(scatter, label='Class Label')
    plt.title(title)
    plt.xlabel(filename.split('_')[1].capitalize() + (' Dimension 1' if 'tsne' in filename else ' Component 1'))
    plt.ylabel(filename.split('_')[1].capitalize() + (' Dimension 2' if 'tsne' in filename else ' Component 2'))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_directory}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

def create_representation_plots(sorted_indices, k, representations, y_train, seed, save_directory = 'representation_visualizations'):

    top_k_indices = sorted_indices[-k:]
    bottom_k_indices = sorted_indices[:k]

    # Perform PCA
    pca = PCA(n_components=2)
    representations_pca = pca.fit_transform(representations)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    representations_tsne = tsne.fit_transform(representations)


    #Create directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Plot PCA
    plot_highlighted_embeddings(
        representations_pca, 
        y_train, 
        f'PCA of Representations (Highlighting Top/Bottom {k} Attention Samples)',
        f'representations_pca_k{k}_seed{seed}.png',
        top_k_indices,
        bottom_k_indices,
        k,
        save_directory=save_directory
    )

    # Plot t-SNE
    plot_highlighted_embeddings(
        representations_tsne, 
        y_train, 
        f't-SNE of Representations (Highlighting Top/Bottom {k} Attention Samples)',
        f'representations_tsne_k{k}_seed{seed}.png',
        top_k_indices,
        bottom_k_indices,
        k,
        save_directory=save_directory
    )

def train_and_score_model_on_subset(X_train, y_train, X_test, y_test, subset_indices=None, n_estimators=1, random_state=0, pos_label='yes'):

    if subset_indices is None:
        X_subset = X_train.reset_index(drop=True)
        y_subset = y_train.reset_index(drop=True)
    else:
        X_subset = X_train.iloc[subset_indices].reset_index(drop=True)
        y_subset = y_train.iloc[subset_indices].reset_index(drop=True)

    classifier = TabICLClassifier(n_estimators=n_estimators, random_state=random_state)
    classifier.fit(X_subset, y_subset)
        
    #Calculate both ROC AUC and F1 Score
    y_probs = classifier.predict_proba(X_test)
    y_pred = np.argmax(y_probs, axis=1)
    y_pred = classifier.y_encoder_.inverse_transform(y_pred)

    #If check if y_subset includes both classes, otherwise fix the output of y_probs
    unique_classes = np.unique(y_subset)
    
    if len(unique_classes) < 2:
        print("Warning: Only one class present in y_subset. Adjusting probabilities for ROC AUC calculation.")
        #If only majority class is present, set probabilities for minority class to 0
        if unique_classes[0] == pos_label:
            y_probs = np.column_stack((1 - y_probs[:, 0], y_probs[:, 0]))
        else:
            y_probs = np.column_stack((y_probs[:, 0], 1 - y_probs[:, 0]))

    roc_auc = roc_auc_score(y_test, y_probs[:, 1])    
    f1 = f1_score(y_test, y_pred, zero_division=0, pos_label=pos_label, average='binary')

    return f1, roc_auc

def estimate_attention_similarities(attention_matrix, subsample_size=100, seed=0, metric='kendall_tau'):
    """Evaluate attention similarities using the kendall tau correlation coefficient."""

    if attention_matrix.shape[0] > subsample_size:
        #Subsample rows for faster computation
        r = np.random.RandomState(seed)
        sampled_row_indices = r.choice(attention_matrix.shape[0], size=subsample_size, replace=False)

    n = len(sampled_row_indices)
    #Measure the differences between pairs of vectors
    if metric == 'cosine_sim':
        sampled_rows = attention_matrix[sampled_row_indices]
        similarity_matrix = cosine_similarity(sampled_rows)
    elif metric == 'kl_div':
        #Not implemented yet
        raise NotImplementedError("KL divergence metric not implemented yet.")
    else:
        sorted_indices = np.argsort(attention_matrix, axis=1)
        sorted_indices = sorted_indices[sampled_row_indices]
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if metric == 'kendall_tau':
                    tau, _ = scipy.stats.kendalltau(sorted_indices[i], sorted_indices[j])
                    similarity_matrix[i, j] = tau
                    similarity_matrix[j, i] = tau
                elif metric == 'spearman':
                    rho, _ = scipy.stats.spearmanr(sorted_indices[i], sorted_indices[j])
                    similarity_matrix[i, j] = rho
                    similarity_matrix[j, i] = rho
                else:
                    raise ValueError(f"Unknown metric: {metric}")

    #Calculate average similarity
    average_sim = np.mean(similarity_matrix[np.triu_indices(n, k=1)])
    return average_sim

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="TabArena Evaluation with TabICL")
    parser.add_argument('--lite', action='store_true', help='Use TabArena-Lite version (first repeat of first fold only)')
    parser.add_argument('--k', type=float, default=0.1, help='Proportion of samples to train on based on attention (default: 0.1)')
    parser.add_argument('--visualize', action='store_true', help='Whether to create representation visualizations')
    parser.add_argument('--save_attention_matrices', action='store_true', help='Whether to save test-to-train attention matrices')
    parser.add_argument('--stratified_subsampling', action='store_true', help='Whether to use stratified subsampling when limiting training samples')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # -- Parameters
    tabarena_version = "tabarena-v0.1"
    """The version of the TabArena benchmark suite to use."""
    tabarena_lite = args.lite
    """If True, will use the TabArena-Lite version of the benchmark suite.
    That is, only the first repeat of the first fold of each task will be used."""

    # -- Get Data
    benchmark_suite = openml.study.get_suite("tabarena-v0.1")
    task_ids = benchmark_suite.tasks

    # Iterate over all data and outer cross-validation splits from TabArena(-Lite)
    print("Getting Data for TabArena tasks...")
    if tabarena_lite:
        print("TabArena Lite is enabled. Getting first repeat of first fold for each task.")

    k = args.k
    print(f"Keeping top-{int(k * 100)}% and bottom-{int(k * 100)}% samples based on average attention.")
    save_attention_matrices = args.save_attention_matrices
    stratified_subsampling = args.stratified_subsampling

    if stratified_subsampling:
        print("Using stratified subsampling based on class labels when limiting training samples.")
        output_dir = 'tabarena_results/stratified_subsampling'
    else:
        output_dir = 'tabarena_results/standard_subsampling'

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        if dataset.qualities['NumberOfClasses'] != 2:
            print("Skipping regression or multi-class task.")
            continue

        if dataset.qualities['NumberOfFeatures'] > 500:
            print("Skipping dataset with more than 500 features.")
            continue
        
        print(f"Task ID: {task.id}, Dataset ID: {dataset.id}, Dataset Name: {dataset.name}")
        
        experiments_to_skip = []
        if dataset.name in experiments_to_skip:
            print(f"Skipping {dataset.name} for now.")
            continue

        # Get the number of folds and repeats used in TabArena
        if tabarena_lite:
            folds = 1
            tabarena_repeats = 1
        else:
            _, folds, _ = task.get_split_dimensions()
            n_samples = dataset.qualities["NumberOfInstances"]
            if n_samples < 2_500:
                tabarena_repeats = 10
            elif n_samples > 250_000:
                tabarena_repeats = 1
            else:
                tabarena_repeats = 3
        print(f"TabArena Repeats: {tabarena_repeats} | Folds: {folds}")
        
        methods = ['full_set', 'top_k', 'bot_k', 
               'random_k', 'weighted_random_k', 'inversly_weighted_random_k']

        results_dict = {
            **{method: {'f1s': [[] for _ in range(tabarena_repeats)], 'roc_aucs': [[] for _ in range(tabarena_repeats)]} for method in methods},
            'cosine_similarities': [[] for _ in range(tabarena_repeats)],
            'test_to_train_attention_matrices': [],
            'metadata': {
                'task_id': task.id,
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'k': k,
                'tabarena_lite': tabarena_lite,
                'folds': folds,
                'tabarena_repeats': tabarena_repeats,
                'max_train_samples': 2000,
                'max_test_samples': 500,
                'n_estimators': 1,
                'dataset_qualities': {
                    'n_instances': dataset.qualities.get('NumberOfInstances'),
                    'n_features': dataset.qualities.get('NumberOfFeatures'),
                    'n_classes': dataset.qualities.get('NumberOfClasses'),
                    'majority_class_size': dataset.qualities.get('MajorityClassSize'),
                    'minority_class_size': dataset.qualities.get('MinorityClassSize'),
                }
            }
        }

        if not os.path.exists(f'{output_dir}/{dataset.name}'):
            os.makedirs(f'{output_dir}/{dataset.name}')
        
        # Load the data for each split
        for repeat in range(tabarena_repeats):
            for fold in range(folds):
                seed = (fold + 1) * (repeat + 1)
                r = np.random.RandomState(seed)
                visualize = args.visualize and (repeat == 0 and fold == 0)
                save_attention = save_attention_matrices and (repeat == 0 and fold == 0)

                X, y, categorical_indicator, attribute_names = dataset.get_data(
                    target=task.target_name, dataset_format="dataframe"
                )

                train_indices, test_indices = task.get_train_test_split_indices(fold=fold, repeat=repeat)
                X_train = X.iloc[train_indices]
                y_train = y.iloc[train_indices]
                X_test = X.iloc[test_indices]
                y_test = y.iloc[test_indices]

                #Get positive class label
                pos_class = task.class_labels[1]  # Assuming binary classification with two classes
                
                if pos_class == 'True':
                    pos_class = True

                max_train_samples = 2000
                max_test_samples = 500
                #Sample only max_samples from train and test sets
                if X_train.shape[0] > max_train_samples:
                    train_indices = X_train.sample(n=max_train_samples, random_state=seed).index
                    X_train = X_train.loc[train_indices].reset_index(drop=True)
                    y_train = y_train.loc[train_indices].reset_index(drop=True)
                    
                if X_test.shape[0] > max_test_samples:
                    test_indices = X_test.sample(n=max_test_samples, random_state=seed).index
                    X_test = X_test.loc[test_indices].reset_index(drop=True)
                    y_test = y_test.loc[test_indices].reset_index(drop=True)


                clf = TabICLClassifier(n_estimators=1, random_state=seed)
                clf.fit(X_train, y_train)  # this is cheap

                #res, rollout = clf.predict_with_rollout(X_test, return_row_emb=True)
                probs, rollout = clf.predict_proba_with_rollout(X_test, return_row_emb=visualize)

                preds = np.argmax(probs, axis=1)
                res = clf.y_encoder_.inverse_transform(preds)

                full_set_f1 = f1_score(y_test, res, zero_division=0, pos_label=pos_class)
                full_set_roc_auc = roc_auc_score(y_test, probs[:, 1])

                #We take the first element because n_estimators=1. If this is higher you get one rollout per estimator.
                #I realize that this might have an effect on the 'row_emb_rollout' since they permute the columns, but
                #it should not affect the 'icl' rollout.
                rollout_icl = rollout['icl_rollout'][0]

                #Average attention on training samples by test samples
                #The batch size is 1 in our case, so we take the first element
                test_to_train_attention = rollout_icl[0, X_train.shape[0]:, :X_train.shape[0]]
                average_attention = np.mean(test_to_train_attention, axis=0)
                sorted_indices = np.argsort(average_attention)
                
                if save_attention:
                    results_dict['test_to_train_attention_matrices'].append(test_to_train_attention)

                metric = 'cosine_sim'
                mean_sim = estimate_attention_similarities(test_to_train_attention, subsample_size=100, seed=seed, metric=metric)
                results_dict['cosine_similarities'][repeat].append(mean_sim)


                #Get binary encoded y_train
                y_train_binary = np.array([1 if label == pos_class else 0 for label in y_train])
                
                if visualize:
                    create_representation_plots(sorted_indices, 20, rollout['row_embeddings'][0][0][:X_train.shape[0]], y_train_binary, seed, save_directory=f'{output_dir}/{dataset.name}/representation_visualizations')

                avg_attention_and_label = list(zip(average_attention, y_train_binary))

                if stratified_subsampling:
                    #Separate positive and negative samples
                    sorted_positive = sorted([(i, att) for i, (att, label) in enumerate(avg_attention_and_label) if label == 1], key=lambda x: x[1], reverse=True)
                    sorted_negative = sorted([(i, att) for i, (att, label) in enumerate(avg_attention_and_label) if label != 1], key=lambda x: x[1], reverse=True)

                    top_k_positive_indices = [idx for idx, _ in sorted_positive[:int(k * len(sorted_positive))]]
                    top_k_negative_indices = [idx for idx, _ in sorted_negative[:int(k * len(sorted_negative))]]

                    bottom_k_positive_indices = [idx for idx, _ in sorted_positive[-int(k * len(sorted_positive)):]]
                    bottom_k_negative_indices = [idx for idx, _ in sorted_negative[-int(k * len(sorted_negative)):]]

                    top_k_indices = np.array(top_k_positive_indices + top_k_negative_indices)
                    bot_k_indices = np.array(bottom_k_positive_indices + bottom_k_negative_indices)

                    # Stratified random variants: sample positives and negatives separately to preserve balance
                    pos_indices = np.array([i for i, lbl in enumerate(y_train_binary) if lbl == 1])
                    neg_indices = np.array([i for i, lbl in enumerate(y_train_binary) if lbl == 0])

                    pos_sample_size = min(len(pos_indices), max(1, int(k * len(pos_indices))))
                    neg_sample_size = min(len(neg_indices), max(1, int(k * len(neg_indices))))

                    # Uniform stratified random
                    random_pos = r.choice(pos_indices, size=pos_sample_size, replace=False)
                    random_neg = r.choice(neg_indices, size=neg_sample_size, replace=False)
                    random_k_indices = np.concatenate([random_pos, random_neg])

                    # Attention-weighted stratified random
                    att_pos = average_attention[pos_indices]
                    att_neg = average_attention[neg_indices]
                    prob_pos = att_pos / att_pos.sum()
                    prob_neg = att_neg / att_neg.sum()
                    weighted_pos = r.choice(pos_indices, size=pos_sample_size, replace=False, p=prob_pos)
                    weighted_neg = r.choice(neg_indices, size=neg_sample_size, replace=False, p=prob_neg)
                    weighted_random_k_indices = np.concatenate([weighted_pos, weighted_neg])

                    # Inverse-attention-weighted stratified random
                    inv_att_pos = 1 / att_pos
                    inv_att_neg = 1 / att_neg
                    prob_pos_inv = inv_att_pos / inv_att_pos.sum()
                    prob_neg_inv = inv_att_neg / inv_att_neg.sum()
                    inverse_weighted_pos = r.choice(pos_indices, size=pos_sample_size, replace=False, p=prob_pos_inv)
                    inverse_weighted_neg = r.choice(neg_indices, size=neg_sample_size, replace=False, p=prob_neg_inv)
                    inverse_weighted_random_k_indices = np.concatenate([inverse_weighted_pos, inverse_weighted_neg])

                else:
                    top_k_indices = sorted_indices[-int(k * X_train.shape[0]):]
                    bot_k_indices = sorted_indices[:int(k * X_train.shape[0])]

                    random_k_indices = r.choice(X_train.shape[0], size=top_k_indices.shape[0], replace=False)
                
                    #Sample random indices with probability proportional to average attention (more likely to sample high attention)
                    attention_probabilities = average_attention / np.sum(average_attention)
                    weighted_random_k_indices = r.choice(X_train.shape[0], size=top_k_indices.shape[0], replace=False, p=attention_probabilities)

                    #Sample random indices with probability INVERSLY proportional to average attention (more likely to sample low attention)
                    weights = 1 / average_attention
                    inverse_attention_probabilities = weights / np.sum(weights)
                    inverse_weighted_random_k_indices = r.choice(X_train.shape[0], size=top_k_indices.shape[0], replace=False, p=inverse_attention_probabilities)

                top_k_f1, top_k_roc_auc = train_and_score_model_on_subset(
                    X_train, y_train, X_test, y_test, subset_indices=top_k_indices, n_estimators=1, random_state=0, pos_label=pos_class
                )
                bot_k_f1, bot_k_roc_auc = train_and_score_model_on_subset(
                    X_train, y_train, X_test, y_test, subset_indices=bot_k_indices, n_estimators=1, random_state=0, pos_label=pos_class
                )
                random_k_f1, random_k_roc_auc = train_and_score_model_on_subset(
                    X_train, y_train, X_test, y_test, subset_indices=random_k_indices, n_estimators=1, random_state=0, pos_label=pos_class
                )

                weighted_random_k_f1, weighted_random_k_roc_auc = train_and_score_model_on_subset(
                    X_train, y_train, X_test, y_test, subset_indices=weighted_random_k_indices, n_estimators=1, random_state=0, pos_label=pos_class
                )


                inversly_weighted_random_k_f1, inversly_weighted_random_k_roc_auc = train_and_score_model_on_subset(
                    X_train, y_train, X_test, y_test, subset_indices=inverse_weighted_random_k_indices, n_estimators=1, random_state=0, pos_label=pos_class
                )

                #Store results
                results_dict['full_set']['f1s'][repeat].append(full_set_f1)
                results_dict['top_k']['f1s'][repeat].append(top_k_f1)
                results_dict['bot_k']['f1s'][repeat].append(bot_k_f1)
                results_dict['random_k']['f1s'][repeat].append(random_k_f1)
                results_dict['weighted_random_k']['f1s'][repeat].append(weighted_random_k_f1)
                results_dict['inversly_weighted_random_k']['f1s'][repeat].append(inversly_weighted_random_k_f1)
                results_dict['full_set']['roc_aucs'][repeat].append(full_set_roc_auc)
                results_dict['top_k']['roc_aucs'][repeat].append(top_k_roc_auc)
                results_dict['bot_k']['roc_aucs'][repeat].append(bot_k_roc_auc)
                results_dict['random_k']['roc_aucs'][repeat].append(random_k_roc_auc)
                results_dict['weighted_random_k']['roc_aucs'][repeat].append(weighted_random_k_roc_auc)
                results_dict['inversly_weighted_random_k']['roc_aucs'][repeat].append(inversly_weighted_random_k_roc_auc)
        
        #save results_dict
        with open(f'{output_dir}/{dataset.name}/results.pkl', 'wb') as f:
            pickle.dump(results_dict, f)