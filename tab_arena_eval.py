"""Minimal example to get the TabArena data and tasks without the TabArena framework.

To run this code, you only need to install `openml`.
    pip install openml
"""

from __future__ import annotations

import openml
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from skrub import TableVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

import os
import pickle

def train_and_score_model_on_subset(X_train, y_train, X_test, y_test, subset_indices=None, n_estimators=1, random_state=0, pos_label='yes', model_type='tab_icl'):

    if subset_indices is None:
        X_subset = X_train.reset_index(drop=True)
        y_subset = y_train.reset_index(drop=True)
    else:
        X_subset = X_train.iloc[subset_indices].reset_index(drop=True)
        y_subset = y_train.iloc[subset_indices].reset_index(drop=True)

    if model_type == 'tab_icl':
        classifier = TabICLClassifier(n_estimators=n_estimators, random_state=random_state)
    elif model_type == 'tab_pfn':
        classifier = TabPFNClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'tab_icl' or 'tab_pfn'.")
    
    # Create pipeline with TableVectorizer
    pipeline = make_pipeline(TableVectorizer(), classifier)
    pipeline.fit(X_subset, y_subset)
        
    #Calculate both ROC AUC and F1 Score
    y_probs = pipeline.predict_proba(X_test)
    y_pred = np.argmax(y_probs, axis=1)

    # Get the classifier from the pipeline
    clf = pipeline.steps[-1][1]
    
    if model_type == 'tab_icl':
        y_pred = clf.y_encoder_.inverse_transform(y_pred)
    elif model_type == 'tab_pfn':
        y_pred = clf.label_encoder_.inverse_transform(y_pred)

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

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="TabArena Evaluation with TabICL")
    parser.add_argument('--lite', action='store_true', help='Use TabArena-Lite version (first repeat of first fold only)')
    parser.add_argument('--k', type=float, default=0.1, help='Proportion of samples to train on based on attention (default: 0.1)')
    parser.add_argument('--n_samples', type=int, nargs='+', default=None, help='Exact number of training samples to use (can specify multiple values, e.g., --n_samples 50 100 200). Overrides --k if specified.')
    parser.add_argument('--stratified_subsampling', action='store_true', help='Whether to use stratified subsampling when limiting training samples')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # -- Parameters
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
    n_samples_list = args.n_samples
    
    if n_samples_list is not None:
        print(f"Using exact training sample sizes: {n_samples_list}")
    else:
        print(f"Using {int(k * 100)}% of samples for training with stratified random sampling.")
    
    stratified_subsampling = args.stratified_subsampling

    if stratified_subsampling:
        print("Using stratified subsampling based on class labels when limiting training samples.")
        output_base_dir = 'tabarena_results/stratified_subsampling'
    else:
        output_base_dir = 'tabarena_results/standard_subsampling'

    # Iterate over each n_samples value if multiple are provided
    n_samples_to_run = n_samples_list if n_samples_list is not None else [None]
    
    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        if dataset.qualities['NumberOfClasses'] != 2:
            print("Skipping regression or multi-class task.")
            continue

        if dataset.qualities['NumberOfFeatures'] > 500:
            print("Skipping dataset with more than 500 features.")
            continue

        if  dataset.qualities.get('MinorityClassSize') / ( dataset.qualities.get('MajorityClassSize') + dataset.qualities.get('MinorityClassSize')) < 0.1:
            print("Skipping highly imbalanced dataset.")
            continue

        #if dataset.name != "online_shoppers_intention":
        #    continue
        
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
            n_instances = dataset.qualities["NumberOfInstances"]
            if n_instances < 2_500:
                tabarena_repeats = 10
            elif n_instances > 250_000:
                tabarena_repeats = 1
            else:
                tabarena_repeats = 3

        print(f"TabArena Repeats: {tabarena_repeats} | Folds: {folds}")
        
        methods = ['tab_icl', 'tab_pfn']

        # Initialize results_dict with structure for all n_samples values
        results_dict = {
            'metadata': {
                'task_id': task.id,
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'k': k,
                'n_samples_list': n_samples_list,
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
        
        # Initialize nested structure for each n_samples value
        for n_samples_key in n_samples_to_run:
            key = f"n_samples_{n_samples_key}" if n_samples_key is not None else "k_fraction"
            results_dict[key] = {
                method: {'f1s': [[] for _ in range(tabarena_repeats)], 'roc_aucs': [[] for _ in range(tabarena_repeats)]} 
                for method in methods
            }

        # Create output directory
        output_dir = f'{output_base_dir}/{dataset.name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load the data for each split
        for repeat in range(tabarena_repeats):
            for fold in range(folds):
                seed = (fold + 1) * (repeat + 1)
                r = np.random.RandomState(seed)

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

                #Get binary encoded y_train
                y_train_binary = np.array([1 if label == pos_class else 0 for label in y_train])

                # Iterate over each n_samples value
                for current_n_samples in n_samples_to_run:
                    n_samples_key = f"n_samples_{current_n_samples}" if current_n_samples is not None else "k_fraction"
                    
                    if stratified_subsampling:
                        # Stratified random variants: sample positives and negatives separately to preserve balance
                        pos_indices = np.array([i for i, lbl in enumerate(y_train_binary) if lbl == 1])
                        neg_indices = np.array([i for i, lbl in enumerate(y_train_binary) if lbl == 0])

                        if current_n_samples is not None:
                            # Use exact number of samples, split proportionally between classes
                            total_samples = len(y_train_binary)
                            pos_ratio = len(pos_indices) / total_samples
                            pos_sample_size = min(len(pos_indices), max(1, int(current_n_samples * pos_ratio)))
                            neg_sample_size = min(len(neg_indices), max(1, current_n_samples - pos_sample_size))
                        else:
                            # Use fraction k
                            pos_sample_size = min(len(pos_indices), max(1, int(k * len(pos_indices))))
                            neg_sample_size = min(len(neg_indices), max(1, int(k * len(neg_indices))))

                        # Uniform stratified random
                        random_pos = r.choice(pos_indices, size=pos_sample_size, replace=False)
                        random_neg = r.choice(neg_indices, size=neg_sample_size, replace=False)
                        random_k_indices = np.concatenate([random_pos, random_neg])
                    else:
                        # Uniform random variant: sample randomly from the entire training set without stratification
                        if current_n_samples is not None:
                            random_k_indices = r.choice(len(y_train_binary), size=min(current_n_samples, len(y_train_binary)), replace=False)
                        else:
                            random_k_indices = r.choice(len(y_train_binary), size=int(k * len(y_train_binary)), replace=False)

                    # Evaluate both TabICL and TabPFN
                    for method in methods:
                        f1, roc_auc = train_and_score_model_on_subset(
                            X_train, y_train, X_test, y_test, 
                            subset_indices=random_k_indices, 
                            n_estimators=1, 
                            random_state=seed, 
                            pos_label=pos_class,
                            model_type=method
                        )
                        
                        #Store results
                        results_dict[n_samples_key][method]['f1s'][repeat].append(f1)
                        results_dict[n_samples_key][method]['roc_aucs'][repeat].append(roc_auc)

                        print(f"Repeat {repeat+1}/{tabarena_repeats}, Fold {fold+1}/{folds}, Method: {method}, n_samples: {current_n_samples} | ROC AUC: {roc_auc:.4f}")
        
        #save results_dict
        with open(f'{output_dir}/results.pkl', 'wb') as f:
            pickle.dump(results_dict, f)