"""
Simple script to run retrieval-based reduction test using TabArena datasets.
Combines the test logic from retrieval_based_reduction.py with TabArena data.
"""

import openml
from tabicl import TabICLClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


def run_retrieval_based_reduction_on_tabarena(lite=True, num_seeds=5):
    """
    Run retrieval-based reduction test on TabArena datasets.
    
    Parameters:
    -----------
    lite : bool
        If True, use TabArena-Lite (first fold, first repeat only)
    num_seeds : int
        Number of random seeds to run
    """
    
    # -- Get TabArena data
    benchmark_suite = openml.study.get_suite("tabarena-v0.1")
    task_ids = benchmark_suite.tasks
    
    print("Running retrieval-based reduction on TabArena tasks...")
    if lite:
        print("Using TabArena Lite (first fold, first repeat only)\n")
    

    # Process each task
    for task_id in task_ids[3:]:  # Process first 3 tasks as a simple example
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        
        # Skip non-binary classification tasks
        if dataset.qualities['NumberOfClasses'] != 2:
            print(f"Skipping task {task_id} - not binary classification")
            continue
        
        # Skip high-dimensional datasets
        if dataset.qualities['NumberOfFeatures'] > 500:
            print(f"Skipping task {task_id} - too many features")
            continue
        
        print(f"Task ID: {task.id}, Dataset: {dataset.name}")
        print("=" * 70)
        
        # Get the data
        #X, y, categorical_indicator, attribute_names = dataset.get_data(
        #    dataset_format='array',
        #    include_row_id=False,
        #    include_ignore_attributes=False
        #)

        X, y, categorical_indicator, attribute_names = dataset.get_data(
                    target=task.target_name, dataset_format="array"
        )
        
        # Run multiple seeds
        for seed in tqdm(range(num_seeds), desc=f"Processing {dataset.name}"):
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )
            
            max_train_samples = 10000000
            max_test_samples = 500

            # Subsample for faster testing if needed
            if X_train.shape[0] > max_train_samples:
                subsample_indices = np.random.choice(
                    X_train.shape[0], max_train_samples, replace=False
                )
                X_train = X_train[subsample_indices]
                y_train = y_train[subsample_indices]
            if X_test.shape[0] > max_test_samples:
                subsample_indices = np.random.choice(
                    X_test.shape[0], max_test_samples, replace=False
                )
                X_test = X_test[subsample_indices]
                y_test = y_test[subsample_indices]

            # Train TabICL classifier
            clf = TabICLClassifier(n_estimators=1, random_state=seed, k=None)
            clf.fit(X_train, y_train)
            
            print(X_train.shape, X_test.shape)
            # Predict on test set
            y_pred = clf.predict(X_test)
            
            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Print results
            print(f"  Seed {seed}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
        
        print()


if __name__ == "__main__":
    run_retrieval_based_reduction_on_tabarena(lite=True, num_seeds=5)
