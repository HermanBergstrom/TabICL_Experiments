#Open /home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/tabarena_results/APSFailure.pkl and analyze the results
import argparse
import pickle
import numpy as np
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import textwrap


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ROC AUC histograms with error bars")
    parser.add_argument(
        "--use-std-error",
        action="store_true",
        help="Use standard error instead of standard deviation for error bars",
    )
    return parser.parse_args()

from typing import List, Tuple
import numpy as np

def aggregate_auroc(
    aurocs: List[List[float]], ddof: int = 1
) -> Tuple[float, float]:
    """
    Aggregate AUROC scores across folds and repeats.
    """

    scores = np.asarray(aurocs)

    if scores.ndim != 2:
        raise ValueError("Input must be a list of lists: [repeat][fold]")

    # Step 1: average over folds (per repeat)
    repeat_means = scores.mean(axis=1)

    # Step 2: average over repeats
    mean_auroc = repeat_means.mean()

    # Uncertainty: variability across repeats
    std_auroc = repeat_means.std(ddof=ddof)

    return mean_auroc, std_auroc

args = parse_args()
use_std_error = args.use_std_error

# Collect all results
all_results = {}
k_value = None

for results_dir in os.listdir('/home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/tabarena_results/'):
    dataset_name = results_dir
    #Check if results file exists
    result_path = os.path.join('/home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/tabarena_results/', results_dir, 'results.pkl')
    if os.path.exists(result_path):
        with open(result_path, 'rb') as f:
            results_dict = pickle.load(f)
            
            all_results[dataset_name] = results_dict
            if 'metadata' in results_dict and isinstance(results_dict['metadata'], dict):
                if 'k' in results_dict['metadata']:
                    current_k = results_dict['metadata']['k']
                    if k_value is None:
                        k_value = current_k
                    elif k_value != current_k:
                        print(f"Warning: K mismatch for {dataset_name}: {current_k} (existing {k_value})")

# Create a large plot with histograms
num_datasets = len(all_results)
num_cols = 3
num_rows = (num_datasets + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6 * num_rows))
axes = axes.flatten() if num_datasets > 1 else [axes]

methods = ['full_set', 'top_k', 'bot_k', 
           'random_k', 'weighted_random_k', 'inversly_weighted_random_k']
method_labels = ['Full Set', 'Top K', 'Bot K', 
                 'Random K', 'Weighted Random K', 'Inversly Weighted Random K']
wrapped_method_labels = [textwrap.fill(label, width=12, break_long_words=False) for label in method_labels]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8B008B']

for idx, (dataset_name, results_dict) in enumerate(sorted(all_results.items())):
    ax = axes[idx]
    
    # Calculate means and errors (std dev or std error)
    means = []
    stds = []
    counts = []
    folds = []

    for method in methods:
        if method in results_dict and len(results_dict[method]['roc_aucs']) > 0:
            roc_values = results_dict[method]['roc_aucs']
            mean_val, std_val = aggregate_auroc(roc_values)
            means.append(mean_val)
            stds.append(std_val)
            counts.append(len(roc_values))
            folds.append(len(roc_values[0]))
        else:
            means.append(0)
            stds.append(0)
            counts.append(0)
            folds.append(0)

    errors = []
    for std_val, count_val in zip(stds, counts):
        if use_std_error and count_val > 0:
            errors.append(std_val / np.sqrt(count_val))
        else:
            errors.append(std_val)
    
    # Create bar plot with error bars
    x_pos = np.arange(len(method_labels))
    bars = ax.bar(
        x_pos,
        means,
        yerr=errors,
        capsize=5,
        alpha=0.8,
        color=colors,
        edgecolor='black',
        linewidth=1.2,
    )
    
    ax.set_xlabel('Method', fontsize=10, fontweight='bold')
    ax.set_ylabel('ROC AUC', fontsize=10, fontweight='bold')
    # Add seed count to title (same for all methods)
    ax.set_title(f"{dataset_name} (repeats = {counts[0] if counts else 0}, folds = {folds[0] if folds else 0})", fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(wrapped_method_labels, rotation=0, ha='center', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(means) * 1.2 + max(errors) if max(means) > 0 else 1])
    
    # Add value labels on bars
    for bar, mean, err in zip(bars, means, errors):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + err,
            f'{mean:.3f}',
            ha='center',
            va='bottom',
            fontsize=7,
        )

# Remove empty subplots
for idx in range(num_datasets, len(axes)):
    fig.delaxes(axes[idx])

title_text = f"ROC AUC by Method (K={int(k_value * 100)}%)" if k_value is not None else "ROC AUC by Method"
fig.suptitle(title_text, fontsize=16, fontweight='bold')

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig('results_histograms.png', dpi=300, bbox_inches='tight')
print("Plot saved as results_histograms.png")
plt.show()