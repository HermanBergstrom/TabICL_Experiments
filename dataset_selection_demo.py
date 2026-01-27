from tabicl import TabICLClassifier
from sklearn.datasets import load_diabetes, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def train_and_score_model(X_train, y_train, X_test, y_test, n_estimators=1, random_state=0):

    classifier = TabICLClassifier(n_estimators=n_estimators, random_state=random_state)
    classifier.fit(X_train, y_train)
        
    y_pred = classifier.predict(X_test)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return f1

def save_results(directory, full_set_f1s, removed_topk_f1s, removed_bottomk_f1s, ks):
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    #Save results in np files
    np.save(os.path.join(directory, 'full_set_f1s.npy'), np.array(full_set_f1s))
    
    #Create topk and botk directories
    topk_dir = os.path.join(directory, "topk")
    bottomk_dir = os.path.join(directory, "bottomk")
    if not os.path.exists(topk_dir):
        os.makedirs(topk_dir)
    if not os.path.exists(bottomk_dir):
        os.makedirs(bottomk_dir)

    for k in ks:
        np.save(os.path.join(topk_dir, f'removed_topk_f1s_k{k}.npy'), np.array(removed_topk_f1s[k]))
        np.save(os.path.join(bottomk_dir, f'removed_bottomk_f1s_k{k}.npy'), np.array(removed_bottomk_f1s[k]))

def get_ks_from_directory(directory):
    #derive ks from files in directory
    ks = []
    for filename in os.listdir(os.path.join(directory, "topk")):
        if filename.startswith('removed_topk_f1s_k') and filename.endswith('.npy'):
            k = int(filename[len('removed_topk_f1s_k'):-len('.npy')])
            ks.append(k)
    ks = sorted(ks)
    return ks

def create_plot(res_directory, output_directory):
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    ks = get_ks_from_directory(res_directory)

    # Load the stored results
    full_set_f1s = np.load(f'{res_directory}/full_set_f1s.npy')
    removed_topk_f1s = {k: np.load(f'{res_directory}/topk/removed_topk_f1s_k{k}.npy') for k in ks}
    removed_bottomk_f1s = {k: np.load(f'{res_directory}/bottomk/removed_bottomk_f1s_k{k}.npy') for k in ks}

    # Create plot with F1 on y-axis and k on x-axis. Include full set as horizontal line
    plt.figure(figsize=(10, 6))
    plt.plot(ks, [np.mean(removed_topk_f1s[k]) for k in ks], marker='o', label='Removed Top-k Attention Samples')
    plt.plot(ks, [np.mean(removed_bottomk_f1s[k]) for k in ks], marker='o', label='Removed Bottom-k Attention Samples')
    plt.axhline(y=np.mean(full_set_f1s), color='r', linestyle='--', label='Full Set F1 Score')
    plt.fill_between(ks, [np.mean(removed_topk_f1s[k]) - np.std(removed_topk_f1s[k]) for k in ks],
                    [np.mean(removed_topk_f1s[k]) + np.std(removed_topk_f1s[k]) for k in ks], alpha=0.2)
    plt.fill_between(ks, [np.mean(removed_bottomk_f1s[k]) - np.std(removed_bottomk_f1s[k]) for k in ks],
                    [np.mean(removed_bottomk_f1s[k]) + np.std(removed_bottomk_f1s[k]) for k in ks], alpha=0.2)
    plt.title('Impact of Removing Training Samples Based on Attention Weights (Total training samples: 455)')
    plt.xlabel('Number of Removed Samples (k)')
    plt.ylabel('F1 Score')
    # Rotate x ticks for readability
    plt.xticks(ks, rotation=45)
    #set y-axis limits to 0 and 1
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_directory}/attention_removal_f1_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {output_directory}/attention_removal_f1_comparison.png')

def plot_highlighted_embeddings(embeddings, labels, title, filename, top_k_indices, bottom_k_indices, k):
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
    plt.savefig(f'representation_visualizations/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

def create_representation_plots(sorted_indices, k, representations, y_train, save_directory, seed):

    top_k_indices = sorted_indices[-k:]
    bottom_k_indices = sorted_indices[:k]

    # Perform PCA
    pca = PCA(n_components=2)
    representations_pca = pca.fit_transform(representations)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    representations_tsne = tsne.fit_transform(representations)


    #Create directory if it does not exist
    if not os.path.exists('representation_visualizations'):
        os.makedirs('representation_visualizations')

    # Plot PCA
    plot_highlighted_embeddings(
        representations_pca, 
        y_train, 
        f'PCA of Representations (Highlighting Top/Bottom {k} Attention Samples)',
        f'representations_pca_k{k}_seed{seed}.png',
        top_k_indices,
        bottom_k_indices,
        k
    )

    # Plot t-SNE
    plot_highlighted_embeddings(
        representations_tsne, 
        y_train, 
        f't-SNE of Representations (Highlighting Top/Bottom {k} Attention Samples)',
        f'representations_tsne_k{k}_seed{seed}.png',
        top_k_indices,
        bottom_k_indices,
        k
    )

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Dataset Selection Demo using TabICL')
    parser.add_argument('--save_dir', type=str, default='subset_results', help='Directory to save results')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--create_representation_plots', action='store_true', help='Whether to create representation plots')
    return parser.parse_args()

#Main function
if __name__ == "__main__":

    args = parse_arguments()
    save_dir = args.save_dir
    plot_directory = args.plot_dir

    # Load the diabetes dataset
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target


    ks = [100, 200, 300, 325, 350, 370, 380, 390, 400, 410, 420, 430, 440]

    full_set_f1s = []
    removed_topk_f1s = {k: [] for k in ks}
    removed_bottomk_f1s = {k: [] for k in ks}

    seeds = 5

    for seed in tqdm(range(seeds)):

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        clf = TabICLClassifier(n_estimators=1, random_state=seed)
        clf.fit(X_train, y_train)  # this is cheap

        res, rollout = clf.predict_with_rollout(X_test, return_row_emb=args.create_representation_plots)

        full_set_f1 = f1_score(y_test, res, zero_division=0)
        full_set_f1s.append(full_set_f1)

        #We take the first element because n_estimators=1. If this is higher you get one rollout per estimator.
        #I realize that this might have an effect on the 'row_emb_rollout' since they permute the columns, but
        #it should not affect the 'icl' rollout.
        rollout_icl = rollout['icl_rollout'][0]

        #Average attention on training samples by test samples
        #The batch size is 1 in our case, so we take the first element
        average_attention = np.mean(rollout_icl[0, X_train.shape[0]:, :X_train.shape[0]], axis=0)
        sorted_indices = np.argsort(average_attention)

        if args.create_representation_plots:
            create_representation_plots(sorted_indices, 20, rollout['row_embeddings'][0][0][:X_train.shape[0]], y_train, plot_directory, seed)

        for k in ks:

            # Order training samples by average attention, compare removing top 100 sample vs bottom 100 samples
            top_k_indices = sorted_indices[-k:]
            bottom_k_indices = sorted_indices[:k]
            X_train_top_k_removed = np.delete(X_train, top_k_indices, axis=0)
            y_train_top_k_removed = np.delete(y_train, top_k_indices, axis=0)
            X_train_bottom_k_removed = np.delete(X_train, bottom_k_indices, axis=0)
            y_train_bottom_k_removed = np.delete(y_train, bottom_k_indices, axis=0)

            # Retrain TabICL without top-k 
            top_k_removed_f1 = train_and_score_model(X_train_top_k_removed, y_train_top_k_removed, X_test, y_test, n_estimators=1, random_state=seed)
            removed_topk_f1s[k].append(top_k_removed_f1)

            #... and without the bottom-k samples
            bottom_k_removed_f1 = train_and_score_model(X_train_bottom_k_removed, y_train_bottom_k_removed, X_test, y_test, n_estimators=1, random_state=seed)
            removed_bottomk_f1s[k].append(bottom_k_removed_f1)

    save_results(save_dir, full_set_f1s, removed_topk_f1s, removed_bottomk_f1s, ks)

    create_plot(save_dir, plot_directory)

    

    


    