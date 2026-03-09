from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:  # pragma: no cover - optional dependency
    umap = None


TOP_ATTENDED_FRACTION = 0.10


def _safe_label_for_filename(label):
    safe = str(label)
    for ch in ["/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"]:
        safe = safe.replace(ch, "_")
    return safe


def _build_subset_indices(train_labels, hit_counts, top_k_per_class, subset_size, random_state):
    n_samples = train_labels.shape[0]
    if subset_size is None or subset_size <= 0 or subset_size >= n_samples:
        return np.arange(n_samples)

    rng = np.random.RandomState(random_state)
    unique_labels = np.unique(train_labels)

    must_keep = []
    for label in unique_labels:
        cls_indices = np.where(train_labels == label)[0]
        if cls_indices.size == 0:
            continue
        cls_hits = hit_counts[cls_indices]
        top_count = min(max(1, top_k_per_class), cls_indices.size)
        top_local = np.argsort(cls_hits)[::-1][:top_count]
        must_keep.extend(cls_indices[top_local].tolist())

    must_keep = np.unique(np.asarray(must_keep, dtype=int))
    if must_keep.size >= subset_size:
        return np.sort(must_keep[:subset_size])

    remaining = np.setdiff1d(np.arange(n_samples), must_keep, assume_unique=False)
    n_to_add = subset_size - must_keep.size
    if n_to_add > 0 and remaining.size > 0:
        sampled = rng.choice(remaining, size=min(n_to_add, remaining.size), replace=False)
        return np.sort(np.concatenate([must_keep, sampled]))

    return np.sort(must_keep)


def _standardize(x):
    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    return (x - feature_mean) / feature_std


def _compute_2d_embeddings(x, random_state):
    x = _standardize(x)
    x_pca = PCA(n_components=2).fit_transform(x)

    if x.shape[0] < 2:
        return x_pca, x_pca, x_pca

    tsne_perplexity = min(30, max(5, x.shape[0] // 10), x.shape[0] - 1)
    x_tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=tsne_perplexity,
    ).fit_transform(x)

    if umap is None:
        #Throw exception
        raise ImportError("UMAP is not installed. Please install it to use UMAP embedding.")
    else:
        n_neighbors = min(15, max(2, x.shape[0] - 1))
        x_umap = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=0.1,
        ).fit_transform(x)

    return x_pca, x_tsne, x_umap


def _class_color_map(labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        return {
            unique_labels[0]: "tab:blue",
            unique_labels[1]: "tab:orange",
        }

    cmap = plt.get_cmap("tab10")
    return {label: cmap(idx % 10) for idx, label in enumerate(unique_labels)}


def _plot_two_panel_embeddings(
    x_pca,
    x_tsne,
    x_umap,
    base_labels,
    class_colors,
    title,
    output_path,
    highlight_mask=None,
    highlight_label="Highlighted",
    highlight_size=120,
    highlight_marker="*",
    highlight_use_class_colors=False,
    query_mask=None,
    query_label="Query sample",
    query_points=None,
    query_colors=None,
):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)
    views = [
        (axes[0], x_pca, "PCA Component 1", "PCA Component 2", "PCA"),
        (axes[1], x_tsne, "t-SNE Component 1", "t-SNE Component 2", "t-SNE"),
        (axes[2], x_umap, "UMAP Component 1", "UMAP Component 2", "UMAP"),
    ]

    unique_labels = np.unique(base_labels)

    for ax, x_view, x_label, y_label, panel_title in views:
        for label in unique_labels:
            cls_idx = np.where(base_labels == label)[0]
            ax.scatter(
                x_view[cls_idx, 0],
                x_view[cls_idx, 1],
                color=class_colors[label],
                s=22,
                alpha=0.35,
                label=f"Class {label}",
            )

        if highlight_mask is not None and np.any(highlight_mask):
            if highlight_use_class_colors:
                labeled_once = False
                for label in unique_labels:
                    cls_hi = (base_labels == label) & highlight_mask
                    if not np.any(cls_hi):
                        continue
                    ax.scatter(
                        x_view[cls_hi, 0],
                        x_view[cls_hi, 1],
                        color=class_colors[label],
                        s=highlight_size,
                        marker=highlight_marker,
                        edgecolors="black",
                        linewidths=1.0,
                        label=highlight_label if not labeled_once else None,
                    )
                    labeled_once = True
            else:
                ax.scatter(
                    x_view[highlight_mask, 0],
                    x_view[highlight_mask, 1],
                    color="gold",
                    s=highlight_size,
                    marker=highlight_marker,
                    edgecolors="black",
                    linewidths=0.9,
                    label=highlight_label,
                )

        if query_mask is not None and np.any(query_mask):
            ax.scatter(
                x_view[query_mask, 0],
                x_view[query_mask, 1],
                color="crimson",
                s=220,
                marker="X",
                edgecolors="black",
                linewidths=1.1,
                label=query_label,
            )

        if query_points is not None:
            if isinstance(query_points, dict):
                qx = query_points.get(panel_title)
            else:
                if len(query_points) == 3:
                    qx = (
                        query_points[0]
                        if panel_title == "PCA"
                        else query_points[1]
                        if panel_title == "t-SNE"
                        else query_points[2]
                    )
                else:
                    qx = query_points[0] if panel_title == "PCA" else query_points[1]

            if qx is None:
                continue
            if query_colors is None:
                point_colors = "crimson"
            else:
                point_colors = query_colors
            ax.scatter(
                qx[:, 0],
                qx[:, 1],
                color=point_colors,
                s=220,
                marker="X",
                edgecolors="black",
                linewidths=1.1,
                label=query_label,
            )

        ax.set_title(panel_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="best", fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_comparison_with_per_class_highlights(
    train_representations,
    train_labels,
    hit_counts,
    top_k_per_class,
    output_path,
    title,
    subset_size=None,
    random_state=0,
):
    if hasattr(train_representations, "toarray"):
        x_dense = train_representations.toarray()
    else:
        x_dense = np.asarray(train_representations)

    if x_dense.shape[0] == 0:
        return

    train_labels = np.asarray(train_labels)
    hit_counts = np.asarray(hit_counts)
    subset_indices = _build_subset_indices(
        train_labels=train_labels,
        hit_counts=hit_counts,
        top_k_per_class=top_k_per_class,
        subset_size=subset_size,
        random_state=random_state,
    )

    x_dense = x_dense[subset_indices]
    train_labels = train_labels[subset_indices]
    hit_counts = hit_counts[subset_indices]

    if x_dense.shape[0] < 2:
        return

    x_pca, x_tsne, x_umap = _compute_2d_embeddings(x_dense, random_state=random_state)

    unique_labels = np.unique(train_labels)
    class_colors = _class_color_map(train_labels)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)
    embedding_views = [
        (axes[0], x_pca, "PCA Component 1", "PCA Component 2", "PCA"),
        (axes[1], x_tsne, "t-SNE Component 1", "t-SNE Component 2", "t-SNE"),
        (axes[2], x_umap, "UMAP Component 1", "UMAP Component 2", "UMAP"),
    ]

    for ax, x_view, x_label, y_label, panel_title in embedding_views:
        for label in unique_labels:
            cls_indices = np.where(train_labels == label)[0]
            ax.scatter(
                x_view[cls_indices, 0],
                x_view[cls_indices, 1],
                color=class_colors[label],
                s=26,
                alpha=0.7,
                label=f"Class {label}",
            )

        for label in unique_labels:
            cls_indices = np.where(train_labels == label)[0]
            if cls_indices.size == 0:
                continue
            cls_hits = hit_counts[cls_indices]
            top_count = min(max(1, top_k_per_class), cls_indices.size)
            top_local = np.argsort(cls_hits)[::-1][:top_count]
            top_global = cls_indices[top_local]

            ax.scatter(
                x_view[top_global, 0],
                x_view[top_global, 1],
                color=class_colors[label],
                s=180,
                marker="*",
                edgecolors="black",
                linewidths=1.1,
                label=f"Class {label} top-{top_count}",
            )

        ax.set_title(panel_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="best", fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_overall_attended_samples(
    train_representations,
    train_labels,
    attn_maps,
    output_path,
    title,
    top_fraction=TOP_ATTENDED_FRACTION,
    random_state=0,
):
    x_dense = np.asarray(train_representations)
    train_labels = np.asarray(train_labels)
    attn_maps = np.asarray(attn_maps)

    if x_dense.shape[0] == 0:
        return

    attn_received = attn_maps.sum(axis=0)
    n_top = max(1, int(np.ceil(top_fraction * x_dense.shape[0])))
    top_idx = np.argsort(attn_received)[::-1][:n_top]
    highlight_mask = np.zeros(x_dense.shape[0], dtype=bool)
    highlight_mask[top_idx] = True

    x_pca, x_tsne, x_umap = _compute_2d_embeddings(x_dense, random_state=random_state)
    class_colors = _class_color_map(train_labels)
    _plot_two_panel_embeddings(
        x_pca=x_pca,
        x_tsne=x_tsne,
        x_umap=x_umap,
        base_labels=train_labels,
        class_colors=class_colors,
        title=title,
        output_path=output_path,
        highlight_mask=highlight_mask,
        highlight_label=f"Top {int(top_fraction * 100)}% overall attention",
    )


def plot_top_attended_per_class_samples(
    train_representations,
    train_labels,
    attn_maps,
    output_path,
    title,
    top_fraction=TOP_ATTENDED_FRACTION,
    random_state=0,
):
    x_dense = np.asarray(train_representations)
    train_labels = np.asarray(train_labels)
    attn_maps = np.asarray(attn_maps)

    if x_dense.shape[0] == 0:
        return

    attn_received = attn_maps.sum(axis=0)
    highlight_mask = np.zeros(x_dense.shape[0], dtype=bool)

    for label in np.unique(train_labels):
        cls_idx = np.where(train_labels == label)[0]
        if cls_idx.size == 0:
            continue
        n_top = max(1, int(np.ceil(top_fraction * cls_idx.size)))
        cls_scores = attn_received[cls_idx]
        cls_top_local = np.argsort(cls_scores)[::-1][:n_top]
        highlight_mask[cls_idx[cls_top_local]] = True

    x_pca, x_tsne, x_umap = _compute_2d_embeddings(x_dense, random_state=random_state)
    class_colors = _class_color_map(train_labels)
    _plot_two_panel_embeddings(
        x_pca=x_pca,
        x_tsne=x_tsne,
        x_umap=x_umap,
        base_labels=train_labels,
        class_colors=class_colors,
        title=title,
        output_path=output_path,
        highlight_mask=highlight_mask,
        highlight_label=f"Top {int(top_fraction * 100)}% attention within each class",
    )


def plot_test_sample_attention_neighborhoods(
    train_representations,
    train_labels,
    test_representations,
    attn_maps,
    output_dir,
    title_prefix,
    n_test_samples=8,
    top_fraction=TOP_ATTENDED_FRACTION,
    random_state=0,
):
    x_train = np.asarray(train_representations)
    y_train = np.asarray(train_labels)
    x_test = np.asarray(test_representations)
    attn_maps = np.asarray(attn_maps)

    if x_train.shape[0] == 0 or x_test.shape[0] == 0 or attn_maps.shape[0] == 0:
        return

    rng = np.random.RandomState(random_state)
    n_pick = min(max(1, n_test_samples), x_test.shape[0])
    sampled_test_idx = rng.choice(np.arange(x_test.shape[0]), size=n_pick, replace=False)

    x_all = np.concatenate([x_train, x_test], axis=0)
    x_pca_all, x_tsne_all, x_umap_all = _compute_2d_embeddings(x_all, random_state=random_state)
    x_pca_train, x_pca_test = x_pca_all[: x_train.shape[0]], x_pca_all[x_train.shape[0] :]
    x_tsne_train, x_tsne_test = x_tsne_all[: x_train.shape[0]], x_tsne_all[x_train.shape[0] :]
    x_umap_train, x_umap_test = x_umap_all[: x_train.shape[0]], x_umap_all[x_train.shape[0] :]

    class_colors = _class_color_map(y_train)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for test_idx in sampled_test_idx:
        train_scores = attn_maps[test_idx]
        n_top = max(1, int(np.ceil(top_fraction * x_train.shape[0])))
        top_train_idx = np.argsort(train_scores)[::-1][:n_top]

        highlight_mask = np.zeros(x_train.shape[0], dtype=bool)
        highlight_mask[top_train_idx] = True

        _plot_two_panel_embeddings(
            x_pca=x_pca_train,
            x_tsne=x_tsne_train,
            x_umap=x_umap_train,
            base_labels=y_train,
            class_colors=class_colors,
            title=f"{title_prefix} | test sample {int(test_idx)}",
            output_path=output_dir / f"test_sample_{int(test_idx)}.png",
            highlight_mask=highlight_mask,
            highlight_label=f"Top {int(top_fraction * 100)}% attended train for this test sample",
            query_label="Test sample",
            query_points=(x_pca_test[[test_idx]], x_tsne_test[[test_idx]]),
        )


def plot_embedding_attention_suite(
    train_representations,
    train_labels,
    hit_counts,
    attn_maps,
    test_representations,
    test_labels,
    top_k_per_class,
    top_fraction,
    subset_size,
    n_test_samples,
    random_state,
    comparison_output_path,
    top_overall_output_path,
    top_per_class_output_dir,
    test_neighborhood_output_dir,
    comparison_title,
    top_overall_title,
    top_per_class_title,
    test_title_prefix,
):
    if hasattr(train_representations, "toarray"):
        x_train = train_representations.toarray()
    else:
        x_train = np.asarray(train_representations)

    if hasattr(test_representations, "toarray"):
        x_test = test_representations.toarray()
    else:
        x_test = np.asarray(test_representations)

    y_train = np.asarray(train_labels)
    y_test = np.asarray(test_labels)
    hit_counts = np.asarray(hit_counts)
    attn_maps = np.asarray(attn_maps)

    if x_train.shape[0] == 0:
        return

    subset_indices = _build_subset_indices(
        train_labels=y_train,
        hit_counts=hit_counts,
        top_k_per_class=top_k_per_class,
        subset_size=subset_size,
        random_state=random_state,
    )
    if subset_indices.size == 0:
        return

    x_train_sub = x_train[subset_indices]
    y_train_sub = y_train[subset_indices]
    hit_counts_sub = hit_counts[subset_indices]
    attn_maps_sub = attn_maps[:, subset_indices] if attn_maps.ndim == 2 else attn_maps

    available_test = min(attn_maps_sub.shape[0], x_test.shape[0], y_test.shape[0]) if x_test.ndim == 2 else min(attn_maps_sub.shape[0], y_test.shape[0])
    if available_test > 0 and n_test_samples > 0:
        rng = np.random.RandomState(random_state)
        n_pick = min(n_test_samples, available_test)
        sampled_test_indices = rng.choice(np.arange(available_test), size=n_pick, replace=False)
        x_test_sub = x_test[sampled_test_indices]
        y_test_sub = y_test[sampled_test_indices]
        x_joint = np.concatenate([x_train_sub, x_test_sub], axis=0)
        x_pca_joint, x_tsne_joint, x_umap_joint = _compute_2d_embeddings(x_joint, random_state=random_state)
        x_pca_train = x_pca_joint[: x_train_sub.shape[0]]
        x_tsne_train = x_tsne_joint[: x_train_sub.shape[0]]
        x_umap_train = x_umap_joint[: x_train_sub.shape[0]]
        x_pca_test = x_pca_joint[x_train_sub.shape[0] :]
        x_tsne_test = x_tsne_joint[x_train_sub.shape[0] :]
        x_umap_test = x_umap_joint[x_train_sub.shape[0] :]
    else:
        sampled_test_indices = np.array([], dtype=int)
        y_test_sub = np.array([])
        x_pca_train, x_tsne_train, x_umap_train = _compute_2d_embeddings(x_train_sub, random_state=random_state)
        x_pca_test = np.empty((0, 2), dtype=float)
        x_tsne_test = np.empty((0, 2), dtype=float)
        x_umap_test = np.empty((0, 2), dtype=float)

    class_colors = _class_color_map(y_train_sub)

    # Plot 1: comparison with top-k-per-class by hit count.
    comparison_highlight_mask = np.zeros(y_train_sub.shape[0], dtype=bool)
    for label in np.unique(y_train_sub):
        cls_idx = np.where(y_train_sub == label)[0]
        if cls_idx.size == 0:
            continue
        cls_hits = hit_counts_sub[cls_idx]
        top_count = min(max(1, top_k_per_class), cls_idx.size)
        cls_top_local = np.argsort(cls_hits)[::-1][:top_count]
        comparison_highlight_mask[cls_idx[cls_top_local]] = True

    _plot_two_panel_embeddings(
        x_pca=x_pca_train,
        x_tsne=x_tsne_train,
        x_umap=x_umap_train,
        base_labels=y_train_sub,
        class_colors=class_colors,
        title=comparison_title,
        output_path=comparison_output_path,
        highlight_mask=comparison_highlight_mask,
        highlight_label="Top hit-count per class",
        highlight_use_class_colors=True,
    )

    # Plot 2: top attended overall by summed attention (within subset).
    attn_received_sub = attn_maps_sub.sum(axis=0)
    n_top_overall = max(1, int(np.ceil(top_fraction * x_train_sub.shape[0])))
    top_overall_idx = np.argsort(attn_received_sub)[::-1][:n_top_overall]
    top_overall_mask = np.zeros(x_train_sub.shape[0], dtype=bool)
    top_overall_mask[top_overall_idx] = True

    _plot_two_panel_embeddings(
        x_pca=x_pca_train,
        x_tsne=x_tsne_train,
        x_umap=x_umap_train,
        base_labels=y_train_sub,
        class_colors=class_colors,
        title=top_overall_title,
        output_path=top_overall_output_path,
        highlight_mask=top_overall_mask,
        highlight_label=f"Top {int(top_fraction * 100)}% overall attention",
        highlight_use_class_colors=True,
    )

    # Plot 3: one plot per class for top attended by summed attention (within subset).
    top_per_class_output_dir = Path(top_per_class_output_dir)
    top_per_class_output_dir.mkdir(parents=True, exist_ok=True)
    for label in np.unique(y_train_sub):
        cls_idx = np.where(y_train_sub == label)[0]
        if cls_idx.size == 0:
            continue
        n_top_cls = max(1, int(np.ceil(top_fraction * cls_idx.size)))
        cls_scores = attn_received_sub[cls_idx]
        cls_top_local = np.argsort(cls_scores)[::-1][:n_top_cls]
        top_per_class_mask = np.zeros(x_train_sub.shape[0], dtype=bool)
        top_per_class_mask[cls_idx[cls_top_local]] = True

        _plot_two_panel_embeddings(
            x_pca=x_pca_train,
            x_tsne=x_tsne_train,
            x_umap=x_umap_train,
            base_labels=y_train_sub,
            class_colors=class_colors,
            title=f"{top_per_class_title} | class {label}",
            output_path=top_per_class_output_dir / f"class_{_safe_label_for_filename(label)}.png",
            highlight_mask=top_per_class_mask,
            highlight_label=f"Top {int(top_fraction * 100)}% attention in class {label}",
            highlight_use_class_colors=True,
        )

    # Plot 4: per-test sampled neighborhoods using same subset + same 2D projection.
    if sampled_test_indices.size == 0:
        return

    test_neighborhood_output_dir = Path(test_neighborhood_output_dir)
    test_neighborhood_output_dir.mkdir(parents=True, exist_ok=True)

    n_top_neighbors = max(1, int(np.ceil(top_fraction * x_train_sub.shape[0])))
    for local_test_pos, global_test_idx in enumerate(sampled_test_indices):
        test_scores = attn_maps_sub[global_test_idx]
        top_neighbor_idx = np.argsort(test_scores)[::-1][:n_top_neighbors]
        neighbor_mask = np.zeros(x_train_sub.shape[0], dtype=bool)
        neighbor_mask[top_neighbor_idx] = True
        top1_neighbor_idx = int(np.argmax(test_scores))
        top1_neighbor_mask = np.zeros(x_train_sub.shape[0], dtype=bool)
        top1_neighbor_mask[top1_neighbor_idx] = True

        test_label = y_test_sub[local_test_pos]
        test_color = class_colors.get(test_label, "crimson")
        test_sample_dir = (
            test_neighborhood_output_dir
            / f"test_sample_{int(global_test_idx)}_class_{_safe_label_for_filename(test_label)}"
        )
        test_sample_dir.mkdir(parents=True, exist_ok=True)

        _plot_two_panel_embeddings(
            x_pca=x_pca_train,
            x_tsne=x_tsne_train,
            x_umap=x_umap_train,
            base_labels=y_train_sub,
            class_colors=class_colors,
            title=f"{test_title_prefix} | test sample {int(global_test_idx)} | class {test_label}",
            output_path=test_sample_dir / "overall_top_attended.png",
            highlight_mask=neighbor_mask,
            highlight_label=f"Top {int(top_fraction * 100)}% attended train for this test sample",
            highlight_use_class_colors=True,
            query_label="Test sample",
            query_points=(
                x_pca_test[[local_test_pos]],
                x_tsne_test[[local_test_pos]],
                x_umap_test[[local_test_pos]],
            ),
            query_colors=[test_color],
        )

        _plot_two_panel_embeddings(
            x_pca=x_pca_train,
            x_tsne=x_tsne_train,
            x_umap=x_umap_train,
            base_labels=y_train_sub,
            class_colors=class_colors,
            title=(
                f"{test_title_prefix} | test sample {int(global_test_idx)} | class {test_label} | "
                "top 1 attended train sample"
            ),
            output_path=test_sample_dir / "top1_attended.png",
            highlight_mask=top1_neighbor_mask,
            highlight_label="Top 1 attended train sample",
            highlight_use_class_colors=True,
            query_label="Test sample",
            query_points=(
                x_pca_test[[local_test_pos]],
                x_tsne_test[[local_test_pos]],
                x_umap_test[[local_test_pos]],
            ),
            query_colors=[test_color],
        )

        # Additional per-test plots: top attended samples within each class.
        for cls_label in np.unique(y_train_sub):
            cls_idx = np.where(y_train_sub == cls_label)[0]
            if cls_idx.size == 0:
                continue

            n_top_cls_neighbors = max(1, int(np.ceil(top_fraction * cls_idx.size)))
            cls_scores = test_scores[cls_idx]
            cls_top_local = np.argsort(cls_scores)[::-1][:n_top_cls_neighbors]
            cls_top_idx = cls_idx[cls_top_local]

            cls_neighbor_mask = np.zeros(x_train_sub.shape[0], dtype=bool)
            cls_neighbor_mask[cls_top_idx] = True

            _plot_two_panel_embeddings(
                x_pca=x_pca_train,
                x_tsne=x_tsne_train,
                x_umap=x_umap_train,
                base_labels=y_train_sub,
                class_colors=class_colors,
                title=(
                    f"{test_title_prefix} | test sample {int(global_test_idx)} | class {test_label} | "
                    f"top {int(top_fraction * 100)}% attended in train class {cls_label}"
                ),
                output_path=(
                    test_sample_dir
                    / f"focus_train_class_{_safe_label_for_filename(cls_label)}.png"
                ),
                highlight_mask=cls_neighbor_mask,
                highlight_label=(
                    f"Top {int(top_fraction * 100)}% attended train in class {cls_label}"
                ),
                highlight_use_class_colors=True,
                query_label="Test sample",
                query_points=(
                    x_pca_test[[local_test_pos]],
                    x_tsne_test[[local_test_pos]],
                    x_umap_test[[local_test_pos]],
                ),
                query_colors=[test_color],
            )


def plot_hit_count_distribution(hit_counts, output_path, title):
    hit_counts = np.asarray(hit_counts)
    if hit_counts.size == 0:
        return

    values, counts = np.unique(hit_counts, return_counts=True)
    proportions = counts / counts.sum()

    plt.figure(figsize=(8, 5))
    plt.bar(values, proportions, color="tab:blue", alpha=0.85)
    plt.xlabel("Hit count")
    plt.ylabel("Proportion of training samples")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
