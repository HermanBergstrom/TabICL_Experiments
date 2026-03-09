import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = weights.sum()
    if total <= 0:
        return np.full(weights.shape[0], 1.0 / max(1, weights.shape[0]))
    return weights / total


def select_subset_indices(
    strategy,
    y_train_binary,
    rng,
    k,
    current_n_samples,
    attn_sums_per_class=None,
    train_labels=None,
):
    def _resolve_scores(indices, class_label=None):
        if attn_sums_per_class is None:
            return None
        if class_label is not None:
            if class_label in attn_sums_per_class:
                return attn_sums_per_class[class_label][indices]
            # If no matching-label test samples exist, avoid leaking other-class attention.
            return np.zeros(len(indices), dtype=float)
        combined = sum(attn_sums_per_class.values())
        return combined[indices]

    def _pick(indices, n_pick, class_label=None):
        n_pick = min(len(indices), max(1, n_pick))
        scores = _resolve_scores(indices, class_label)
        if strategy == "topk_attn_sum":
            order = np.argsort(scores)[::-1]
            return indices[order[:n_pick]]
        if strategy == "weighted_attn_sum":
            probs = normalize_weights(scores)
            return rng.choice(indices, size=n_pick, replace=False, p=probs)
        return rng.choice(indices, size=n_pick, replace=False)

    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    if strategy in {"topk_attn_sum", "weighted_attn_sum"}:
        if attn_sums_per_class is None:
            raise ValueError(
                f"{strategy} requires classwise attention sums (attn_sums_per_class), got None"
            )
        if train_labels is None:
            raise ValueError(
                f"{strategy} requires train_labels for label-matched classwise sampling"
            )

    requested_n_samples = None
    if current_n_samples is not None:
        requested_n_samples = int(current_n_samples)
    elif k > 1:
        requested_n_samples = int(k)

    # Always stratify sampling by train class distribution.
    if train_labels is None:
        labels_for_strat = np.asarray(y_train_binary)
    else:
        labels_for_strat = np.asarray(train_labels)

    total_len = len(labels_for_strat)
    unique_labels, counts = np.unique(labels_for_strat, return_counts=True)
    class_indices = {label: np.where(labels_for_strat == label)[0] for label in unique_labels}

    if requested_n_samples is not None:
        requested_n_samples = min(requested_n_samples, total_len)
        class_raw = requested_n_samples * (counts / counts.sum())
        class_base = np.floor(class_raw).astype(int)
        class_fracs = class_raw - class_base

        remainder = int(requested_n_samples - class_base.sum())
        if remainder > 0:
            order = np.argsort(class_fracs)[::-1]
            for i in range(remainder):
                class_base[order[i % len(order)]] += 1

        class_sample_sizes = {
            label: min(len(class_indices[label]), int(class_base[i]))
            for i, label in enumerate(unique_labels)
        }
    else:
        class_sample_sizes = {
            label: min(len(class_indices[label]), max(1, int(k * len(class_indices[label]))))
            for label in unique_labels
        }

    chosen_by_class = []
    for label in unique_labels:
        idx = class_indices[label]
        n_pick = class_sample_sizes[label]
        if len(idx) == 0 or n_pick <= 0:
            continue
        chosen_by_class.append(_pick(idx, n_pick, class_label=label))

    if len(chosen_by_class) == 0:
        return np.array([], dtype=int)
    return np.concatenate(chosen_by_class)


def compute_cluster_attention_summary(
    train_row_embeddings,
    attention_weights,
    n_clusters=200,
    random_state=0,
):
    """Cluster train embeddings and aggregate attention mass per cluster.

    Parameters
    ----------
    train_row_embeddings : array-like, shape (n_train, d)
        Row embeddings for train samples.
    attention_weights : array-like, shape (n_train,)
        Attention mass received per train sample.
    n_clusters : int
        Number of KMeans clusters to form (capped at n_train).

    Returns
    -------
    dict with:
        - cluster_labels: ndarray (n_train,)
        - cluster_to_indices: dict[int, ndarray]
        - cluster_attention_sum: dict[int, float]
        - cluster_attention_ratio: dict[int, float]
        - cluster_sizes: dict[int, int]
        - clusters_sorted_by_attention: list[int]
    """
    x = np.asarray(train_row_embeddings)
    attn = np.asarray(attention_weights, dtype=float)

    if x.ndim != 2:
        raise ValueError(f"train_row_embeddings must be 2-D, got shape={x.shape}")
    if attn.ndim != 1:
        raise ValueError(f"attention_weights must be 1-D, got shape={attn.shape}")
    if x.shape[0] != attn.shape[0]:
        raise ValueError(
            f"Row count mismatch: embeddings={x.shape[0]} vs attention_weights={attn.shape[0]}"
        )
    if x.shape[0] == 0:
        raise ValueError("No training rows were provided.")

    n_train = x.shape[0]
    n_clusters_eff = int(min(max(1, n_clusters), n_train))

    kmeans = KMeans(n_clusters=n_clusters_eff, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(x)

    cluster_to_indices = {}
    cluster_attention_sum = {}
    cluster_sizes = {}

    for cluster_id in range(n_clusters_eff):
        idx = np.where(cluster_labels == cluster_id)[0]
        cluster_to_indices[cluster_id] = idx
        cluster_sizes[cluster_id] = int(idx.size)
        cluster_attention_sum[cluster_id] = float(np.sum(attn[idx])) if idx.size > 0 else 0.0

    total_attn = float(np.sum(attn))
    if total_attn > 0:
        cluster_attention_ratio = {
            cid: float(cluster_attention_sum[cid] / total_attn) for cid in cluster_attention_sum
        }
    else:
        uniform = 1.0 / max(1, n_clusters_eff)
        cluster_attention_ratio = {cid: uniform for cid in cluster_attention_sum}

    clusters_sorted_by_attention = sorted(
        cluster_attention_sum.keys(),
        key=lambda cid: cluster_attention_sum[cid],
        reverse=True,
    )

    return {
        "cluster_labels": cluster_labels,
        "cluster_to_indices": cluster_to_indices,
        "cluster_attention_sum": cluster_attention_sum,
        "cluster_attention_ratio": cluster_attention_ratio,
        "cluster_sizes": cluster_sizes,
        "clusters_sorted_by_attention": clusters_sorted_by_attention,
        "kmeans_inertia": float(kmeans.inertia_),
        "n_clusters": int(n_clusters_eff),
    }


def sample_rows_from_cluster(cluster_to_indices, cluster_id, n_samples, rng):
    """Sample row indices from a specific cluster (without replacement)."""
    if cluster_id not in cluster_to_indices:
        raise KeyError(f"Unknown cluster_id={cluster_id}")

    indices = np.asarray(cluster_to_indices[cluster_id])
    if indices.size == 0:
        return np.array([], dtype=int)

    n_pick = min(int(n_samples), indices.size)
    if n_pick <= 0:
        return np.array([], dtype=int)
    return rng.choice(indices, size=n_pick, replace=False)


def sample_rows_from_top_attention_clusters(
    cluster_summary,
    total_samples,
    rng,
):
    """Sample rows across clusters with probability proportional to cluster attention mass."""
    cluster_to_indices = cluster_summary["cluster_to_indices"]
    cluster_ids = list(cluster_to_indices.keys())
    if not cluster_ids:
        return np.array([], dtype=int)

    cluster_probs = np.array(
        [cluster_summary["cluster_attention_ratio"][cid] for cid in cluster_ids],
        dtype=float,
    )
    prob_sum = cluster_probs.sum()
    if prob_sum <= 0:
        cluster_probs = np.full(len(cluster_ids), 1.0 / len(cluster_ids))
    else:
        cluster_probs = cluster_probs / prob_sum

    remaining = int(max(0, total_samples))
    chosen = []
    available_per_cluster = {cid: set(np.asarray(cluster_to_indices[cid]).tolist()) for cid in cluster_ids}

    while remaining > 0:
        non_empty = [cid for cid in cluster_ids if len(available_per_cluster[cid]) > 0]
        if not non_empty:
            break

        non_empty_probs = np.array([cluster_probs[cluster_ids.index(cid)] for cid in non_empty], dtype=float)
        non_empty_probs = non_empty_probs / non_empty_probs.sum()
        cid = rng.choice(non_empty, p=non_empty_probs)

        candidate_list = list(available_per_cluster[cid])
        picked = rng.choice(candidate_list)
        available_per_cluster[cid].remove(int(picked))
        chosen.append(int(picked))
        remaining -= 1

    return np.asarray(chosen, dtype=int)


def build_classwise_clusters(
    train_row_embeddings,
    train_labels,
    cluster_fraction=0.1,
    random_state=0,
):
    """Cluster train rows separately per class label.

    The number of clusters in each class is proportional to that class size.
    """
    x = np.asarray(train_row_embeddings)
    labels = np.asarray(train_labels)
    if x.ndim != 2:
        raise ValueError(f"train_row_embeddings must be 2-D, got shape={x.shape}")
    if labels.ndim != 1:
        raise ValueError(f"train_labels must be 1-D, got shape={labels.shape}")
    if x.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Row count mismatch: embeddings={x.shape[0]} vs labels={labels.shape[0]}"
        )
    if not (0 < cluster_fraction <= 1):
        raise ValueError(f"cluster_fraction must be in (0, 1], got {cluster_fraction}")

    n_train = x.shape[0]
    cluster_ids = np.full(n_train, -1, dtype=int)
    cluster_to_indices = {}
    cluster_meta = {}
    next_cluster_id = 0

    for label in np.unique(labels):
        class_idx = np.where(labels == label)[0]
        n_class = class_idx.size
        if n_class == 0:
            continue

        n_clusters = min(n_class, max(1, int(round(cluster_fraction * n_class))))
        if n_clusters == 1:
            local_assign = np.zeros(n_class, dtype=int)
        else:
            km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            local_assign = km.fit_predict(x[class_idx])

        label_key = label.item() if hasattr(label, "item") else label
        for local_cid in range(n_clusters):
            members = class_idx[local_assign == local_cid]
            global_cid = next_cluster_id
            next_cluster_id += 1
            cluster_to_indices[global_cid] = members
            cluster_ids[members] = global_cid
            cluster_meta[global_cid] = {
                "label": label_key,
                "n_members": int(members.size),
                "local_cluster_id": int(local_cid),
            }

    return {
        "cluster_ids": cluster_ids,
        "cluster_to_indices": cluster_to_indices,
        "cluster_meta": cluster_meta,
        "n_clusters_total": int(next_cluster_id),
        "cluster_fraction": float(cluster_fraction),
    }


def compute_cluster_hit_statistics(attn_maps, cluster_summary):
    """Count how often each cluster receives highest attention for a test sample."""
    attn = np.asarray(attn_maps)
    if attn.ndim != 2:
        raise ValueError(f"attn_maps must be 2-D, got shape={attn.shape}")

    cluster_to_indices = cluster_summary["cluster_to_indices"]
    n_test, n_train = attn.shape

    # Validate cluster indices are within attn column range.
    for cid, idx in cluster_to_indices.items():
        if len(idx) == 0:
            continue
        if np.min(idx) < 0 or np.max(idx) >= n_train:
            raise ValueError(f"Cluster {cid} has member index outside [0, {n_train - 1}]")

    cluster_ids = sorted(cluster_to_indices.keys())
    hit_counts = {cid: 0 for cid in cluster_ids}

    for t in range(n_test):
        best_cluster = None
        best_score = -np.inf
        row_attn = attn[t]
        for cid in cluster_ids:
            members = cluster_to_indices[cid]
            if len(members) == 0:
                continue
            score = float(np.sum(row_attn[members]))
            if score > best_score:
                best_score = score
                best_cluster = cid
        if best_cluster is not None:
            hit_counts[best_cluster] += 1

    total_hits = int(sum(hit_counts.values()))
    hit_rates = {
        cid: (float(hit_counts[cid] / total_hits) if total_hits > 0 else 0.0)
        for cid in cluster_ids
    }

    cluster_stats = {}
    for cid in cluster_ids:
        meta = cluster_summary["cluster_meta"].get(cid, {})
        cluster_stats[cid] = {
            "hits": int(hit_counts[cid]),
            "hit_rate": float(hit_rates[cid]),
            "n_members": int(meta.get("n_members", 0)),
            "label": meta.get("label"),
            "local_cluster_id": meta.get("local_cluster_id"),
        }

    return {
        "total_test_samples": int(n_test),
        "total_hits": total_hits,
        "cluster_stats": cluster_stats,
    }


def plot_cluster_hit_histogram(cluster_hit_stats, output_path, title="Cluster hit-rate histogram"):
    """Plot a histogram/bar chart of cluster hit rates."""
    stats = cluster_hit_stats["cluster_stats"]
    if not stats:
        return

    cluster_ids = sorted(stats.keys())
    hit_rates = np.array([stats[cid]["hit_rate"] for cid in cluster_ids], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(cluster_ids)), hit_rates, color="tab:blue", alpha=0.85)
    plt.xlabel("Cluster index (sorted by id)")
    plt.ylabel("Hit rate")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_hit_histogram_per_class(
    cluster_hit_stats,
    output_path,
    title="Cluster hit-rate histogram by class",
):
    """Plot hit-rate bars split by cluster class label (one subplot per class)."""
    stats = cluster_hit_stats.get("cluster_stats", {})
    if not stats:
        return

    class_to_items = {}
    for cid, info in stats.items():
        label = info.get("label")
        if label not in class_to_items:
            class_to_items[label] = []
        class_to_items[label].append((cid, info))

    class_labels = list(class_to_items.keys())
    n_classes = len(class_labels)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5), squeeze=False)
    axes = axes.ravel()

    for ax, label in zip(axes, class_labels):
        items = sorted(class_to_items[label], key=lambda x: x[0])
        rates = np.array([x[1]["hit_rate"] for x in items], dtype=float)
        ids = [x[0] for x in items]
        x_pos = np.arange(len(ids))

        ax.bar(x_pos, rates, color="tab:orange", alpha=0.85)
        ax.set_title(f"Class {label}")
        ax.set_xlabel("Cluster index")
        ax.set_ylabel("Hit rate")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(i) for i in ids], rotation=90, fontsize=7)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resolve_subset_size(k, n_total):
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if k > 1:
        return min(int(k), int(n_total))
    return max(1, int(round(k * n_total)))


def _allocate_proportional_integer_budget(total_budget, keys, weights):
    """Allocate integer budget across keys proportional to nonnegative weights."""
    keys = list(keys)
    if total_budget <= 0 or len(keys) == 0:
        return {k: 0 for k in keys}

    w = np.asarray(weights, dtype=float)
    w = np.clip(w, a_min=0.0, a_max=None)
    if w.sum() <= 0:
        w = np.ones_like(w)

    raw = total_budget * (w / w.sum())
    base = np.floor(raw).astype(int)
    remainder = int(total_budget - base.sum())

    frac = raw - base
    order = np.argsort(frac)[::-1]
    for i in range(remainder):
        base[order[i % len(order)]] += 1

    return {keys[i]: int(base[i]) for i in range(len(keys))}


def _class_counts(train_labels):
    labels = np.asarray(train_labels)
    unique = np.unique(labels)
    return unique, {lbl: int(np.sum(labels == lbl)) for lbl in unique}


def _build_classwise_clusters_with_class_counts(
    train_row_embeddings,
    train_labels,
    class_cluster_counts,
    random_state=0,
):
    """Build classwise clusters where each class has a specified cluster count."""
    x = np.asarray(train_row_embeddings)
    labels = np.asarray(train_labels)
    if x.ndim != 2:
        raise ValueError(f"train_row_embeddings must be 2-D, got shape={x.shape}")
    if labels.ndim != 1:
        raise ValueError(f"train_labels must be 1-D, got shape={labels.shape}")
    if x.shape[0] != labels.shape[0]:
        raise ValueError(f"Row count mismatch: embeddings={x.shape[0]} vs labels={labels.shape[0]}")

    n_train = x.shape[0]
    cluster_ids = np.full(n_train, -1, dtype=int)
    cluster_to_indices = {}
    cluster_meta = {}
    next_cluster_id = 0

    for label in np.unique(labels):
        class_idx = np.where(labels == label)[0]
        n_class = class_idx.size
        if n_class == 0:
            continue

        requested = int(class_cluster_counts.get(label, 0))
        n_clusters = min(n_class, max(1, requested))

        if n_clusters == 1:
            local_assign = np.zeros(n_class, dtype=int)
        else:
            km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            local_assign = km.fit_predict(x[class_idx])

        label_key = label.item() if hasattr(label, "item") else label
        for local_cid in range(n_clusters):
            members = class_idx[local_assign == local_cid]
            global_cid = next_cluster_id
            next_cluster_id += 1
            cluster_to_indices[global_cid] = members
            cluster_ids[members] = global_cid
            cluster_meta[global_cid] = {
                "label": label_key,
                "n_members": int(members.size),
                "local_cluster_id": int(local_cid),
            }

    return {
        "cluster_ids": cluster_ids,
        "cluster_to_indices": cluster_to_indices,
        "cluster_meta": cluster_meta,
        "n_clusters_total": int(next_cluster_id),
    }


def select_by_classwise_cluster_representatives(
    train_row_embeddings,
    train_labels,
    attention_weights,
    k,
    random_state=0,
):
    """Method 1: #clusters equals subset size; pick top-attended row from each cluster.

    Clusters are built separately per class with class-proportional cluster counts.
    """
    labels = np.asarray(train_labels)
    attn = np.asarray(attention_weights, dtype=float)
    n_train = len(labels)
    if attn.shape[0] != n_train:
        raise ValueError(f"attention_weights length {attn.shape[0]} != n_train {n_train}")

    subset_size = _resolve_subset_size(k, n_train)
    unique, class_count_map = _class_counts(labels)

    class_budgets = _allocate_proportional_integer_budget(
        total_budget=subset_size,
        keys=list(unique),
        weights=[class_count_map[lbl] for lbl in unique],
    )

    cluster_summary = _build_classwise_clusters_with_class_counts(
        train_row_embeddings=train_row_embeddings,
        train_labels=labels,
        class_cluster_counts=class_budgets,
        random_state=random_state,
    )

    selected = []
    per_cluster_pick = {}
    for cid, members in cluster_summary["cluster_to_indices"].items():
        if len(members) == 0:
            continue
        local_best = members[np.argmax(attn[members])]
        selected.append(int(local_best))
        per_cluster_pick[cid] = int(local_best)

    # Guard in case of rare empty clusters; fill from globally highest-attended remaining points.
    if len(selected) < subset_size:
        remaining_pool = np.setdiff1d(np.arange(n_train), np.asarray(selected, dtype=int))
        top_remaining = remaining_pool[np.argsort(attn[remaining_pool])[::-1]]
        need = subset_size - len(selected)
        selected.extend(top_remaining[:need].astype(int).tolist())

    selected = np.asarray(selected[:subset_size], dtype=int)
    return {
        "subset_indices": selected,
        "subset_size": int(subset_size),
        "cluster_summary": cluster_summary,
        "class_cluster_budget": {k.item() if hasattr(k, "item") else k: int(v) for k, v in class_budgets.items()},
        "picked_representatives": per_cluster_pick,
    }


def select_by_classwise_cluster_hit_proportional(
    train_row_embeddings,
    train_labels,
    attention_weights,
    attn_maps,
    k,
    cluster_fraction=0.1,
    random_state=0,
):
    """Method 2: sample per-cluster counts proportional to cluster hits, class-balanced overall.

    - Clusters are built classwise.
    - Total subset budget determined by k.
    - Class budgets are proportional to class sizes (maintains class balance).
    - Within each class, cluster allocations are proportional to cluster hit counts.
    - Selected rows are highest-attended rows within each cluster.
    """
    labels = np.asarray(train_labels)
    attn = np.asarray(attention_weights, dtype=float)
    n_train = len(labels)
    if attn.shape[0] != n_train:
        raise ValueError(f"attention_weights length {attn.shape[0]} != n_train {n_train}")

    subset_size = _resolve_subset_size(k, n_train)
    unique, class_count_map = _class_counts(labels)
    class_budgets = _allocate_proportional_integer_budget(
        total_budget=subset_size,
        keys=list(unique),
        weights=[class_count_map[lbl] for lbl in unique],
    )

    cluster_summary = build_classwise_clusters(
        train_row_embeddings=train_row_embeddings,
        train_labels=labels,
        cluster_fraction=cluster_fraction,
        random_state=random_state,
    )
    hit_stats = compute_cluster_hit_statistics(attn_maps=attn_maps, cluster_summary=cluster_summary)

    selected = []
    class_alloc_debug = {}
    for label in unique:
        label_key = label.item() if hasattr(label, "item") else label
        class_budget = int(class_budgets[label])
        if class_budget <= 0:
            class_alloc_debug[label_key] = {"budget": 0, "cluster_alloc": {}}
            continue

        class_cluster_ids = [
            cid for cid, meta in cluster_summary["cluster_meta"].items()
            if meta.get("label") == label_key
        ]
        if not class_cluster_ids:
            class_alloc_debug[label_key] = {"budget": class_budget, "cluster_alloc": {}}
            continue

        class_hits = [hit_stats["cluster_stats"].get(cid, {}).get("hits", 0) for cid in class_cluster_ids]
        if sum(class_hits) <= 0:
            # fallback to cluster size weighting when hits are all zero
            class_hits = [cluster_summary["cluster_meta"][cid]["n_members"] for cid in class_cluster_ids]

        cluster_alloc = _allocate_proportional_integer_budget(
            total_budget=class_budget,
            keys=class_cluster_ids,
            weights=class_hits,
        )

        # Pick top-attended rows within each allocated cluster.
        class_selected = []
        for cid in class_cluster_ids:
            n_pick = int(cluster_alloc[cid])
            if n_pick <= 0:
                continue
            members = np.asarray(cluster_summary["cluster_to_indices"][cid])
            if members.size == 0:
                continue
            order = members[np.argsort(attn[members])[::-1]]
            class_selected.extend(order[:n_pick].astype(int).tolist())

        # If duplicates/shortfalls happen, top-up from remaining rows of same class by attention.
        class_selected = list(dict.fromkeys(class_selected))
        if len(class_selected) < class_budget:
            class_idx = np.where(labels == label)[0]
            remaining = np.setdiff1d(class_idx, np.asarray(class_selected, dtype=int))
            top_remaining = remaining[np.argsort(attn[remaining])[::-1]]
            need = class_budget - len(class_selected)
            class_selected.extend(top_remaining[:need].astype(int).tolist())

        selected.extend(class_selected[:class_budget])
        class_alloc_debug[label_key] = {
            "budget": class_budget,
            "cluster_alloc": {int(k): int(v) for k, v in cluster_alloc.items()},
        }

    selected = np.asarray(selected[:subset_size], dtype=int)
    return {
        "subset_indices": selected,
        "subset_size": int(subset_size),
        "cluster_summary": cluster_summary,
        "hit_stats": hit_stats,
        "class_budget": {k.item() if hasattr(k, "item") else k: int(v) for k, v in class_budgets.items()},
        "class_cluster_allocation": class_alloc_debug,
    }


def select_by_classwise_cluster_attention_weighted(
    train_row_embeddings,
    train_labels,
    attn_sums_per_class,
    k,
    cluster_fraction=0.1,
    random_state=0,
):
    """Two-stage class-conditional sampling: sample cluster, then sample row.

    Sampling is fully class-conditional and class-stratified:
    - Class budgets follow the train class distribution.
    - For each class, cluster sampling probabilities are proportional to
      cluster-level summed attention from test samples of the same class.
    - Within the sampled cluster, row sampling probabilities are proportional
      to row-level summed attention from test samples of the same class.
    - Sampling is without replacement.
    """
    labels = np.asarray(train_labels)
    rng = np.random.RandomState(random_state)
    n_train = len(labels)
    subset_size = _resolve_subset_size(k, n_train)

    unique_labels, class_count_map = _class_counts(labels)
    class_budgets = _allocate_proportional_integer_budget(
        total_budget=subset_size,
        keys=list(unique_labels),
        weights=[class_count_map[lbl] for lbl in unique_labels],
    )

    cluster_summary = build_classwise_clusters(
        train_row_embeddings=train_row_embeddings,
        train_labels=labels,
        cluster_fraction=cluster_fraction,
        random_state=random_state,
    )

    selected = []
    class_alloc_debug = {}

    for label in unique_labels:
        label_key = label.item() if hasattr(label, "item") else label
        class_budget = int(class_budgets[label])
        if class_budget <= 0:
            class_alloc_debug[label_key] = {
                "budget": 0,
                "selected": 0,
                "sampled_cluster_counts": {},
            }
            continue

        if label not in attn_sums_per_class:
            class_attn = np.zeros(n_train, dtype=float)
        else:
            class_attn = np.asarray(attn_sums_per_class[label], dtype=float)

        class_cluster_ids = [
            cid for cid, meta in cluster_summary["cluster_meta"].items()
            if meta.get("label") == label_key
        ]

        available_per_cluster = {
            cid: set(np.asarray(cluster_summary["cluster_to_indices"][cid], dtype=int).tolist())
            for cid in class_cluster_ids
        }
        sampled_cluster_counts = {int(cid): 0 for cid in class_cluster_ids}

        class_selected = []
        remaining = class_budget

        while remaining > 0:
            non_empty_clusters = [cid for cid in class_cluster_ids if len(available_per_cluster[cid]) > 0]
            if not non_empty_clusters:
                break

            # Stage 1: sample cluster by class-matched mean attention.
            cluster_weights = []
            for cid in non_empty_clusters:
                members = np.fromiter(available_per_cluster[cid], dtype=int)
                #cluster_weights.append(float(np.mean(class_attn[members])) if members.size > 0 else 0.0)
                #cluster_weights.append(float(np.sum(class_attn[members])) if members.size > 0 else 0.0)
                if members.size > 0:
                    attention_sum = float(np.sum(class_attn[members]))
                    cluster_size = float(members.size)
                    cluster_score = attention_sum / np.sqrt(cluster_size)
                else:
                    cluster_score = 0.0
                cluster_weights.append(cluster_score)
            cluster_probs = normalize_weights(cluster_weights)
            chosen_cluster = rng.choice(non_empty_clusters, p=cluster_probs)

            # Stage 2: sample row inside chosen cluster by class-matched attention.
            cluster_members = np.fromiter(available_per_cluster[chosen_cluster], dtype=int)
            row_weights = class_attn[cluster_members]
            row_probs = normalize_weights(row_weights)
            picked_row = int(rng.choice(cluster_members, p=row_probs))

            available_per_cluster[chosen_cluster].remove(picked_row)
            class_selected.append(picked_row)
            sampled_cluster_counts[int(chosen_cluster)] += 1
            remaining -= 1

        # Top up within-class if needed (e.g., due to cluster exhaustion).
        if len(class_selected) < class_budget:
            class_idx = np.where(labels == label)[0]
            remaining_pool = np.setdiff1d(class_idx, np.asarray(class_selected, dtype=int), assume_unique=False)
            if remaining_pool.size > 0:
                remaining_probs = normalize_weights(class_attn[remaining_pool])
                need = min(class_budget - len(class_selected), remaining_pool.size)
                extra = rng.choice(remaining_pool, size=need, replace=False, p=remaining_probs)
                class_selected.extend(extra.astype(int).tolist())

        selected.extend(class_selected[:class_budget])
        class_alloc_debug[label_key] = {
            "budget": int(class_budget),
            "selected": int(min(len(class_selected), class_budget)),
            "sampled_cluster_counts": sampled_cluster_counts,
        }

    selected = np.asarray(selected[:subset_size], dtype=int)
    return {
        "subset_indices": selected,
        "subset_size": int(subset_size),
        "cluster_summary": cluster_summary,
        "class_budget": {k.item() if hasattr(k, "item") else k: int(v) for k, v in class_budgets.items()},
        "class_cluster_sampling": class_alloc_debug,
    }


def select_by_classwise_cluster_uniform_cluster_attention_row(
    train_row_embeddings,
    train_labels,
    attn_sums_per_class,
    k,
    cluster_fraction=0.1,
    random_state=0,
):
    """Two-stage class-conditional sampling with uniform cluster choice.

    - Class budgets are stratified by train class frequency.
    - Stage 1: sample a non-empty cluster uniformly within each class.
    - Stage 2: sample a row within that cluster using same-class attention weights.
    - Sampling is without replacement.
    """
    labels = np.asarray(train_labels)
    rng = np.random.RandomState(random_state)
    n_train = len(labels)
    subset_size = _resolve_subset_size(k, n_train)

    unique_labels, class_count_map = _class_counts(labels)
    class_budgets = _allocate_proportional_integer_budget(
        total_budget=subset_size,
        keys=list(unique_labels),
        weights=[class_count_map[lbl] for lbl in unique_labels],
    )

    cluster_summary = build_classwise_clusters(
        train_row_embeddings=train_row_embeddings,
        train_labels=labels,
        cluster_fraction=cluster_fraction,
        random_state=random_state,
    )

    selected = []
    class_alloc_debug = {}

    for label in unique_labels:
        label_key = label.item() if hasattr(label, "item") else label
        class_budget = int(class_budgets[label])
        if class_budget <= 0:
            class_alloc_debug[label_key] = {
                "budget": 0,
                "selected": 0,
                "sampled_cluster_counts": {},
            }
            continue

        if label not in attn_sums_per_class:
            class_attn = np.zeros(n_train, dtype=float)
        else:
            class_attn = np.asarray(attn_sums_per_class[label], dtype=float)

        class_cluster_ids = [
            cid for cid, meta in cluster_summary["cluster_meta"].items()
            if meta.get("label") == label_key
        ]

        available_per_cluster = {
            cid: set(np.asarray(cluster_summary["cluster_to_indices"][cid], dtype=int).tolist())
            for cid in class_cluster_ids
        }
        sampled_cluster_counts = {int(cid): 0 for cid in class_cluster_ids}

        class_selected = []
        remaining = class_budget

        while remaining > 0:
            non_empty_clusters = [cid for cid in class_cluster_ids if len(available_per_cluster[cid]) > 0]
            if not non_empty_clusters:
                break

            # Stage 1: uniform cluster sampling within class.
            chosen_cluster = rng.choice(non_empty_clusters)

            # Stage 2: attention-weighted row sampling within cluster.
            cluster_members = np.fromiter(available_per_cluster[chosen_cluster], dtype=int)
            row_weights = class_attn[cluster_members]
            row_probs = normalize_weights(row_weights)
            picked_row = int(rng.choice(cluster_members, p=row_probs))

            available_per_cluster[chosen_cluster].remove(picked_row)
            class_selected.append(picked_row)
            sampled_cluster_counts[int(chosen_cluster)] += 1
            remaining -= 1

        # Top up within class if needed.
        if len(class_selected) < class_budget:
            class_idx = np.where(labels == label)[0]
            remaining_pool = np.setdiff1d(class_idx, np.asarray(class_selected, dtype=int), assume_unique=False)
            if remaining_pool.size > 0:
                remaining_probs = normalize_weights(class_attn[remaining_pool])
                need = min(class_budget - len(class_selected), remaining_pool.size)
                extra = rng.choice(remaining_pool, size=need, replace=False, p=remaining_probs)
                class_selected.extend(extra.astype(int).tolist())

        selected.extend(class_selected[:class_budget])
        class_alloc_debug[label_key] = {
            "budget": int(class_budget),
            "selected": int(min(len(class_selected), class_budget)),
            "sampled_cluster_counts": sampled_cluster_counts,
        }

    selected = np.asarray(selected[:subset_size], dtype=int)
    return {
        "subset_indices": selected,
        "subset_size": int(subset_size),
        "cluster_summary": cluster_summary,
        "class_budget": {k.item() if hasattr(k, "item") else k: int(v) for k, v in class_budgets.items()},
        "class_cluster_sampling": class_alloc_debug,
    }
