import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .common import adjust_probs_for_single_class, to_label_array
from .modeling import fit_eval_subset_metrics


def label_balance_dict(labels):
    values, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    return {
        value.item() if hasattr(value, "item") else value: {
            "count": int(count),
            "ratio": float(count / total) if total > 0 else 0.0,
        }
        for value, count in zip(values, counts)
    }


def compute_hit_statistics(attn_maps, train_labels, y_pred_labels, hit_count_mode="predicted_class"):
    hit_counts = np.zeros(train_labels.shape[0], dtype=int)
    total_preds = {label.item() if hasattr(label, "item") else label: 0 for label in np.unique(train_labels)}
    class_to_train_indices = {
        label.item() if hasattr(label, "item") else label: np.where(train_labels == label)[0]
        for label in np.unique(train_labels)
    }

    if hit_count_mode not in {"predicted_class", "all_classes"}:
        raise ValueError(
            f"Invalid hit_count_mode={hit_count_mode}. Expected 'predicted_class' or 'all_classes'."
        )

    for i, pred_label in enumerate(y_pred_labels):
        if hit_count_mode == "predicted_class":
            labels_to_score = [pred_label]
        else:
            labels_to_score = list(class_to_train_indices.keys())

        for label in labels_to_score:
            label_key = label.item() if hasattr(label, "item") else label
            class_indices = class_to_train_indices.get(label_key)
            if class_indices is None or class_indices.size == 0:
                continue
            total_preds[label_key] = total_preds.get(label_key, 0) + 1
            best_local = np.argmax(attn_maps[i, class_indices])
            hit_counts[class_indices[best_local]] += 1

    top_hits_per_class = {}
    for label_key, class_indices in class_to_train_indices.items():
        predicted_as_class = total_preds.get(label_key, 0)
        cls_hits = hit_counts[class_indices]
        if cls_hits.size == 0:
            top_hits_per_class[label_key] = {
                "predicted_count": int(predicted_as_class),
                "top_hits": [],
            }
            continue

        top_order = np.argsort(cls_hits)[::-1][:3]
        top_hits_per_class[label_key] = {
            "predicted_count": int(predicted_as_class),
            "top_hits": [
                {
                    "train_index": int(class_indices[j]),
                    "hits": int(cls_hits[j]),
                    "hit_ratio": float(cls_hits[j] / predicted_as_class) if predicted_as_class > 0 else 0.0,
                }
                for j in top_order
            ],
        }

    return hit_counts, total_preds, top_hits_per_class


def aggregate_attention_by_class(attn_maps, train_labels, attn_aggregation):
    class_labels = np.unique(train_labels)
    class_masks = [train_labels == label for label in class_labels]
    if attn_aggregation == "sum":
        label_mean_attn = np.column_stack([attn_maps[:, class_mask].sum(axis=1) for class_mask in class_masks])
    else:
        label_mean_attn = np.column_stack([attn_maps[:, class_mask].mean(axis=1) for class_mask in class_masks])
    return label_mean_attn, class_labels


def rankdata_simple(values):
    order = np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return ranks


def _attention_mass_by_density_quintile(attn_received, mean_knn_distance):
    """Compute attention share by density-distance quintile (dense -> outlier)."""
    valid_idx = np.where(np.isfinite(attn_received) & np.isfinite(mean_knn_distance))[0]
    if valid_idx.size == 0:
        return {
            "bins": [],
            "top_20pct_most_isolated_attention_share": float("nan"),
            "n_valid_samples": 0,
        }

    order = valid_idx[np.argsort(mean_knn_distance[valid_idx])]  # dense to sparse
    groups = np.array_split(order, 5)

    total_attn = float(np.sum(attn_received[valid_idx]))
    labels = ["0-20% (dense)", "20-40%", "40-60%", "60-80%", "80-100% (outliers)"]

    bins = []
    for i, group in enumerate(groups):
        if group.size == 0 or total_attn <= 0:
            share = float("nan")
        else:
            share = float(np.sum(attn_received[group]) / total_attn)
        bins.append(
            {
                "label": labels[i],
                "quantile_range": f"{i*20}-{(i+1)*20}%",
                "n_samples": int(group.size),
                "attention_share": share,
            }
        )

    top_20_share = bins[-1]["attention_share"] if bins else float("nan")
    return {
        "bins": bins,
        "top_20pct_most_isolated_attention_share": top_20_share,
        "n_valid_samples": int(valid_idx.size),
    }


def _attention_mass_by_density_quintile_per_class(attn_received, mean_knn_distance, train_labels):
    """Compute density-distance quintiles inside each class, then aggregate shares across classes."""
    labels = ["0-20% (dense)", "20-40%", "40-60%", "60-80%", "80-100% (outliers)"]
    per_class = {}
    bin_share_lists = [[] for _ in range(5)]
    bin_share_weighted = [[] for _ in range(5)]
    top20_vals = []
    top20_weighted = []

    for label in np.unique(train_labels):
        class_idx = np.where(train_labels == label)[0]
        valid_cls = class_idx[np.isfinite(attn_received[class_idx]) & np.isfinite(mean_knn_distance[class_idx])]
        label_key = label.item() if hasattr(label, "item") else label

        if valid_cls.size == 0:
            per_class[label_key] = {
                "n_valid_samples": 0,
                "bins": [],
                "top_20pct_most_isolated_attention_share": float("nan"),
            }
            continue

        order = valid_cls[np.argsort(mean_knn_distance[valid_cls])]  # dense to sparse within class
        groups = np.array_split(order, 5)
        total_attn = float(np.sum(attn_received[valid_cls]))

        bins = []
        for i, group in enumerate(groups):
            if group.size == 0 or total_attn <= 0:
                share = float("nan")
            else:
                share = float(np.sum(attn_received[group]) / total_attn)

            bins.append(
                {
                    "label": labels[i],
                    "quantile_range": f"{i*20}-{(i+1)*20}%",
                    "n_samples": int(group.size),
                    "attention_share": share,
                }
            )

            if np.isfinite(share):
                bin_share_lists[i].append(share)
                bin_share_weighted[i].append((share, int(valid_cls.size)))

        top20 = bins[-1]["attention_share"] if bins else float("nan")
        if np.isfinite(top20):
            top20_vals.append(top20)
            top20_weighted.append((top20, int(valid_cls.size)))

        per_class[label_key] = {
            "n_valid_samples": int(valid_cls.size),
            "bins": bins,
            "top_20pct_most_isolated_attention_share": top20,
        }

    aggregate_bins = []
    for i in range(5):
        if bin_share_lists[i]:
            mean_share = float(np.mean(bin_share_lists[i]))
            total_w = sum(w for _, w in bin_share_weighted[i])
            weighted_share = (
                float(sum(v * w for v, w in bin_share_weighted[i]) / total_w) if total_w > 0 else float("nan")
            )
        else:
            mean_share = float("nan")
            weighted_share = float("nan")

        aggregate_bins.append(
            {
                "label": labels[i],
                "quantile_range": f"{i*20}-{(i+1)*20}%",
                "attention_share_mean": mean_share,
                "attention_share_weighted_mean": weighted_share,
            }
        )

    if top20_vals:
        top20_mean = float(np.mean(top20_vals))
        total_w = sum(w for _, w in top20_weighted)
        top20_weighted_mean = (
            float(sum(v * w for v, w in top20_weighted) / total_w) if total_w > 0 else float("nan")
        )
    else:
        top20_mean = float("nan")
        top20_weighted_mean = float("nan")

    return {
        "per_class": per_class,
        "aggregate_bins": aggregate_bins,
        "top_20pct_most_isolated_attention_share_mean": top20_mean,
        "top_20pct_most_isolated_attention_share_weighted_mean": top20_weighted_mean,
    }


def compute_outlier_attention_stats(
    attn_maps,
    train_row_embeddings,
    train_labels=None,
    top_fracs=(0.01, 0.05, 0.10),
    n_neighbors=20,
):
    """Measure whether high-attention train samples lie in low-density embedding regions.

    Density is computed within each sample's true class only.
    """
    if train_row_embeddings is None:
        return {"enabled": False, "reason": "missing_train_row_embeddings"}

    x = np.asarray(train_row_embeddings)
    if x.ndim != 2 or x.shape[0] == 0:
        return {"enabled": False, "reason": "invalid_train_row_embeddings"}

    n_train = x.shape[0]
    if attn_maps.shape[1] != n_train:
        return {
            "enabled": False,
            "reason": "shape_mismatch",
            "n_train_embeddings": int(n_train),
            "n_train_attn": int(attn_maps.shape[1]),
        }

    if train_labels is None:
        return {"enabled": False, "reason": "missing_train_labels_for_class_conditional_density"}

    train_labels = to_label_array(train_labels)
    if len(train_labels) != n_train:
        return {
            "enabled": False,
            "reason": "train_label_length_mismatch",
            "n_train_embeddings": int(n_train),
            "n_train_labels": int(len(train_labels)),
        }

    # Compute neighborhood sparsity inside each class only.
    mean_knn_distance = np.full(n_train, np.nan, dtype=float)
    for label in np.unique(train_labels):
        class_idx = np.where(train_labels == label)[0]
        if class_idx.size <= 1:
            continue

        class_x = x[class_idx]
        class_k = min(max(2, n_neighbors + 1), class_idx.size)
        nn = NearestNeighbors(n_neighbors=class_k, metric="euclidean")
        nn.fit(class_x)
        class_distances, _ = nn.kneighbors(class_x, return_distance=True)
        mean_knn_distance[class_idx] = class_distances[:, 1:].mean(axis=1)

    local_density = np.full(n_train, np.nan, dtype=float)
    valid_dist_mask = np.isfinite(mean_knn_distance)
    local_density[valid_dist_mask] = 1.0 / (mean_knn_distance[valid_dist_mask] + 1e-12)

    attn_received = attn_maps.sum(axis=0)
    attn_order_desc = np.argsort(attn_received)[::-1]

    global_mean_density = float(np.nanmean(local_density))
    global_mean_knn_dist = float(np.nanmean(mean_knn_distance))

    top_slice_stats = {}
    for frac in top_fracs:
        n_top = min(n_train, max(1, int(round(frac * n_train))))
        top_idx = attn_order_desc[:n_top]

        top_mean_density = float(np.nanmean(local_density[top_idx]))
        top_mean_knn_dist = float(np.nanmean(mean_knn_distance[top_idx]))

        key = f"top_{int(frac * 100)}pct"
        top_slice_stats[key] = {
            "n_top": int(n_top),
            "mean_density": top_mean_density,
            "mean_knn_distance": top_mean_knn_dist,
            "density_ratio_vs_global": (
                float(top_mean_density / global_mean_density) if global_mean_density > 0 else float("nan")
            ),
            "knn_distance_ratio_vs_global": (
                float(top_mean_knn_dist / global_mean_knn_dist) if global_mean_knn_dist > 0 else float("nan")
            ),
        }

    valid_corr_mask = np.isfinite(attn_received) & np.isfinite(mean_knn_distance)
    attn_valid = attn_received[valid_corr_mask]
    dist_valid = mean_knn_distance[valid_corr_mask]
    attn_ranks = rankdata_simple(attn_valid).astype(float) if attn_valid.size > 0 else np.array([])
    dist_ranks = rankdata_simple(dist_valid).astype(float) if dist_valid.size > 0 else np.array([])
    if np.std(attn_ranks) > 0 and np.std(dist_ranks) > 0:
        spearman_attn_vs_knn_dist = float(np.corrcoef(attn_ranks, dist_ranks)[0, 1])
    else:
        spearman_attn_vs_knn_dist = float("nan")

    per_class_corr = {}
    per_class_vals = []
    per_class_weighted = []
    for label in np.unique(train_labels):
        class_idx = np.where(train_labels == label)[0]
        label_key = label.item() if hasattr(label, "item") else label
        valid_cls = class_idx[np.isfinite(mean_knn_distance[class_idx])]
        if valid_cls.size < 2:
            corr_val = float("nan")
        else:
            cls_attn_ranks = rankdata_simple(attn_received[valid_cls]).astype(float)
            cls_dist_ranks = rankdata_simple(mean_knn_distance[valid_cls]).astype(float)
            if np.std(cls_attn_ranks) > 0 and np.std(cls_dist_ranks) > 0:
                corr_val = float(np.corrcoef(cls_attn_ranks, cls_dist_ranks)[0, 1])
            else:
                corr_val = float("nan")

        per_class_corr[label_key] = {
            "n_train_class": int(class_idx.size),
            "n_valid_density_class": int(valid_cls.size),
            "spearman_attn_vs_knn_distance": corr_val,
        }
        if np.isfinite(corr_val):
            per_class_vals.append(corr_val)
            per_class_weighted.append((corr_val, int(valid_cls.size)))

    if per_class_vals:
        per_class_mean = float(np.mean(per_class_vals))
        total_weight = sum(weight for _, weight in per_class_weighted)
        per_class_weighted_mean = (
            float(sum(val * weight for val, weight in per_class_weighted) / total_weight)
            if total_weight > 0
            else float("nan")
        )
    else:
        per_class_mean = float("nan")
        per_class_weighted_mean = float("nan")

    density_quintile_attention = _attention_mass_by_density_quintile(
        attn_received=attn_received,
        mean_knn_distance=mean_knn_distance,
    )
    density_quintile_attention_per_class = _attention_mass_by_density_quintile_per_class(
        attn_received=attn_received,
        mean_knn_distance=mean_knn_distance,
        train_labels=train_labels,
    )

    return {
        "enabled": True,
        "n_train": int(n_train),
        "n_neighbors_density_target": int(max(1, n_neighbors)),
        "global_mean_density": global_mean_density,
        "global_mean_knn_distance": global_mean_knn_dist,
        "spearman_attn_vs_knn_distance": spearman_attn_vs_knn_dist,
        "spearman_attn_vs_knn_distance_per_class": per_class_corr,
        "spearman_attn_vs_knn_distance_per_class_mean": per_class_mean,
        "spearman_attn_vs_knn_distance_per_class_weighted_mean": per_class_weighted_mean,
        "density_quintile_attention": density_quintile_attention,
        "density_quintile_attention_per_class": density_quintile_attention_per_class,
        "top_slices": top_slice_stats,
    }


def compute_full_model_stats(
    classifier,
    x_test_transformed,
    attn_maps,
    train_labels,
    y_test,
    attn_aggregation,
    pos_label,
    train_row_embeddings=None,
    hit_count_mode="predicted_class",
):
    y_probs = classifier.predict_proba(x_test_transformed)
    y_pred_idx = np.argmax(y_probs, axis=1)
    y_pred = classifier.y_encoder_.inverse_transform(y_pred_idx)
    y_pred_labels = to_label_array(y_pred)
    y_true_labels = to_label_array(y_test)

    hit_counts, total_preds, top_hits_per_class = compute_hit_statistics(
        attn_maps=attn_maps,
        train_labels=train_labels,
        y_pred_labels=y_pred_labels,
        hit_count_mode=hit_count_mode,
    )
    unique_true_labels = np.unique(y_true_labels)
    attn_sums_per_class = {label: attn_maps[y_true_labels == label, :].sum(axis=0) for label in unique_true_labels}

    label_mean_attn, class_labels = aggregate_attention_by_class(
        attn_maps=attn_maps,
        train_labels=train_labels,
        attn_aggregation=attn_aggregation,
    )
    mean_attn_pred_labels = class_labels[np.argmax(label_mean_attn, axis=1)]

    train_label_balance = label_balance_dict(train_labels)

    y_probs_adj = adjust_probs_for_single_class(y_probs, train_labels, pos_label)
    full_accuracy = float(accuracy_score(y_test, y_pred))
    full_f1 = float(f1_score(y_test, y_pred, zero_division=0, pos_label=pos_label, average="binary"))
    full_roc_auc = float(roc_auc_score(y_test, y_probs_adj[:, 1]))

    full_model_stats = {
        "accuracy": full_accuracy,
        "f1": full_f1,
        "roc_auc": full_roc_auc,
        "hit_count_mode": hit_count_mode,
        "train_label_balance": train_label_balance,
        "mean_attn_vs_pred_acc": float(np.mean(mean_attn_pred_labels == y_pred_labels)),
        "mean_attn_vs_true_acc": float(np.mean(mean_attn_pred_labels == y_true_labels)),
        "pred_vs_true_acc": float(np.mean(y_pred_labels == y_true_labels)),
        "mean_attn_f1": float(
            f1_score(y_true_labels, mean_attn_pred_labels, zero_division=0, pos_label=pos_label, average="binary")
        ),
        "pred_f1": float(
            f1_score(y_true_labels, y_pred_labels, zero_division=0, pos_label=pos_label, average="binary")
        ),
        "highest_attn_hit_counts": hit_counts,
        "highest_attn_total_preds": total_preds,
        "highest_attn_top3_per_class": top_hits_per_class,
        "attention_outlier_analysis": compute_outlier_attention_stats(
            attn_maps=attn_maps,
            train_row_embeddings=train_row_embeddings,
            train_labels=train_labels,
        ),
    }

    return hit_counts, attn_sums_per_class, full_model_stats


def build_refit_on_hit_subset_stats(
    x_subset,
    y_subset,
    x_test,
    y_test,
    hit_counts,
    n_estimators,
    random_state,
    pos_label,
    full_accuracy,
    full_f1,
    full_roc_auc,
    full_label_balance,
):
    keep_mask = hit_counts > 0
    n_kept = int(np.sum(keep_mask))
    n_total = int(len(keep_mask))

    if n_kept > 0:
        x_subset_reduced = x_subset.iloc[keep_mask].reset_index(drop=True)
        y_subset_reduced = y_subset.iloc[keep_mask].reset_index(drop=True)
        reduced_accuracy, reduced_f1, reduced_roc_auc = fit_eval_subset_metrics(
            x_subset_reduced,
            y_subset_reduced,
            x_test,
            y_test,
            n_estimators,
            random_state,
            pos_label,
        )
        reduced_label_balance = label_balance_dict(to_label_array(y_subset_reduced))

        random_state_rng = np.random.RandomState(random_state)
        random_keep_indices = random_state_rng.choice(n_total, size=n_kept, replace=False)
        x_subset_random = x_subset.iloc[random_keep_indices].reset_index(drop=True)
        y_subset_random = y_subset.iloc[random_keep_indices].reset_index(drop=True)
        random_accuracy, random_f1, random_roc_auc = fit_eval_subset_metrics(
            x_subset_random,
            y_subset_random,
            x_test,
            y_test,
            n_estimators,
            random_state,
            pos_label,
        )
        random_label_balance = label_balance_dict(to_label_array(y_subset_random))
    else:
        reduced_accuracy = float("nan")
        reduced_roc_auc = float("nan")
        reduced_f1 = float("nan")
        random_accuracy = float("nan")
        random_roc_auc = float("nan")
        random_f1 = float("nan")
        reduced_label_balance = {}
        random_label_balance = {}

    return {
        "enabled": True,
        "kept_samples": n_kept,
        "total_samples": n_total,
        "compression_ratio": float(n_kept / n_total) if n_total > 0 else 0.0,
        "full_accuracy": float(full_accuracy),
        "full_f1": float(full_f1),
        "full_roc_auc": float(full_roc_auc),
        "full_label_balance": full_label_balance,
        "reduced_accuracy": reduced_accuracy,
        "reduced_f1": reduced_f1,
        "reduced_roc_auc": reduced_roc_auc,
        "reduced_label_balance": reduced_label_balance,
        "random_matched_accuracy": random_accuracy,
        "random_matched_f1": random_f1,
        "random_matched_roc_auc": random_roc_auc,
        "random_matched_label_balance": random_label_balance,
        "f1_delta": float(reduced_f1 - full_f1) if n_kept > 0 else float("nan"),
        "roc_auc_delta": float(reduced_roc_auc - full_roc_auc) if n_kept > 0 else float("nan"),
        "reduced_vs_random_f1_delta": float(reduced_f1 - random_f1) if n_kept > 0 else float("nan"),
        "reduced_vs_random_roc_auc_delta": (
            float(reduced_roc_auc - random_roc_auc) if n_kept > 0 else float("nan")
        ),
    }
