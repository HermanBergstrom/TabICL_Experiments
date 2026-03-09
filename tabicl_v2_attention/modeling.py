import numpy as np
from tabicl import TabICLClassifier
from skrub import TableVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .common import adjust_probs_for_single_class, to_numpy_array


DEFAULT_HEAD = 7


def fit_eval_subset_metrics(
    x_sub,
    y_sub,
    x_test,
    y_test,
    n_estimators,
    random_state,
    pos_label,
):
    vectorizer_sub = TableVectorizer()
    x_sub_transformed = vectorizer_sub.fit_transform(x_sub)
    x_test_sub_transformed = vectorizer_sub.transform(x_test)

    classifier_sub = TabICLClassifier(n_estimators=n_estimators, random_state=random_state)
    classifier_sub.fit(x_sub_transformed, y_sub)

    y_probs_sub = classifier_sub.predict_proba(x_test_sub_transformed)
    y_pred_sub_idx = np.argmax(y_probs_sub, axis=1)
    y_pred_sub = classifier_sub.y_encoder_.inverse_transform(y_pred_sub_idx)
    y_probs_sub = adjust_probs_for_single_class(y_probs_sub, y_sub, pos_label)

    subset_accuracy = float(accuracy_score(y_test, y_pred_sub))
    subset_roc_auc = float(roc_auc_score(y_test, y_probs_sub[:, 1]))
    subset_f1 = float(f1_score(y_test, y_pred_sub, zero_division=0, pos_label=pos_label, average="binary"))
    return subset_accuracy, subset_f1, subset_roc_auc


def extract_embeddings(
    x_data,
    y_data,
    n_estimators,
    random_state,
    batch_size,
    n_folds=10,
):
    print("Extracting row embeddings with cross-validation …")
    n_samples = x_data.shape[0]
    if n_samples == 0:
        return np.empty((0, 0), dtype=np.float32)

    if n_samples == 1:
        vectorizer = TableVectorizer()
        x_data_transformed = vectorizer.fit_transform(x_data)
        classifier = TabICLClassifier(n_estimators=n_estimators, random_state=random_state)
        classifier.fit(x_data_transformed, y_data)

        _, _, row_embeddings = classifier.predict(
            x_data_transformed,
            return_attn=True,
            return_row_representations=True,
        )
        row_embeddings_np = to_numpy_array(row_embeddings)
        if row_embeddings_np.ndim == 3:
            row_embeddings_np = row_embeddings_np[0]

        return row_embeddings_np[1:]

    n_folds = max(1, min(int(n_folds), n_samples))
    folds = np.array_split(np.arange(n_samples), n_folds)

    embeddings_out = None

    for fold_id, test_idx in enumerate(folds):
        if test_idx.size == 0:
            continue

        train_idx = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_id])

        x_train_fold = x_data.iloc[train_idx].reset_index(drop=True)
        y_train_fold = y_data.iloc[train_idx].reset_index(drop=True)
        x_test_fold = x_data.iloc[test_idx].reset_index(drop=True)

        vectorizer = TableVectorizer()
        x_train_fold_transformed = vectorizer.fit_transform(x_train_fold)
        x_test_fold_transformed = vectorizer.transform(x_test_fold)

        classifier_fold = TabICLClassifier(
            n_estimators=n_estimators,
            random_state=random_state + fold_id,
        )
        classifier_fold.fit(x_train_fold_transformed, y_train_fold)

        n_train_fold = x_train_fold_transformed.shape[0]
        fold_test_embeddings = []

        for start in range(0, x_test_fold_transformed.shape[0], batch_size):
            stop = min(start + batch_size, x_test_fold_transformed.shape[0])
            chunk = x_test_fold_transformed[start:stop]

            _, _, row_embeddings = classifier_fold.predict(
                chunk,
                return_attn=True,
                return_row_representations=True,
            )

            row_embeddings_np = to_numpy_array(row_embeddings)
            if row_embeddings_np.ndim == 3:
                row_embeddings_np = row_embeddings_np[0]

            # Keep only held-out (test) rows from this fold.
            fold_test_embeddings.append(row_embeddings_np[n_train_fold:])

        fold_test_embeddings = np.concatenate(fold_test_embeddings, axis=0)

        if embeddings_out is None:
            emb_dim = fold_test_embeddings.shape[1]
            embeddings_out = np.empty((n_samples, emb_dim), dtype=fold_test_embeddings.dtype)

        embeddings_out[test_idx] = fold_test_embeddings
    print("Finished extracting row embeddings.")
    return embeddings_out


def extract_attention_data(classifier, x_test_transformed, n_train, attn_batch_size, head=DEFAULT_HEAD):
    attn_chunks = []

    for start in range(0, x_test_transformed.shape[0], attn_batch_size):
        stop = min(start + attn_batch_size, x_test_transformed.shape[0])
        chunk = x_test_transformed[start:stop]

        _, attn = classifier.predict(chunk, return_attn=True)

        head_attn = to_numpy_array(attn[0, head, n_train:, :])
        attn_chunks.append(head_attn)

    if len(attn_chunks) == 0:
        attn_maps = np.empty((0, n_train))
    else:
        attn_maps = np.concatenate(attn_chunks, axis=0)

    return attn_maps


def extract_test_embeddings(classifier, x_test_transformed, n_train, batch_size):
    test_embeddings_chunks = []

    for start in range(0, x_test_transformed.shape[0], batch_size):
        stop = min(start + batch_size, x_test_transformed.shape[0])
        chunk = x_test_transformed[start:stop]

        _, _, row_embeddings = classifier.predict(
            chunk,
            return_attn=True,
            return_row_representations=True,
        )

        row_embeddings_np = to_numpy_array(row_embeddings)
        if row_embeddings_np.ndim == 3:
            row_embeddings_np = row_embeddings_np[0]

        # Only keep rows corresponding to the held-out chunk.
        test_embeddings_chunks.append(row_embeddings_np[n_train:])

    if len(test_embeddings_chunks) == 0:
        return np.empty((0, 0), dtype=np.float32)

    return np.concatenate(test_embeddings_chunks, axis=0)


def fit_full_model_and_extract_attention(
    x_train,
    y_train,
    x_test,
    n_estimators=1,
    random_state=0,
    attn_batch_size=128,
    head=DEFAULT_HEAD,
    extract_train_embeddings=True,
    extract_test_embeddings_flag=False,
):
    vectorizer = TableVectorizer()
    x_train_transformed = vectorizer.fit_transform(x_train)
    x_test_transformed = vectorizer.transform(x_test)

    classifier = TabICLClassifier(n_estimators=n_estimators, random_state=random_state)
    classifier.fit(x_train_transformed, y_train)

    n_train = x_train_transformed.shape[0]
    attn_maps = extract_attention_data(
        classifier=classifier,
        x_test_transformed=x_test_transformed,
        n_train=n_train,
        attn_batch_size=attn_batch_size,
        head=head,
    )

    train_row_embeddings = None
    if extract_train_embeddings:
        train_row_embeddings = extract_embeddings(
            x_data=x_train.reset_index(drop=True),
            y_data=y_train.reset_index(drop=True),
            n_estimators=n_estimators,
            random_state=random_state,
            batch_size=attn_batch_size,
            n_folds=10,
        )

    test_row_embeddings = None
    if extract_test_embeddings_flag:
        test_row_embeddings = extract_test_embeddings(
            classifier=classifier,
            x_test_transformed=x_test_transformed,
            n_train=n_train,
            batch_size=attn_batch_size,
        )

    return classifier, x_test_transformed, attn_maps, train_row_embeddings, test_row_embeddings


def train_and_score_model_on_subset(
    x_train,
    y_train,
    x_test,
    y_test,
    subset_indices=None,
    n_estimators=1,
    random_state=0,
    pos_label="yes",
):
    if subset_indices is None:
        x_subset = x_train.reset_index(drop=True)
        y_subset = y_train.reset_index(drop=True)
    else:
        x_subset = x_train.iloc[subset_indices].reset_index(drop=True)
        y_subset = y_train.iloc[subset_indices].reset_index(drop=True)

    vectorizer = TableVectorizer()
    x_subset_transformed = vectorizer.fit_transform(x_subset)
    x_test_transformed = vectorizer.transform(x_test)

    classifier = TabICLClassifier(n_estimators=n_estimators, random_state=random_state)
    classifier.fit(x_subset_transformed, y_subset)

    y_probs = classifier.predict_proba(x_test_transformed)
    y_pred_idx = np.argmax(y_probs, axis=1)
    y_pred = classifier.y_encoder_.inverse_transform(y_pred_idx)

    y_probs = adjust_probs_for_single_class(y_probs, y_subset, pos_label)
    accuracy = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_probs[:, 1]))
    f1 = float(f1_score(y_test, y_pred, zero_division=0, pos_label=pos_label, average="binary"))

    return accuracy, f1, roc_auc
