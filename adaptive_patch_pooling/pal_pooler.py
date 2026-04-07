"""PALPooler — Pseudo-Attention Label Pooler.

Encapsulates a single stage of Ridge-based adaptive patch pooling.
Chain multiple instances via :class:`IterativePALPooler`, or manually by
passing ``_support_projected_`` and ``_pca_`` from one stage into
``initial_support`` / ``initial_pca`` on the next.

Typical single-stage usage
--------------------------
>>> from tabicl import TabICLClassifier
>>> from adaptive_patch_pooling import PALPooler
>>>
>>> tabicl = TabICLClassifier(n_estimators=1, random_state=42)
>>> pooler = PALPooler(tabicl, patch_group_size=4, temperature=2.0, ridge_alpha=10.0)
>>> pooler.fit(train_patches, train_labels)          # [N, P, D], [N]
>>> X_train = pooler.transform(train_patches)        # [N, D]  raw DINO-space
>>> X_test  = pooler.transform(test_patches)         # [N_test, D]

Two-stage chained refinement (manual)
--------------------------------------
>>> stage0 = PALPooler(tabicl, patch_group_size=4, temperature=2.0, ridge_alpha=10.0)
>>> stage0.fit(train_patches, train_labels)
>>>
>>> stage1 = PALPooler(tabicl, patch_group_size=1, temperature=1.0, ridge_alpha=1.0)
>>> stage1.fit(train_patches, train_labels,
...            initial_support=stage0._support_projected_,
...            initial_pca=stage0._pca_)
>>>
>>> X_train = stage1.transform(train_patches)
>>> X_test  = stage1.transform(test_patches)

Two-stage chained refinement (IterativePALPooler)
--------------------------------------------------
>>> from adaptive_patch_pooling import IterativePALPooler
>>>
>>> pooler = IterativePALPooler(
...     tabicl,
...     patch_group_sizes=[4, 1],
...     temperature=[2.0, 1.0],
...     ridge_alpha=[10.0, 1.0],
... )
>>> pooler.fit(train_patches, train_labels)
>>> X_train = pooler.transform(train_patches)
>>> X_test  = pooler.transform(test_patches)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from tabicl import TabICLClassifier

from adaptive_patch_pooling.patch_pooling import (
    _ridge_pool_weights,
    group_patches,
    refine_dataset_features,
)


class PALPooler:
    """Pseudo-Attention Label Pooler.

    Fits a Ridge regression model that predicts per-patch quality logits from
    raw DINO patch embeddings.  The Ridge model is trained to replicate quality
    signals produced by a TabICL classifier evaluated on a support set.  At
    transform time the Ridge logits drive a softmax-weighted pooling that
    collapses ``[N, P, D] → [N, D]``.

    A single ``PALPooler`` represents one stage of refinement at a fixed
    ``patch_group_size``.  Multi-stage refinement is achieved by chaining
    instances via :class:`IterativePALPooler`.

    The pooler intentionally outputs raw DINO-space features (``[N, D]``)
    so the caller can apply any dimensionality reduction they choose.
    PCA is used *internally* during refinement (to build a compact support
    for TabICL scoring) but the fitted projection is not part of the public
    API.

    Parameters
    ----------
    tabicl : TabICLClassifier
        Pre-initialized (unfitted) TabICL model whose ``n_estimators`` and
        ``random_state`` are forwarded to the internal quality scorer.
    patch_group_size : int
        Spatial grouping factor before pooling.  Must be a perfect square
        (1, 4, 9, …).  ``1`` = individual patches (no grouping).
    weight_method : str
        Quality scoring method.  One of:
        ``"correct_class_prob"`` (log true-class probability),
        ``"entropy"`` (normalised Shannon entropy, label-agnostic),
        ``"kl_div"`` (KL divergence from empirical class prior, label-agnostic).
    temperature : float
        Softmax temperature applied to quality logits.
        Large → uniform / mean pooling.  Small → peaked on best patch.
    ridge_alpha : float
        Ridge regularisation strength (same semantics as ``sklearn.linear_model.Ridge``).
    max_query_rows : int or None
        Maximum number of ``(image × patch)`` rows forwarded to TabICL in a
        single call.  ``None`` = no limit (all active rows in batches of
        ``batch_size`` images).
    use_random_subsampling : bool
        When active rows exceed ``max_query_rows``, draw a random subset of
        that many rows instead of processing all rows in batches.
    pca_dim : int or None
        Number of PCA components used internally for the support.
        ``None`` disables internal PCA (full D-dimensional DINO features are
        used for TabICL scoring).
    batch_size : int
        Images per batch in the batched TabICL forward pass (only used when
        ``max_query_rows`` is ``None`` or the row count is below the cap).
    seed : int
        Random seed for PCA, subsampling, and TabICL initialisation.
    aoe_class : int or None
        Class index of the absence-of-evidence class.  Triggers special
        handling during Ridge fitting (controlled by ``aoe_handling``).
    aoe_handling : str
        How to handle AoE patches during Ridge fitting.
        ``"filter"`` excludes them; ``"entropy"`` includes them but scores
        them with the label-agnostic entropy method.
    gpu_ridge : bool
        Solve the Ridge normal equations on the GPU (requires PyTorch + CUDA).
    gpu_ridge_device : str
        Torch device string for ``RidgeGPU`` (e.g. ``"cuda"``, ``"cuda:1"``).
    normalize_features : bool
        Fit a ``StandardScaler`` on training patches before Ridge fitting.
    Fitted attributes (available after ``fit``)
    -------------------------------------------
    ridge_model_ : Ridge or RidgeGPU
        Fitted quality predictor.  The core learned component of the pooler.
    feature_scaler_ : StandardScaler or None
        Fitted scaler applied before Ridge prediction (``None`` unless
        ``normalize_features=True``).
    support_ : np.ndarray, shape [N, D]
        Refined support set embeddings in the original DINO feature space
        (raw, no PCA applied).  Apply your own dimensionality reduction
        before passing to a downstream classifier.
    support_labels_ : np.ndarray, shape [N]
        Labels corresponding to ``support_``.
    scoring_clf_ : TabICLClassifier
        Quality scorer fitted on the *input* support (used to generate Ridge
        training targets).  Retained for post-hoc visualisation.
    n_patches_grouped_ : int
        ``P'`` — number of patch groups after spatial grouping.
    embed_dim_ : int
        ``D`` — original patch embedding dimension.
    fit_time_s_ : float
        Seconds for the TabICL forward pass and Ridge fitting phase.
    pool_time_s_ : float
        Seconds for Ridge prediction over all images, repooling, and PCA refit.
    """

    def __init__(
        self,
        tabicl: TabICLClassifier,
        patch_group_size: int = 1,
        weight_method: str = "correct_class_prob",
        temperature: float = 1.0,
        ridge_alpha: float = 1.0,
        max_query_rows: Optional[int] = None,
        use_random_subsampling: bool = False,
        pca_dim: Optional[int] = 128,
        batch_size: int = 1000,
        seed: int = 42,
        aoe_class: Optional[int] = None,
        aoe_handling: str = "filter",
        gpu_ridge: bool = False,
        gpu_ridge_device: str = "cuda",
        normalize_features: bool = False,
    ) -> None:
        self.tabicl = tabicl
        self.patch_group_size = patch_group_size
        self.weight_method = weight_method
        self.temperature = temperature
        self.ridge_alpha = ridge_alpha
        self.max_query_rows = max_query_rows
        self.use_random_subsampling = use_random_subsampling
        self.pca_dim = pca_dim
        self.batch_size = batch_size
        self.seed = seed
        self.aoe_class = aoe_class
        self.aoe_handling = aoe_handling
        self.gpu_ridge = gpu_ridge
        self.gpu_ridge_device = gpu_ridge_device
        self.normalize_features = normalize_features

    # ------------------------------------------------------------------
    # Core sklearn-style API
    # ------------------------------------------------------------------

    def fit(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        initial_support: Optional[np.ndarray] = None,
        initial_pca: Optional[PCA] = None,
    ) -> "PALPooler":
        """Fit the Ridge quality model.

        Constructs a mean-pool support set from *patches* (or uses the
        provided *initial_support*), feeds it to TabICL to generate per-patch
        quality logit targets, then fits a Ridge model to predict those logits
        from raw patch features.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D]
            Raw patch embeddings for the training set.
        labels : np.ndarray, shape [N]
            Integer class labels.
        initial_support : np.ndarray or None
            Pre-built support (in the PCA-projected space of the previous
            stage) to use instead of the mean-pool baseline.  Pass
            ``prev_stage._support_projected_`` when chaining stages manually.
        initial_pca : PCA or None
            PCA corresponding to *initial_support*.  Pass
            ``prev_stage._pca_`` when chaining stages manually.

        Returns
        -------
        self
        """
        patches = np.asarray(patches, dtype=np.float32)
        labels = np.asarray(labels)
        N, P, D = patches.shape
        self.embed_dim_ = D

        grouped = group_patches(patches, self.patch_group_size)  # [N, P', D]
        self.n_patches_grouped_ = grouped.shape[1]

        if initial_support is not None:
            support = initial_support
            current_pca = initial_pca
        else:
            support_raw = patches.mean(axis=1)  # [N, D]
            if self.pca_dim is not None:
                n_comp = min(self.pca_dim, N, D)
                current_pca = PCA(n_components=n_comp, random_state=self.seed)
                support = current_pca.fit_transform(support_raw).astype(np.float32)
            else:
                current_pca = None
                support = support_raw

        aoe_mask: Optional[np.ndarray] = None
        if self.aoe_class is not None:
            aoe_mask = (labels == self.aoe_class)

        n_estimators = getattr(self.tabicl, "n_estimators", 1)

        (refined_support, new_pca, weights_ridge, ridge_model, feature_scaler, scoring_clf,
         fit_time_s, pool_time_s) = refine_dataset_features(
            train_patches=grouped,
            train_labels=labels,
            support_features=support,
            pca=current_pca,
            n_estimators=n_estimators,
            temperature=self.temperature,
            seed=self.seed,
            batch_size=self.batch_size,
            weight_method=self.weight_method,
            ridge_alpha=self.ridge_alpha,
            normalize_features=self.normalize_features,
            max_query_rows=self.max_query_rows,
            use_random_subsampling=self.use_random_subsampling,
            aoe_mask=aoe_mask,
            aoe_handling=self.aoe_handling,
            use_gpu_ridge=self.gpu_ridge,
            gpu_ridge_device=self.gpu_ridge_device,
        )

        # Derive raw (pre-PCA) support in the original D-dimensional DINO space.
        # weights_ridge [N, P'] are the Ridge softmax weights already computed
        # inside refine_dataset_features; reuse them to avoid a second prediction.
        repooled_raw = (weights_ridge[:, :, None] * grouped).sum(axis=1).astype(np.float32)  # [N, D]
        raw_support = repooled_raw

        self.ridge_model_ = ridge_model
        self.feature_scaler_ = feature_scaler
        # Private: internal PCA and projected support for stage chaining and score_tabicl.
        self._pca_ = new_pca
        self._support_projected_ = refined_support
        # Public: raw DINO-space support (caller applies own dim reduction).
        self.support_ = raw_support
        self.support_labels_ = labels
        self.scoring_clf_ = scoring_clf
        self.fit_time_s_ = fit_time_s
        self.pool_time_s_ = pool_time_s

        return self

    def transform(
        self,
        patches: np.ndarray,
    ) -> np.ndarray:
        """Pool patches using the fitted Ridge model.

        Groups patches spatially at ``patch_group_size``, computes softmax
        pooling weights from the Ridge model, and returns the weighted sum
        in the original DINO feature space.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D]

        Returns
        -------
        np.ndarray, shape [N, D]
            Quality-weighted pooled embeddings in the original DINO feature
            space.  Apply your own dimensionality reduction before passing
            to a downstream classifier.
        """
        self._check_fitted()
        patches = np.asarray(patches, dtype=np.float32)
        grouped = group_patches(patches, self.patch_group_size)      # [N, P', D]
        weights = _ridge_pool_weights(grouped, self.ridge_model_, self.feature_scaler_)  # [N, P']
        return (weights[:, :, None] * grouped).sum(axis=1)           # [N, D]

    def fit_transform(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        initial_support: Optional[np.ndarray] = None,
        initial_pca: Optional[PCA] = None,
    ) -> np.ndarray:
        """Fit and transform in one call.

        Equivalent to ``fit(...).transform(patches)``.  Returns ``[N, D]``
        raw pooled embeddings in the original DINO feature space.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D]
        labels : np.ndarray, shape [N]
        initial_support, initial_pca : optional, see ``fit``

        Returns
        -------
        np.ndarray, shape [N, D]
        """
        return self.fit(patches, labels, initial_support, initial_pca).transform(patches)

    # ------------------------------------------------------------------
    # Visualisation support
    # ------------------------------------------------------------------

    def patch_weights(
        self,
        patches: np.ndarray,
    ) -> np.ndarray:
        """Per-patch Ridge softmax weights.

        Useful for heatmap overlays: the returned array directly describes
        how much each (grouped) patch contributes to the pooled embedding.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D] or [P, D]

        Returns
        -------
        np.ndarray, shape [N, P'] or [P']
            Softmax weights summing to 1 per image.  ``P'`` is the number of
            patch groups (``P / patch_group_size``).
        """
        self._check_fitted()
        single = patches.ndim == 2
        if single:
            patches = patches[None]
        patches = np.asarray(patches, dtype=np.float32)
        grouped = group_patches(patches, self.patch_group_size)
        weights = _ridge_pool_weights(grouped, self.ridge_model_, self.feature_scaler_)
        return weights[0] if single else weights

    def patch_quality_logits(
        self,
        patches: np.ndarray,
    ) -> np.ndarray:
        """Raw Ridge quality predictions before softmax normalisation.

        These are the values the Ridge model predicts from raw patch features.
        They have the same semantics as the targets in ``compute_patch_quality_logits``
        (negative log-probability scores, in ``(-∞, 0]`` for ``correct_class_prob``
        and ``entropy``).  Useful for debugging and understanding the Ridge fit.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D] or [P, D]

        Returns
        -------
        np.ndarray, shape [N, P'] or [P']
            Raw Ridge logits before softmax.
        """
        self._check_fitted()
        single = patches.ndim == 2
        if single:
            patches = patches[None]
        patches = np.asarray(patches, dtype=np.float32)
        N, P, D = patches.shape
        grouped = group_patches(patches, self.patch_group_size)   # [N, P', D]
        _, P_prime, _ = grouped.shape
        flat = grouped.reshape(N * P_prime, D)
        if self.feature_scaler_ is not None:
            flat = self.feature_scaler_.transform(flat)
        logits = self.ridge_model_.predict(flat).reshape(N, P_prime).astype(np.float32)
        return logits[0] if single else logits

    def score_tabicl(
        self,
        query_patches: np.ndarray,
        query_labels: np.ndarray,
        n_estimators: Optional[int] = None,
    ) -> tuple[float, float]:
        """Evaluate test-set accuracy and AUROC using the fitted support.

        Pools *query_patches* with the fitted Ridge model, projects into the
        internal support space (via ``_pca_``), fits a fresh
        ``TabICLClassifier`` on ``_support_projected_``, and reports accuracy
        + AUROC on the query set.

        This mirrors the ``_compute_accuracy_from_features`` evaluation step
        in ``local_embedding_patch_quality.py``.

        Parameters
        ----------
        query_patches : np.ndarray, shape [N_test, P, D]
        query_labels : np.ndarray, shape [N_test]
        n_estimators : int or None
            TabICL ensemble size.  Defaults to ``self.tabicl.n_estimators``.

        Returns
        -------
        (accuracy, auroc) : tuple[float, float]
            AUROC is ``float("nan")`` when it cannot be computed (e.g. single
            class in the query set).
        """
        from sklearn.metrics import roc_auc_score

        self._check_fitted()
        n_est = n_estimators or getattr(self.tabicl, "n_estimators", 1)
        query_raw = self.transform(query_patches)  # [N, D]
        if self._pca_ is not None:
            query_features = self._pca_.transform(query_raw).astype(np.float32)
        else:
            query_features = query_raw

        clf = TabICLClassifier(n_estimators=n_est, random_state=self.seed)
        clf.fit(self._support_projected_, self.support_labels_)
        proba = clf.predict_proba(query_features)
        acc = float((np.argmax(proba, axis=1) == query_labels).mean())
        try:
            if proba.shape[1] == 2:
                auroc = float(roc_auc_score(query_labels, proba[:, 1]))
            else:
                auroc = float(roc_auc_score(query_labels, proba, multi_class="ovr", average="macro"))
        except ValueError:
            auroc = float("nan")
        return acc, auroc

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the entire fitted pooler to a ``joblib`` file.

        The saved file includes the Ridge model, scaler, and all
        hyperparameters.  Load with ``PALPooler.load(path)``.
        """
        import joblib
        self._check_fitted()
        joblib.dump(self, Path(path))
        print(f"[PALPooler] Saved → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PALPooler":
        """Load a previously saved ``PALPooler`` from a ``joblib`` file."""
        import joblib
        pooler = joblib.load(Path(path))
        if not isinstance(pooler, cls):
            raise TypeError(f"Loaded object is {type(pooler).__name__}, expected PALPooler")
        return pooler

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "ridge_model_"):
            raise RuntimeError("PALPooler is not fitted yet.  Call fit() first.")

    def __repr__(self) -> str:
        fitted = hasattr(self, "ridge_model_")
        status = (
            f"fitted: D={self.embed_dim_}  P'={self.n_patches_grouped_}  "
            f"fit={self.fit_time_s_:.1f}s  pool={self.pool_time_s_:.1f}s"
            if fitted else "not fitted"
        )
        return (
            f"PALPooler("
            f"patch_group_size={self.patch_group_size}, "
            f"weight_method='{self.weight_method}', "
            f"temperature={self.temperature}, "
            f"ridge_alpha={self.ridge_alpha}, "
            f"pca_dim={self.pca_dim}, "
            f"[{status}])"
        )


class IterativePALPooler:
    """Multi-stage PAL pooler that chains refinement stages automatically.

    Each stage fits a :class:`PALPooler`, refines the support, and passes the
    refined projected support to the next stage as its starting point — exactly
    replicating the iterative loop in ``local_embedding_patch_quality.py``.
    After fitting, all transform / scoring calls are delegated to the final
    stage.

    Parameters
    ----------
    tabicl : TabICLClassifier
        Pre-initialized (unfitted) TabICL model shared across all stages.
    patch_group_sizes : list of int
        Ordered list of spatial grouping factors, one per stage.  Each must be
        a perfect square (1, 4, 9, …).  Example: ``[4, 1]`` runs a coarse
        group-of-4 stage followed by an individual-patch stage.
    weight_method : str
        Quality scoring method passed to every stage (see :class:`PALPooler`).
    temperature : float or list of float
        Softmax temperature.  A single value is broadcast to all stages; a list
        must have one entry per stage.
    ridge_alpha : float or list of float
        Ridge regularisation strength.  A single value is broadcast to all
        stages; a list must have one entry per stage.
    max_query_rows : int or None
        Forwarded to every stage unchanged.
    use_random_subsampling : bool
        Forwarded to every stage unchanged.
    pca_dim : int or None
        Internal PCA dimension used at each stage for building the compact
        support fed to TabICL.  ``None`` disables internal PCA.
    batch_size : int
        Forwarded to every stage unchanged.
    seed : int
        Forwarded to every stage unchanged.
    aoe_class : int or None
        Forwarded to every stage unchanged.
    aoe_handling : str
        Forwarded to every stage unchanged.
    gpu_ridge : bool
        Forwarded to every stage unchanged.
    gpu_ridge_device : str
        Forwarded to every stage unchanged.
    normalize_features : bool
        Forwarded to every stage unchanged.
    Fitted attributes (available after ``fit``)
    -------------------------------------------
    stages_ : list of PALPooler
        All fitted stage poolers, in order.
    support_ : np.ndarray, shape [N, D]
        Raw DINO-space support from the final stage (``stages_[-1].support_``).
    support_labels_ : np.ndarray, shape [N]
        Labels corresponding to ``support_``.
    """

    def __init__(
        self,
        tabicl: TabICLClassifier,
        patch_group_sizes: List[int],
        weight_method: str = "correct_class_prob",
        temperature: Union[float, List[float]] = 1.0,
        ridge_alpha: Union[float, List[float]] = 1.0,
        max_query_rows: Optional[int] = None,
        use_random_subsampling: bool = False,
        pca_dim: Optional[int] = 128,
        batch_size: int = 1000,
        seed: int = 42,
        aoe_class: Optional[int] = None,
        aoe_handling: str = "filter",
        gpu_ridge: bool = False,
        gpu_ridge_device: str = "cuda",
        normalize_features: bool = False,
    ) -> None:
        self.tabicl = tabicl
        self.patch_group_sizes = list(patch_group_sizes)
        self.weight_method = weight_method
        self.temperature = temperature
        self.ridge_alpha = ridge_alpha
        self.max_query_rows = max_query_rows
        self.use_random_subsampling = use_random_subsampling
        self.pca_dim = pca_dim
        self.batch_size = batch_size
        self.seed = seed
        self.aoe_class = aoe_class
        self.aoe_handling = aoe_handling
        self.gpu_ridge = gpu_ridge
        self.gpu_ridge_device = gpu_ridge_device
        self.normalize_features = normalize_features

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
    ) -> "IterativePALPooler":
        """Fit all stages sequentially, passing the refined support forward.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D]
            Raw patch embeddings for the training set.
        labels : np.ndarray, shape [N]
            Integer class labels.

        Returns
        -------
        self
        """
        n_stages = len(self.patch_group_sizes)
        if n_stages == 0:
            raise ValueError("patch_group_sizes must contain at least one entry.")

        temps = self._expand_param(self.temperature, n_stages, "temperature")
        alphas = self._expand_param(self.ridge_alpha, n_stages, "ridge_alpha")

        stages: List[PALPooler] = []
        initial_support: Optional[np.ndarray] = None
        initial_pca: Optional[PCA] = None

        for k, group_size in enumerate(self.patch_group_sizes):
            print(f"[IterativePALPooler] Stage {k}/{n_stages - 1} "
                  f"— patch_group_size={group_size}, "
                  f"temperature={temps[k]}, ridge_alpha={alphas[k]}")
            stage = PALPooler(
                tabicl=self.tabicl,
                patch_group_size=group_size,
                weight_method=self.weight_method,
                temperature=temps[k],
                ridge_alpha=alphas[k],
                max_query_rows=self.max_query_rows,
                use_random_subsampling=self.use_random_subsampling,
                pca_dim=self.pca_dim,
                batch_size=self.batch_size,
                seed=self.seed,
                aoe_class=self.aoe_class,
                aoe_handling=self.aoe_handling,
                gpu_ridge=self.gpu_ridge,
                gpu_ridge_device=self.gpu_ridge_device,
                normalize_features=self.normalize_features,
            )
            stage.fit(patches, labels, initial_support=initial_support, initial_pca=initial_pca)
            stages.append(stage)

            # Hand the internal projected support to the next stage.
            initial_support = stage._support_projected_
            initial_pca = stage._pca_

        self.stages_ = stages
        return self

    def transform(self, patches: np.ndarray) -> np.ndarray:
        """Pool patches using the final fitted stage.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D]

        Returns
        -------
        np.ndarray, shape [N, D]
            Quality-weighted pooled embeddings in the original DINO feature space.
        """
        self._check_fitted()
        return self.stages_[-1].transform(patches)

    def fit_transform(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Fit all stages then transform *patches* with the final stage.

        Returns
        -------
        np.ndarray, shape [N, D]
        """
        return self.fit(patches, labels).transform(patches)

    # ------------------------------------------------------------------
    # Convenience delegations to final stage
    # ------------------------------------------------------------------

    @property
    def support_(self) -> np.ndarray:
        """Raw DINO-space support from the final stage."""
        self._check_fitted()
        return self.stages_[-1].support_

    @property
    def support_labels_(self) -> np.ndarray:
        self._check_fitted()
        return self.stages_[-1].support_labels_

    def patch_weights(self, patches: np.ndarray) -> np.ndarray:
        """Per-patch Ridge softmax weights from the final stage."""
        self._check_fitted()
        return self.stages_[-1].patch_weights(patches)

    def patch_quality_logits(self, patches: np.ndarray) -> np.ndarray:
        """Raw Ridge quality logits from the final stage."""
        self._check_fitted()
        return self.stages_[-1].patch_quality_logits(patches)

    def score_tabicl(
        self,
        query_patches: np.ndarray,
        query_labels: np.ndarray,
        n_estimators: Optional[int] = None,
    ) -> tuple[float, float]:
        """Evaluate accuracy and AUROC using the final stage's support.

        See :meth:`PALPooler.score_tabicl` for full documentation.
        """
        self._check_fitted()
        return self.stages_[-1].score_tabicl(query_patches, query_labels, n_estimators)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the entire fitted iterative pooler to a ``joblib`` file."""
        import joblib
        self._check_fitted()
        joblib.dump(self, Path(path))
        print(f"[IterativePALPooler] Saved → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IterativePALPooler":
        """Load a previously saved ``IterativePALPooler`` from a ``joblib`` file."""
        import joblib
        pooler = joblib.load(Path(path))
        if not isinstance(pooler, cls):
            raise TypeError(
                f"Loaded object is {type(pooler).__name__}, expected IterativePALPooler"
            )
        return pooler

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_param(param, n_stages: int, name: str) -> list:
        if isinstance(param, list):
            if len(param) != n_stages:
                raise ValueError(
                    f"{name} list has {len(param)} entries but patch_group_sizes "
                    f"has {n_stages} stages."
                )
            return param
        return [param] * n_stages

    def _check_fitted(self) -> None:
        if not hasattr(self, "stages_"):
            raise RuntimeError(
                "IterativePALPooler is not fitted yet.  Call fit() first."
            )

    def __repr__(self) -> str:
        fitted = hasattr(self, "stages_")
        if fitted:
            stage_strs = [
                f"g{s.patch_group_size}(T={s.temperature}, α={s.ridge_alpha})"
                for s in self.stages_
            ]
        else:
            temps = self._expand_param(self.temperature, len(self.patch_group_sizes), "temperature")
            alphas = self._expand_param(self.ridge_alpha, len(self.patch_group_sizes), "ridge_alpha")
            stage_strs = [
                f"g{g}(T={t}, α={a})"
                for g, t, a in zip(self.patch_group_sizes, temps, alphas)
            ]
        return (
            f"IterativePALPooler("
            f"{'fitted' if fitted else 'not fitted'}: "
            f"[{', '.join(stage_strs)}])"
        )
