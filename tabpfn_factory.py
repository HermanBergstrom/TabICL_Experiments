"""Factory for constructing TabPFN models with modified preprocessing configurations.

This module lets you easily ablate or replace individual preprocessing steps in
TabPFN without touching the model weights. Use the high-level `build_tabpfn_*`
functions for common cases, or construct a `PreprocessorConfig` list manually for
full control.

Compatible with tabpfn >= 7.1 (v2.6 model weights).

Quick example
-------------
    from tabpfn_factory import build_tabpfn_classifier, PreprocessingPreset

    # Default TabPFN (uses checkpoint config unchanged)
    clf = build_tabpfn_classifier()

    # No feature-distribution reshaping (identity transform), no SVD
    clf = build_tabpfn_classifier(preset=PreprocessingPreset.NO_RESHAPE)

    # No SVD, all features passed through (no subsampling)
    clf = build_tabpfn_classifier(
        preset=PreprocessingPreset.NO_SVD,
        max_features_per_estimator=10_000_000,
        ignore_pretraining_limits=True,
    )

    # Fully custom: one ensemble member with quantile normalisation, no SVD
    from tabpfn.preprocessing import PreprocessorConfig
    clf = build_tabpfn_classifier(
        preprocess_transforms=[
            PreprocessorConfig(
                name="quantile_norm_coarse",
                categorical_name="ordinal_very_common_categories_shuffled",
                global_transformer_name=None,
            )
        ]
    )

Model versions
--------------
    The `model_version` parameter selects which default preprocessor configs to
    derive presets from. It should match the model weights you are using:
        "v2"   – original TabPFN v2 weights
        "v2.5" – TabPFN v2.5 weights
        "v2.6" – TabPFN v2.6 weights (default; used when model_path="auto")

    For the DEFAULT preset, no preprocessing override is injected — the checkpoint
    supplies its own config, which is the correct behaviour for v2.5 and v2.6.

Inspecting preprocessed features
----------------------------------
    After fitting, `clf.executor_.ensemble_members` is a list of
    `TabPFNEnsembleMember` objects. Each has:
        member.X_train          – preprocessed training array (numpy)
        member.X_train.shape    – (n_samples, n_preprocessed_features)
        member.cpu_preprocessor.transform(X_test).X  – transform test data

Configurable fields in PreprocessorConfig
------------------------------------------
    name : str
        Feature-distribution transform. Key options:
            "none"                        – identity (no transform)
            "squashing_scaler_default"    – squashing scaler (default clf v2.5)
            "safepower"                   – Yeo-Johnson power transform
            "quantile_uni_coarse"         – uniform quantile
            "quantile_uni"                – uniform quantile (default clf v2.6)
            "quantile_norm_coarse"        – normal quantile
            "robust"                      – robust scaler
            "kdi" and many kdi_* variants
    categorical_name : str
        Categorical encoding. Options:
            "none"                                 – no encoding
            "numeric"                              – treat as numbers
            "ordinal"                              – ordinal encode
            "ordinal_shuffled"                     – ordinal, random order
            "ordinal_very_common_categories_shuffled"  – ordinal, rare → NaN
            "onehot"                               – one-hot encode
    append_original : bool | "auto"
        If True, append transformed features to originals instead of replacing.
        "auto" enables this when n_features < 500.
    max_features_per_estimator : int
        Randomly subsample features to at most this many per estimator.
        NOTE: this is independent of ignore_pretraining_limits. Set to a large
        value (e.g. 10_000_000) to disable subsampling entirely.
    global_transformer_name : str | None
        Global transform applied after column-wise transform. Options:
            None                      – no global transform
            "svd_quarter_components"  – SVD with n_features/4 components
            "svd"                     – SVD with min(n//10+1, n//2) components
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing import (
    PreprocessorConfig,
    v2_classifier_preprocessor_configs,
    v2_regressor_preprocessor_configs,
    v2_5_classifier_preprocessor_configs,
    v2_5_regressor_preprocessor_configs,
    v2_6_classifier_preprocessor_configs,
    v2_6_regressor_preprocessor_configs,
)

ModelVersionStr = Literal["v2", "v2.5", "v2.6"]

_CLF_DEFAULTS: dict[str, object] = {
    "v2": v2_classifier_preprocessor_configs,
    "v2.5": v2_5_classifier_preprocessor_configs,
    "v2.6": v2_6_classifier_preprocessor_configs,
}

_REG_DEFAULTS: dict[str, object] = {
    "v2": v2_regressor_preprocessor_configs,
    "v2.5": v2_5_regressor_preprocessor_configs,
    "v2.6": v2_6_regressor_preprocessor_configs,
}


class PreprocessingPreset(Enum):
    """Named presets for common ablation scenarios."""

    DEFAULT = "default"
    """The default TabPFN preprocessing (unmodified checkpoint config)."""

    NO_RESHAPE = "no_reshape"
    """Identity transform — no feature-distribution reshaping at all, no SVD.
    Categorical features still get ordinal-encoded (first member) or treated as
    numeric (second member), mirroring the default ensemble structure.
    """

    NO_SVD = "no_svd"
    """Default transforms, but with the SVD global step removed."""

    NO_CATEGORICAL_ENCODING = "no_categorical_encoding"
    """Default transforms, but categorical features are passed through as-is
    (treated as numeric). No ordinal encoding or shuffling.
    """

    IDENTITY_ONLY = "identity_only"
    """Single ensemble member: pure identity transform and numeric encoding.
    Closest to feeding raw data directly to the model.
    """


def _defaults_for(task: Literal["clf", "reg"], version: ModelVersionStr) -> list[PreprocessorConfig]:
    factory = (_CLF_DEFAULTS if task == "clf" else _REG_DEFAULTS)[version]
    return factory()  # type: ignore[operator]


def _apply_preset(
    task: Literal["clf", "reg"],
    preset: PreprocessingPreset,
    version: ModelVersionStr,
    max_features_per_estimator: int | None,
) -> list[PreprocessorConfig] | None:
    """Return the transforms for the given preset, or None for DEFAULT."""
    if preset == PreprocessingPreset.DEFAULT:
        return None

    defaults = _defaults_for(task, version)

    if preset == PreprocessingPreset.NO_RESHAPE:
        mf = max_features_per_estimator if max_features_per_estimator is not None else 500
        if task == "clf":
            return [
                PreprocessorConfig(
                    name="none",
                    categorical_name="ordinal_very_common_categories_shuffled",
                    global_transformer_name=None,
                    max_features_per_estimator=mf,
                ),
                PreprocessorConfig(
                    name="none",
                    categorical_name="numeric",
                    global_transformer_name=None,
                    max_features_per_estimator=mf,
                ),
            ]
        else:
            return [
                PreprocessorConfig(
                    name="none",
                    categorical_name=c.categorical_name,
                    global_transformer_name=None,
                    max_features_per_estimator=(
                        max_features_per_estimator
                        if max_features_per_estimator is not None
                        else c.max_features_per_estimator
                    ),
                )
                for c in defaults
            ]

    if preset == PreprocessingPreset.NO_SVD:
        return [
            PreprocessorConfig(
                name=c.name,
                categorical_name=c.categorical_name,
                append_original=c.append_original,
                max_features_per_estimator=(
                    max_features_per_estimator
                    if max_features_per_estimator is not None
                    else c.max_features_per_estimator
                ),
                global_transformer_name=None,
            )
            for c in defaults
        ]

    if preset == PreprocessingPreset.NO_CATEGORICAL_ENCODING:
        return [
            PreprocessorConfig(
                name=c.name,
                categorical_name="numeric",
                append_original=c.append_original,
                max_features_per_estimator=(
                    max_features_per_estimator
                    if max_features_per_estimator is not None
                    else c.max_features_per_estimator
                ),
                global_transformer_name=c.global_transformer_name,
            )
            for c in defaults
        ]

    if preset == PreprocessingPreset.IDENTITY_ONLY:
        return [
            PreprocessorConfig(
                name="none",
                categorical_name="numeric",
                global_transformer_name=None,
                max_features_per_estimator=(
                    max_features_per_estimator if max_features_per_estimator is not None else 500
                ),
            )
        ]

    raise ValueError(f"Unknown preset: {preset}")


def build_tabpfn_classifier(
    *,
    preset: PreprocessingPreset = PreprocessingPreset.DEFAULT,
    preprocess_transforms: list[PreprocessorConfig] | None = None,
    max_features_per_estimator: int | None = None,
    model_version: ModelVersionStr = "v2.6",
    n_estimators: int = 8,
    model_path: str = "auto",
    device: str = "auto",
    random_state: int | None = 0,
    ignore_pretraining_limits: bool = False,
) -> TabPFNClassifier:
    """Build a TabPFNClassifier with configurable preprocessing.

    Parameters
    ----------
    preset:
        A named preset from `PreprocessingPreset`. Ignored when
        `preprocess_transforms` is provided.
    preprocess_transforms:
        Explicit list of `PreprocessorConfig` objects. Overrides `preset` when
        given. Each config becomes one group of ensemble members.
    max_features_per_estimator:
        Cap on the number of features passed to each ensemble member. Features
        are randomly subsampled to this limit *before* any transformation.
        Default (None) keeps whatever each preset specifies (typically 500 for
        v2.6, 1_000_000 for v2). Set to a large value (e.g. 10_000_000) to
        disable subsampling entirely. Has no effect when `preprocess_transforms`
        is provided directly, or when `preset=DEFAULT`.
        NOTE: this is entirely independent of `ignore_pretraining_limits`.
    model_version:
        Which model version's defaults to derive presets from. Should match
        the weights specified by `model_path`. Defaults to "v2.6", which is
        what `model_path="auto"` resolves to.
    n_estimators:
        Number of ensemble members (forward passes) per preprocessor config.
    model_path:
        Path to model weights, or "auto" to use the default downloaded model.
    device:
        Device string passed to TabPFNClassifier (e.g. "cpu", "cuda", "auto").
    random_state:
        Random seed for reproducibility.
    ignore_pretraining_limits:
        If True, skip the hard limits on n_samples / n_features that TabPFN
        enforces by default (useful for research on larger datasets).
        NOTE: this only skips the validation check — it does NOT disable the
        `max_features_per_estimator` subsampling in the preprocessing pipeline.

    Returns
    -------
    TabPFNClassifier
        A configured but unfitted classifier.
    """
    transforms = (
        preprocess_transforms
        if preprocess_transforms is not None
        else _apply_preset("clf", preset, model_version, max_features_per_estimator)
    )

    # Pass as a dict so only PREPROCESS_TRANSFORMS is overridden; all other
    # fields (outlier removal, fingerprint, etc.) stay as the checkpoint defaults.
    inference_cfg = {"PREPROCESS_TRANSFORMS": transforms} if transforms is not None else None

    return TabPFNClassifier(
        n_estimators=n_estimators,
        model_path=model_path,
        device=device,
        random_state=random_state,
        inference_config=inference_cfg,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )


def build_tabpfn_regressor(
    *,
    preset: PreprocessingPreset = PreprocessingPreset.DEFAULT,
    preprocess_transforms: list[PreprocessorConfig] | None = None,
    max_features_per_estimator: int | None = None,
    model_version: ModelVersionStr = "v2.6",
    n_estimators: int = 8,
    model_path: str = "auto",
    device: str = "auto",
    random_state: int | None = 0,
    ignore_pretraining_limits: bool = False,
) -> TabPFNRegressor:
    """Build a TabPFNRegressor with configurable preprocessing.

    Parameters
    ----------
    preset:
        A named preset from `PreprocessingPreset`. Ignored when
        `preprocess_transforms` is provided.
    preprocess_transforms:
        Explicit list of `PreprocessorConfig` objects. Overrides `preset` when
        given.
    max_features_per_estimator:
        Cap on the number of features passed to each ensemble member. Features
        are randomly subsampled to this limit *before* any transformation.
        Default (None) keeps whatever each preset specifies.
        Set to a large value (e.g. 10_000_000) to disable subsampling entirely.
        NOTE: this is entirely independent of `ignore_pretraining_limits`.
    model_version:
        Which model version's defaults to derive presets from. Should match
        the weights specified by `model_path`. Defaults to "v2.6".
    n_estimators:
        Number of ensemble members (forward passes) per preprocessor config.
    model_path:
        Path to model weights, or "auto" to use the default downloaded model.
    device:
        Device string passed to TabPFNRegressor (e.g. "cpu", "cuda", "auto").
    random_state:
        Random seed for reproducibility.
    ignore_pretraining_limits:
        If True, skip the hard limits on n_samples / n_features.
        NOTE: this only skips the validation check — it does NOT disable the
        `max_features_per_estimator` subsampling in the preprocessing pipeline.

    Returns
    -------
    TabPFNRegressor
        A configured but unfitted regressor.
    """
    transforms = (
        preprocess_transforms
        if preprocess_transforms is not None
        else _apply_preset("reg", preset, model_version, max_features_per_estimator)
    )

    inference_cfg = {"PREPROCESS_TRANSFORMS": transforms} if transforms is not None else None

    return TabPFNRegressor(
        n_estimators=n_estimators,
        model_path=model_path,
        device=device,
        random_state=random_state,
        inference_config=inference_cfg,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )


__all__ = [
    "ModelVersionStr",
    "PreprocessingPreset",
    "PreprocessorConfig",
    "build_tabpfn_classifier",
    "build_tabpfn_regressor",
]
