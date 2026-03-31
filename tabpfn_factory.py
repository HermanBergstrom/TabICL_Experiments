"""Factory for constructing TabPFN models with modified preprocessing configurations.

This module lets you easily ablate or replace individual preprocessing steps in
TabPFN without touching the model weights. Use the high-level `build_tabpfn_*`
functions for common cases, or construct a `PreprocessorConfig` list manually for
full control.

Quick example
-------------
    from tabpfn_factory import build_tabpfn_classifier, PreprocessingPreset

    # Default TabPFN
    clf = build_tabpfn_classifier()

    # No feature-distribution reshaping (identity transform), no SVD
    clf = build_tabpfn_classifier(preset=PreprocessingPreset.NO_RESHAPE)

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

Configurable fields in PreprocessorConfig
------------------------------------------
    name : str
        Feature-distribution transform. Key options:
            "none"                        – identity (no transform)
            "squashing_scaler_default"    – squashing scaler (default for clf)
            "safepower"                   – Yeo-Johnson power transform
            "quantile_uni_coarse"         – uniform quantile (default for reg)
            "quantile_norm_coarse"        – normal quantile
            "quantile_uni" / "quantile_norm" / "_fine" variants
            "robust"                      – robust scaler
            "kdi" and many kdi_* variants
    categorical_name : str
        Categorical encoding. Options:
            "none"                                 – no encoding
            "numeric"                              – treat as numbers
            "ordinal"                              – ordinal encode
            "ordinal_shuffled"                     – ordinal, random order
            "ordinal_very_common_categories_shuffled"  – ordinal, rare → NaN
    append_original : bool | "auto"
        If True, append transformed features to originals instead of replacing.
        "auto" enables this when n_features < 500.
    max_features_per_estimator : int
        Randomly subsample features to at most this many per estimator.
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
from tabpfn.inference_config import InferenceConfig
from tabpfn.preprocessing import (
    PreprocessorConfig,
    default_classifier_preprocessor_configs,
    default_regressor_preprocessor_configs,
)


class PreprocessingPreset(Enum):
    """Named presets for common ablation scenarios."""

    DEFAULT = "default"
    """The default TabPFN preprocessing (unmodified)."""

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


def _apply_preset_classifier(preset: PreprocessingPreset) -> list[PreprocessorConfig]:
    defaults = default_classifier_preprocessor_configs()

    if preset == PreprocessingPreset.DEFAULT:
        return defaults

    if preset == PreprocessingPreset.NO_RESHAPE:
        return [
            PreprocessorConfig(
                name="none",
                categorical_name="ordinal_very_common_categories_shuffled",
                global_transformer_name=None,
                max_features_per_estimator=500,
            ),
            PreprocessorConfig(
                name="none",
                categorical_name="numeric",
                global_transformer_name=None,
                max_features_per_estimator=500,
            ),
        ]

    if preset == PreprocessingPreset.NO_SVD:
        return [
            PreprocessorConfig(
                name=c.name,
                categorical_name=c.categorical_name,
                append_original=c.append_original,
                max_features_per_estimator=c.max_features_per_estimator,
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
                max_features_per_estimator=c.max_features_per_estimator,
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
                max_features_per_estimator=500,
            )
        ]

    raise ValueError(f"Unknown preset: {preset}")


def _apply_preset_regressor(preset: PreprocessingPreset) -> list[PreprocessorConfig]:
    defaults = default_regressor_preprocessor_configs()

    if preset == PreprocessingPreset.DEFAULT:
        return defaults

    if preset == PreprocessingPreset.NO_RESHAPE:
        return [
            PreprocessorConfig(
                name="none",
                categorical_name=c.categorical_name,
                global_transformer_name=None,
                max_features_per_estimator=c.max_features_per_estimator,
            )
            for c in defaults
        ]

    if preset == PreprocessingPreset.NO_SVD:
        return [
            PreprocessorConfig(
                name=c.name,
                categorical_name=c.categorical_name,
                append_original=c.append_original,
                max_features_per_estimator=c.max_features_per_estimator,
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
                max_features_per_estimator=c.max_features_per_estimator,
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
                max_features_per_estimator=500,
            )
        ]

    raise ValueError(f"Unknown preset: {preset}")


def build_tabpfn_classifier(
    *,
    preset: PreprocessingPreset = PreprocessingPreset.DEFAULT,
    preprocess_transforms: list[PreprocessorConfig] | None = None,
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
    n_estimators:
        Number of ensemble members (forward passes) per preprocessor config.
    model_path:
        Path to model weights, or "auto" to use the default downloaded model.
    device:
        Device string passed to TabPFNClassifier (e.g. "cpu", "cuda", "auto").
    random_state:
        Random seed for reproducibility.
    ignore_pretraining_limits:
        If True, remove the hard limits on n_samples / n_features that
        TabPFN enforces by default (useful for research on larger datasets).

    Returns
    -------
    TabPFNClassifier
        A configured but unfitted classifier.
    """
    transforms = (
        preprocess_transforms
        if preprocess_transforms is not None
        else _apply_preset_classifier(preset)
    )

    inference_cfg = InferenceConfig(PREPROCESS_TRANSFORMS=transforms)

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
    n_estimators:
        Number of ensemble members (forward passes) per preprocessor config.
    model_path:
        Path to model weights, or "auto" to use the default downloaded model.
    device:
        Device string passed to TabPFNRegressor (e.g. "cpu", "cuda", "auto").
    random_state:
        Random seed for reproducibility.
    ignore_pretraining_limits:
        If True, remove the hard limits on n_samples / n_features.

    Returns
    -------
    TabPFNRegressor
        A configured but unfitted regressor.
    """
    transforms = (
        preprocess_transforms
        if preprocess_transforms is not None
        else _apply_preset_regressor(preset)
    )

    inference_cfg = InferenceConfig(PREPROCESS_TRANSFORMS=transforms)

    return TabPFNRegressor(
        n_estimators=n_estimators,
        model_path=model_path,
        device=device,
        random_state=random_state,
        inference_config=inference_cfg,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )


__all__ = [
    "PreprocessingPreset",
    "PreprocessorConfig",
    "build_tabpfn_classifier",
    "build_tabpfn_regressor",
]
