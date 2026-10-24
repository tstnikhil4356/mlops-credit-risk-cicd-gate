"""Model families and full pipeline construction."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_risk.features import build_encoder, build_feature_builder

# Families that need standardised inputs.
_SCALED = {"logistic_regression"}


def make_estimator(family: str, params: dict[str, Any], seed: int):
    params = dict(params)
    if family == "dummy":
        return DummyClassifier(**params)
    if family == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if family == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown model family: {family}")


def build_pipeline(cfg, family: str, params: dict[str, Any], seed: int) -> Pipeline:
    """features -> encoding -> model. This single object is trained, logged and served."""
    return Pipeline(
        [
            ("features", build_feature_builder(cfg)),
            ("encode", build_encoder(cfg, scale_numeric=family in _SCALED)),
            ("model", make_estimator(family, params, seed)),
        ]
    )


def search_grid(search_space: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Prefix search-space keys with the pipeline step name."""
    return {f"model__{k}": v for k, v in search_space.items()}
