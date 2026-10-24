"""Cleaning and feature engineering.

All logic lives in a single scikit-learn transformer (``CreditFeatureBuilder``)
that is the first step of the model pipeline. Imputation statistics are learned
in ``fit`` on training rows only, and the fitted pipeline is what gets logged
and registered, so training, CV folds, the CI gate and inference all apply
byte-for-byte the same transformations. See docs/features.md for the table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Upper cap for ratios: protects models from extreme values when a limit is tiny.
RATIO_CAP = 3.0

ENGINEERED_NUMERIC = [
    "utilisation",
    "spend_to_limit",
    "credit_stress_index",
    "spend_per_transaction",
]
ENGINEERED_FLAGS = ["high_utilisation_flag", "late_payer_flag", "record_incomplete"]
ENGINEERED_CATEGORICAL = ["age_band"]
ENGINEERED = ENGINEERED_NUMERIC + ENGINEERED_FLAGS + ENGINEERED_CATEGORICAL


def safe_ratio(numerator: pd.Series, denominator: pd.Series, cap: float = RATIO_CAP) -> pd.Series:
    """numerator / denominator with explicit, documented edge-case handling.

    * denominator > 0  -> plain ratio, clipped to [0, cap]
    * denominator <= 0 and numerator > 0 -> cap (owing money with no limit is maximal risk)
    * denominator <= 0 and numerator == 0 -> 0 (nothing owed, nothing at risk)
    * either side missing -> NaN (handled by imputation upstream)
    """
    num = numerator.astype(float)
    den = denominator.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den.where(den > 0)
    ratio = ratio.clip(lower=0.0, upper=cap)
    no_limit = (den <= 0) & num.notna()
    ratio = ratio.mask(no_limit & (num > 0), cap).mask(no_limit & (num <= 0), 0.0)
    return ratio


class CreditFeatureBuilder(BaseEstimator, TransformerMixin):
    """Impute raw columns and derive engineered features.

    Parameters mirror the ``features`` section of the YAML config so that the
    whole behaviour is configuration-driven and recorded with the model.
    """

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        id_column: str = "Customer_ID",
        target: str = "High_Risk_Customer",
        utilisation_risk_cutoff: float = 0.8,
        payment_delay_flag_days: float = 7,
        age_bins: list[float] | None = None,
        age_labels: list[str] | None = None,
        impute_strategy: str = "median",
        add_missing_indicator: bool = True,
        drop_features: list[str] | None = None,
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.id_column = id_column
        self.target = target
        self.utilisation_risk_cutoff = utilisation_risk_cutoff
        self.payment_delay_flag_days = payment_delay_flag_days
        self.age_bins = age_bins
        self.age_labels = age_labels
        self.impute_strategy = impute_strategy
        self.add_missing_indicator = add_missing_indicator
        self.drop_features = drop_features

    # -- fitting ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X)
        agg = X[self.numeric_columns].median if self.impute_strategy == "median" else X[self.numeric_columns].mean
        self.fill_values_ = agg(numeric_only=True).to_dict()
        self.category_modes_ = {c: X[c].mode(dropna=True).iloc[0] for c in self.categorical_columns}
        self.feature_names_out_ = list(self.transform(X.head(1)).columns)
        return self

    # -- transforming ----------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._check_input(X)
        raw = X[self.numeric_columns + self.categorical_columns].copy()
        out = pd.DataFrame(index=raw.index)

        if self.add_missing_indicator:
            out["record_incomplete"] = raw[self.numeric_columns].isna().any(axis=1).astype(int)

        num = raw[self.numeric_columns].astype(float).fillna(self.fill_values_)
        cats = raw[self.categorical_columns].fillna(self.category_modes_)
        out = pd.concat([num, cats, out], axis=1)

        out["utilisation"] = safe_ratio(num["Current_Balance"], num["Credit_Limit"])
        out["spend_to_limit"] = safe_ratio(num["Monthly_Spend"], num["Credit_Limit"])
        out["high_utilisation_flag"] = (out["utilisation"] >= self.utilisation_risk_cutoff).astype(int)
        out["late_payer_flag"] = (num["Payment_Delay_Days"] > self.payment_delay_flag_days).astype(int)
        # Own feature: risk compounds when a customer is both stretched and paying late.
        out["credit_stress_index"] = out["utilisation"] * np.log1p(num["Payment_Delay_Days"].clip(lower=0))
        out["spend_per_transaction"] = num["Monthly_Spend"] / num["Num_Transactions_Monthly"].clip(lower=1)
        out["age_band"] = pd.cut(
            num["Age"], bins=self.age_bins, labels=self.age_labels, right=False, include_lowest=True
        ).astype(str)

        drops = [c for c in (self.drop_features or []) if c in out.columns]
        return out.drop(columns=drops)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CreditFeatureBuilder expects a pandas DataFrame")
        if self.target in X.columns:
            raise ValueError(f"Target column '{self.target}' must not be passed to the feature builder")
        missing = [c for c in self.numeric_columns + self.categorical_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        # Customer_ID (and any other extra column) is deliberately ignored here.
        return X


def build_feature_builder(cfg) -> CreditFeatureBuilder:
    f, d = cfg.features, cfg.data
    return CreditFeatureBuilder(
        numeric_columns=list(d.numeric_columns),
        categorical_columns=list(d.categorical_columns),
        id_column=d.id_column,
        target=d.target,
        utilisation_risk_cutoff=f.utilisation_risk_cutoff,
        payment_delay_flag_days=f.payment_delay_flag_days,
        age_bins=list(f.age_bins),
        age_labels=list(f.age_labels),
        impute_strategy=f.impute_strategy,
        add_missing_indicator=f.add_missing_indicator,
        drop_features=list(f.drop_features),
    )


def build_encoder(cfg, scale_numeric: bool) -> ColumnTransformer:
    """Encode the feature-builder output: ordinal income, one-hot age band, scaled numerics."""
    dropped = set(cfg.features.drop_features)
    ordinal = [c for c in cfg.data.categorical_columns if c not in dropped]
    onehot = [c for c in ENGINEERED_CATEGORICAL if c not in dropped]
    numeric = [
        c
        for c in [*cfg.data.numeric_columns, *ENGINEERED_NUMERIC, *ENGINEERED_FLAGS]
        if c not in dropped and (c != "record_incomplete" or cfg.features.add_missing_indicator)
    ]
    transformers = [("numeric", StandardScaler() if scale_numeric else "passthrough", numeric)]
    if ordinal:
        transformers.append(
            (
                "income",
                OrdinalEncoder(
                    categories=[list(cfg.data.income_categories)] * len(ordinal),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal,
            )
        )
    if onehot:
        transformers.append(
            ("age_band", OneHotEncoder(categories=[list(cfg.features.age_labels)], handle_unknown="ignore"), onehot)
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
