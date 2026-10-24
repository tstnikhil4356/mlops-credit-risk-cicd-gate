"""Data loading, validation, versioning and splitting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from credit_risk.config import Config


class DataValidationError(ValueError):
    """Raised when the raw dataset violates the expected contract."""


# Plausible value ranges (domain contract). Values outside these are data errors,
# not modelling problems, so they stop the pipeline instead of being silently fixed.
VALUE_RANGES: dict[str, tuple[float, float]] = {
    "Age": (18, 120),
    "Credit_Limit": (0, 1_000_000),
    "Current_Balance": (0, 1_000_000),
    "Monthly_Spend": (0, 1_000_000),
    "Num_Transactions_Monthly": (0, 10_000),
    "Payment_Delay_Days": (0, 365),
    "Tenure_Months": (0, 1_200),
}


@dataclass(frozen=True)
class DataVersion:
    md5: str
    label: str
    rows: int
    path: str

    def as_tags(self) -> dict[str, str]:
        return {
            "data_md5": self.md5,
            "data_version": self.label,
            "data_rows": str(self.rows),
            "data_path": self.path,
        }


def file_md5(path: str | Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_version(cfg: Config, path: str | Path | None = None) -> DataVersion:
    """Identify the dataset by its DVC md5 and map it to a label via data/versions.yaml."""
    path = Path(path or cfg.paths.raw_data)
    md5 = file_md5(path)
    registry_path = Path(cfg.paths.data_versions)
    registry = yaml.safe_load(registry_path.read_text())["versions"] if registry_path.exists() else {}
    label = registry.get(md5, {}).get("label", f"unregistered-{md5[:8]}")
    rows = sum(1 for _ in open(path)) - 1
    return DataVersion(md5=md5, label=label, rows=rows, path=str(path))


def load_raw(cfg: Config, path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or cfg.paths.raw_data)
    if not path.is_file():
        raise DataValidationError(
            f"Dataset not found at {path}. Run `dvc pull` (see README, 'Getting the data')."
        )
    df = pd.read_csv(path)
    validate(df, cfg)
    return df


def validate(df: pd.DataFrame, cfg: Config) -> None:
    """Check schema, uniqueness, target values, categories, ranges and missingness."""
    d = cfg.data
    expected = [d.id_column, *d.numeric_columns, *d.categorical_columns, d.target]
    missing_cols = sorted(set(expected) - set(df.columns))
    if missing_cols:
        raise DataValidationError(f"Missing columns: {missing_cols}")
    problems: list[str] = []
    if df[d.id_column].duplicated().any():
        problems.append(f"{int(df[d.id_column].duplicated().sum())} duplicate {d.id_column} values")
    if not set(df[d.target].dropna().unique()) <= {0, 1}:
        problems.append(f"target {d.target} must be 0/1")
    if df[d.target].isna().any():
        problems.append(f"target {d.target} has missing values")
    for col in d.categorical_columns:
        unknown = set(df[col].dropna().unique()) - set(d.income_categories)
        if unknown:
            problems.append(f"{col} has unknown categories {sorted(unknown)}")
    for col in d.numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col} is not numeric")
            continue
        frac = df[col].isna().mean()
        if frac > d.max_missing_fraction:
            problems.append(f"{col} missing fraction {frac:.1%} exceeds {d.max_missing_fraction:.0%}")
        lo, hi = VALUE_RANGES.get(col, (-np.inf, np.inf))
        bad = df[col].dropna().pipe(lambda s: (s < lo) | (s > hi)).sum()
        if bad:
            problems.append(f"{col}: {int(bad)} values outside [{lo}, {hi}]")
    if problems:
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(problems))


def split_mask(ids: pd.Series, test_size: float, seed: int) -> np.ndarray:
    """Deterministic test-set membership from a hash of (seed, Customer_ID).

    A customer's assignment depends only on its id and the seed, never on the
    other rows. So when v2 appends new customers, every v1 customer stays on the
    same side of the split, and v1-vs-v2 comparisons are paired and fair.
    """
    def bucket(value: str) -> float:
        h = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
        return int(h[:12], 16) / float(16**12)

    return ids.astype(str).map(bucket).to_numpy() < test_size


def train_test_split_by_id(
    df: pd.DataFrame, cfg: Config, seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    seed = cfg.seed if seed is None else seed
    is_test = split_mask(df[cfg.data.id_column], cfg.data.test_size, seed)
    X = df.drop(columns=[cfg.data.target])
    y = df[cfg.data.target].astype(int)
    return X[~is_test], X[is_test], y[~is_test], y[is_test]
