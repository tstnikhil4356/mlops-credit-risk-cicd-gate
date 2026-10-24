"""YAML configuration loading and validation.

Every tunable value used by the pipeline lives in ``configs/*.yaml``. An
environment file (``dev.yaml``, ``ci.yaml``, ``full.yaml``) declares
``extends: base.yaml`` and overrides only what differs; the merged result is
validated against the Pydantic schema below. Unknown keys, wrong types and
inconsistent values fail immediately with a message naming the offending field.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded or is invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    raw_data: str
    data_versions: str
    artifacts_dir: str
    reports_dir: str
    baseline: str


class DataConfig(_Strict):
    target: str
    id_column: str
    numeric_columns: list[str] = Field(min_length=1)
    categorical_columns: list[str]
    income_categories: list[str] = Field(min_length=1)
    test_size: float = Field(gt=0.0, lt=0.5)
    max_missing_fraction: float = Field(ge=0.0, le=1.0)


class FeatureConfig(_Strict):
    utilisation_risk_cutoff: float = Field(gt=0.0, le=2.0)
    payment_delay_flag_days: float = Field(ge=0.0)
    age_bins: list[float] = Field(min_length=3)
    age_labels: list[str]
    impute_strategy: Literal["median", "mean"]
    add_missing_indicator: bool
    drop_features: list[str] = []

    @model_validator(mode="after")
    def _bins_consistent(self) -> "FeatureConfig":
        if any(lo >= hi for lo, hi in zip(self.age_bins[:-1], self.age_bins[1:])):
            raise ValueError("features.age_bins must be strictly increasing")
        if len(self.age_labels) != len(self.age_bins) - 1:
            raise ValueError(
                f"features.age_labels needs {len(self.age_bins) - 1} labels for "
                f"{len(self.age_bins)} bin edges, got {len(self.age_labels)}"
            )
        return self


Family = Literal["dummy", "logistic_regression", "random_forest", "hist_gradient_boosting"]


class CandidateConfig(_Strict):
    name: str
    family: Family
    params: dict[str, Any] = {}
    search_space: dict[str, list[Any]] = {}

    @field_validator("search_space")
    @classmethod
    def _non_empty_lists(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for key, values in v.items():
            if not values:
                raise ValueError(f"search_space.{key} must list at least one value")
        return v


class TrainingConfig(_Strict):
    cv_folds: int = Field(ge=2, le=20)
    selection_metric: Literal["average_precision", "roc_auc", "f1", "recall", "balanced_accuracy"]
    search: Literal["grid", "random"]
    n_iter: int = Field(ge=1)
    threshold_strategy: Literal["max_fbeta", "fixed"]
    threshold_beta: float = Field(gt=0.0)
    fixed_threshold: float = Field(gt=0.0, lt=1.0)
    n_jobs: int
    candidates: list[CandidateConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> "TrainingConfig":
        names = [c.name for c in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError(f"training.candidates names must be unique, got {names}")
        return self


class TrackingConfig(_Strict):
    enabled: bool
    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    log_model: bool


class GateConfig(_Strict):
    metric: Literal["average_precision", "roc_auc", "f1", "recall", "f2", "balanced_accuracy"]
    max_drop: float = Field(ge=0.0, lt=1.0)
    guard_metric: Literal["recall", "f2", "roc_auc", "average_precision"] | None = None
    guard_min: float = Field(default=0.0, ge=0.0, le=1.0)
    max_minutes: float = Field(gt=0.0)


class Config(_Strict):
    name: str
    description: str = ""
    seed: int = Field(ge=0)
    paths: PathsConfig
    data: DataConfig
    features: FeatureConfig
    training: TrainingConfig
    tracking: TrackingConfig
    gate: GateConfig

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def sha256(self) -> str:
        """Stable fingerprint of the resolved configuration (logged with each run)."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def with_overrides(self, **dotted: Any) -> "Config":
        """Return a validated copy with ``section.key=value`` overrides applied."""
        raw = self.to_dict()
        for key, value in dotted.items():
            _set_dotted(raw, key.replace("__", "."), value)
        return _validate(raw, source=f"{self.name} + overrides")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_dotted(raw: dict[str, Any], dotted: str, value: Any) -> None:
    node = raw
    *parents, leaf = dotted.split(".")
    for part in parents:
        if not isinstance(node.get(part), dict):
            raise ConfigError(f"Override '{dotted}': '{part}' is not a config section")
        node = node[part]
    node[leaf] = value


def _read_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"Circular 'extends' chain: {' -> '.join(str(p) for p in (*seen, path))}")
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: invalid YAML syntax\n{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping, got {type(raw).__name__}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_read_yaml(path.parent / parent, (*seen, path)), raw)


def _validate(raw: dict[str, Any], source: str) -> Config:
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        lines = [f"Invalid configuration ({source}):"]
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"  - {loc}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from None


def load_config(path: str | Path) -> Config:
    """Load ``path`` (following ``extends``), validate it and return a frozen Config."""
    path = Path(path)
    return _validate(_read_yaml(path), source=str(path))
