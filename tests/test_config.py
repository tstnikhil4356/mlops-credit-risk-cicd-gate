from pathlib import Path

import pytest

from credit_risk.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", ["base", "dev", "ci", "full"])
def test_all_shipped_configs_are_valid(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg.name == name


def test_environment_overrides_base():
    base, dev = load_config(ROOT / "configs/base.yaml"), load_config(ROOT / "configs/dev.yaml")
    assert dev.training.cv_folds < base.training.cv_folds
    assert dev.seed == base.seed  # inherited
    assert dev.tracking.experiment_name != base.tracking.experiment_name


def test_unknown_key_fails_with_clear_message():
    with pytest.raises(ConfigError, match=r"features\.utilization_risk_cutoff: Extra inputs are not permitted"):
        load_config(FIX / "broken_unknown_key.yaml")


def test_bad_values_report_every_field():
    with pytest.raises(ConfigError) as exc:
        load_config(FIX / "broken_bad_values.yaml")
    msg = str(exc.value)
    for field in ("seed", "data.test_size", "training.cv_folds", "gate.metric"):
        assert field in msg


def test_inconsistent_bins_fail():
    with pytest.raises(ConfigError, match="strictly increasing"):
        load_config(FIX / "broken_bins.yaml")


def test_yaml_syntax_error_fails():
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(FIX / "broken_syntax.yaml")


def test_missing_file_fails():
    with pytest.raises(ConfigError, match="not found"):
        load_config(FIX / "nope.yaml")


def test_override_is_validated(cfg):
    assert cfg.with_overrides(**{"gate.max_drop": 0.05}).gate.max_drop == 0.05
    with pytest.raises(ConfigError, match="gate.max_drop"):
        cfg.with_overrides(**{"gate.max_drop": -1})


def test_config_hash_is_stable_and_sensitive(cfg):
    assert cfg.sha256() == load_config(ROOT / "configs/base.yaml").sha256()
    assert cfg.sha256() != cfg.with_overrides(seed=7).sha256()


def test_config_is_immutable(cfg):
    with pytest.raises(Exception):
        cfg.seed = 1
