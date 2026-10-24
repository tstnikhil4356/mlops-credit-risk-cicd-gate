# Task 3: Configuration management

Every value that changes behaviour lives in YAML under [`configs/`](../configs) and is validated by a Pydantic schema ([`src/credit_risk/config.py`](../src/credit_risk/config.py)) the moment it is loaded.

## Files and inheritance

| File | Purpose | Key differences |
|---|---|---|
| [`base.yaml`](../configs/base.yaml) | All defaults: paths, data contract, feature thresholds, search spaces, seed, tracking, gate | — |
| [`dev.yaml`](../configs/dev.yaml) | Fast local loop (≈5 s) | 3-fold CV, one small logistic-regression grid, experiment `credit-risk-dev`, no model logging |
| [`ci.yaml`](../configs/ci.yaml) | Trained on **every pull request** by the quality gate | Selected family only (logistic regression, 10-setting grid), `n_jobs: 2`, experiment `credit-risk-ci`; ≈5 s on a laptop |
| [`full.yaml`](../configs/full.yaml) | Model-selection campaign and registry | All five candidates (dummy, 2× logistic regression, random forest, gradient boosting) with full grids (≈50 s) |

An environment file declares `extends: base.yaml` and overrides only what differs (deep merge; a list replaces a list). Ad-hoc experiments use validated overrides without editing any file:

```bash
python -m credit_risk.train --config configs/full.yaml --set features.utilisation_risk_cutoff=0.75 --set seed=7
```

## Everything the brief asks to be configurable

| Brief item | YAML key |
|---|---|
| File paths | `paths.raw_data`, `paths.data_versions`, `paths.artifacts_dir`, `paths.reports_dir`, `paths.baseline` |
| Feature thresholds (utilisation risk cut-off, etc.) | `features.utilisation_risk_cutoff`, `features.payment_delay_flag_days`, `features.age_bins/age_labels`, `features.impute_strategy`, `features.add_missing_indicator`, `features.drop_features` |
| Hyperparameter search space | `training.candidates[*].search_space` (+ `params`, `search`, `n_iter`, `cv_folds`, `selection_metric`, threshold strategy / beta) |
| Random seed | `seed` (also `--seed` on the CLI; drives split, CV folds and model randomness) |
| Experiment-tracking settings | `tracking.enabled`, `tracking.tracking_uri`, `tracking.experiment_name`, `tracking.registered_model_name`, `tracking.log_model` |
| Quality-gate threshold | `gate.metric`, `gate.max_drop`, `gate.guard_metric`, `gate.guard_min`, `gate.max_minutes` |
| Data contract | `data.target`, `data.id_column`, column lists, `income_categories`, `test_size`, `max_missing_fraction` |

No code module contains a literal for any of these values. Model families are chosen by name (`family:`) and built by `credit_risk.models.make_estimator`.

## "Changing a value in the YAML changes behaviour without editing code"

- `tests/test_features.py::test_config_change_changes_behaviour_without_code_change`: moving the utilisation cut-off from 0.1 to 1.5 changes the number of flagged customers.
- In the tracked gate history, the cut-off 0.75 run, the delay-flag 5 run and the mean-imputation run are all pure YAML overrides, each a separate tracked run with its own `config_sha256` ([gate history](../reports/gate/gate_rule_comparison.md)).
- The two failing demonstration PRs change only YAML ([`ci_quality_gate.md`](ci_quality_gate.md)).

## Validation: broken configurations fail immediately with clear messages

The schema forbids unknown keys (typos), checks types and ranges, and checks cross-field consistency (bins strictly increasing, one label per bin, unique candidate names, non-empty search lists). CLIs print the error and exit with code 2 before any data is read.

```text
$ python -m credit_risk.train --config tests/fixtures/broken_unknown_key.yaml
ERROR: Invalid configuration (tests/fixtures/broken_unknown_key.yaml):
  - features.utilization_risk_cutoff: Extra inputs are not permitted

$ python -m credit_risk.train --config tests/fixtures/broken_bad_values.yaml
ERROR: Invalid configuration (tests/fixtures/broken_bad_values.yaml):
  - seed: Input should be greater than or equal to 0
  - data.test_size: Input should be less than 0.5
  - training.cv_folds: Input should be a valid integer, unable to parse string as an integer
  - gate.metric: Input should be 'average_precision', 'roc_auc', 'f1', 'recall', 'f2' or 'balanced_accuracy'
```

Also covered by tests: inconsistent age bins, YAML syntax errors, a missing file, invalid CLI overrides, and an immutable config object. CI validates every file in `configs/` as its own step before training.

## The configuration used for each run is stored with that run

Each MLflow run gets:

- the **resolved** configuration (after `extends` and overrides) as the artefact `config/resolved_config.yaml`;
- every scalar config value as a parameter `cfg.<section>.<key>`;
- the tag `config_name` and the tag `config_sha256` (SHA-256 of the canonical JSON of the resolved config).

The gate summary and the baseline file record the same hash, so a reviewer can see whether a PR changed configuration at all.
