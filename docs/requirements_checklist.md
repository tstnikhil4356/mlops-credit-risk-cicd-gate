# Requirements checklist: every point of the brief → evidence

Status legend: ✅ done and verified · 🔗 done on GitHub (CI history; links in `reports/ci/pr_history.md`)

## Task 1: Lifecycle map (Topic 1)

| Requirement | Status | Evidence |
|---|---|---|
| Activities mapped to the MLOps lifecycle and the DevOps lifecycle (plan, code, build, test, release, deploy, operate, monitor) | ✅ | `docs/lifecycle.md`, `reports/figures/lifecycle_map.png` |
| Every stage linked to a concrete artefact / workflow step | ✅ | "artefact" column in `docs/lifecycle.md` (file paths, CI steps) |
| ≥ 3 differences ML vs traditional software delivery, explained | ✅ | 5 differences, each with evidence from our runs |
| Stages intentionally left out, named with a reason | ✅ | "Stages intentionally left out" table |

## Task 2: Repository and data versioning (Topic 4)

| Requirement | Status | Evidence |
|---|---|---|
| Code, configuration and data all tracked | ✅ | Git (`src/`, `configs/`), DVC pointer `data/raw/credit_card_customers.csv.dvc`, `data/versions.yaml` |
| Any earlier commit restorable together with its exact data | ✅ | `scripts/restore_demo.sh`, `docs/data_versioning.md` |
| v1 is the starting point; v2 enters later as a tracked change | ✅ 🔗 | tag `data-v1`; branch/PR `data/v2` (pointer diff + registry entry), tag `data-v2` |
| Repository layout documented in a README | ✅ | `README.md#repository-layout` |
| Datasets not stored directly in ordinary Git history | ✅ | `data/raw/.gitignore`, `datasets/` ignored; blobs only in DVC storage (orphan branch) |
| Demonstrate restoring an earlier version of code and data with the same result | ✅ | `reports/restore/restore_baseline-v1.log` (recorded = reproduced AP 0.876207) |
| Meaningful commit history (small commits, clear messages) | ✅ | `git log --oneline` (conventional prefixes, single-purpose commits) |

## Task 3: Configuration management (Topic 10)

| Requirement | Status | Evidence |
|---|---|---|
| No hard-coded values: file paths | ✅ | `paths.*` |
| Feature thresholds (e.g. utilisation risk cut-off) | ✅ | `features.utilisation_risk_cutoff`, `payment_delay_flag_days`, `age_bins` … |
| Hyperparameter search space | ✅ | `training.candidates[*].search_space` |
| Random seed | ✅ | `seed` (+ `--seed`) |
| Experiment-tracking settings | ✅ | `tracking.*` |
| Quality-gate threshold | ✅ | `gate.*` |
| ≥ 2 configurations for different environments | ✅ | `dev.yaml`, `ci.yaml`, `full.yaml` (extend `base.yaml`) |
| Changing YAML changes behaviour without editing code | ✅ | `test_config_change_changes_behaviour_without_code_change`; gate-history config-only runs; PR 2 is YAML-only |
| Configs validated on load; a broken config fails immediately with a clear message | ✅ | `src/credit_risk/config.py`; `tests/fixtures/broken_*.yaml`; `tests/test_config.py`; CI "Validate configuration files" step |
| Configuration used for each run stored with that run | ✅ | MLflow artefact `config/resolved_config.yaml`, params `cfg.*`, tags `config_name`, `config_sha256` |

## Task 4: Cleaning and feature engineering (Topic 11)

| Requirement | Status | Evidence |
|---|---|---|
| Missing values in Credit_Limit / Current_Balance / Payment_Delay_Days: decided and justified | ✅ | `docs/features.md` "Cleaning decisions" (co-missing rows, median + flag) |
| ≥ 4 engineered features: a ratio | ✅ | `utilisation` (+ `spend_to_limit`, `spend_per_transaction`) |
| a binary flag | ✅ | `high_utilisation_flag`, `late_payer_flag` (+ `record_incomplete`) |
| a binned / categorical feature | ✅ | `age_band` |
| a feature of our own design justified by domain reasoning | ✅ | `credit_stress_index` (with an honest ablation) |
| Categorical variables encoded appropriately | ✅ | ordinal `Income_Bracket`, one-hot `age_band` |
| Same transformations re-applicable to new data | ✅ | fitted `Pipeline`; `predict.py`; pickle-parity test |
| Feature table: name, formula, reason for every engineered feature | ✅ | `docs/features.md` |
| No target leakage; Customer_ID not a predictor | ✅ | builder rejects the target; ID invariance test; fold-fitted statistics |
| Training and inference use identical transformation logic | ✅ | single pipeline object logged and registered; `test_training_and_inference_use_identical_transformations` |
| Unit tests cover feature logic including zero credit limit | ✅ | `tests/test_features.py` (zero/negative limit, zero transactions, missing values, boundaries …) |

## Task 5: Training, tuning and tracking (Topic 12)

| Requirement | Status | Evidence |
|---|---|---|
| Baseline model established | ✅ | `dummy_prior` and `logreg_default` runs |
| Tune ≥ 2 model families / settings | ✅ | logistic regression (14 settings), random forest (48), gradient boosting (48) |
| Every run logged with parameters, metrics, dataset version, configuration and model artefact | ✅ | MLflow (`reports/tracking/all_runs.csv` columns `data_version`, `data_md5`, `has_model`, `has_config`) |
| Final model selected with a metric justified for imbalanced data | ✅ | CV PR-AUC; `docs/metric_justification.md` |
| Model registered | ✅ | `credit-risk-classifier` v1 `@champion`; `reports/tracking/registry.md` |
| ≥ 10 tracked runs available for comparison | ✅ | 38 runs; `reports/tracking/model_comparison.md`; MLflow screenshots |
| Comparison table + one-page justification of why accuracy is not sufficient | ✅ | `docs/metric_justification.md` |
| Registered model traceable to code commit, data version and configuration | ✅ | model-version tags `git_commit`, `git_dirty`, `data_md5`, `data_version`, `config_sha256`; `reports/registry/registered_model.json` |

## Task 6: Automated quality gate (Topic 2)

| Requirement | Status | Evidence |
|---|---|---|
| CI workflow on every pull request | ✅ | `.github/workflows/quality-gate.yml` (`on: pull_request`) |
| Runs the tests | ✅ | pytest step + JUnit XML |
| Trains the model with the CI configuration | ✅ | `python -m credit_risk.train --config configs/ci.yaml` |
| Compares with a baseline stored in the repository | ✅ | `baseline/baseline_metrics.json`, `credit_risk/gate.py` |
| Blocks the merge if quality falls or a test fails | ✅ 🔗 | final "Enforce" step + required status check (branch protection) |
| Short run summary readable without opening logs | ✅ 🔗 | `$GITHUB_STEP_SUMMARY` + PR comment |
| CI history shows three PRs: pass / fails (model worse) / fails (test fails) | 🔗 | `reports/ci/pr_history.md`, GitHub PRs 1–3 |
| Baseline update rule documented and protected against accidental changes | ✅ | `docs/ci_quality_gate.md`; `scripts/update_baseline.py`; `baseline-guard` job; CODEOWNERS; label rule |
| Finishes in reasonable time with the CI configuration | ✅ 🔗 | ≈5 s training; `gate.max_minutes` budget enforced; Actions durations |

## Task 7: Data-change drill (Topics 2, 4)

| Requirement | Status | Evidence |
|---|---|---|
| Retrain on v2 through the same workflow | ✅ 🔗 | PR "data: dataset v2"; same `ci.yaml` and code path |
| Compare v1 vs v2 models | ✅ | `docs/data_change_drill.md`, `reports/drill/` |
| Investigate whether improvement is real or within run-to-run variation | ✅ | 5 seeds, paired common-row comparison, paired bootstrap, t-test |
| ≥ 3 random seeds per data version | ✅ | seeds 42, 1, 2, 3, 4 |
| Data version identifiers in the tracking records | ✅ | tags `data_version`, `data_md5`, `drill_rev` on every drill run |
| Short written conclusion: should the gate accept the new model, and why | ✅ | "accept as not worse, not as better" |

## Task 8: Gate threshold and fairness defense (Topics 2, 12)

| Requirement | Status | Evidence |
|---|---|---|
| Compare ≥ 3 candidate gate rules against our run history | ✅ | 8 rules × 17 labelled runs, `reports/gate/` |
| Show what each would have accepted / rejected, with concrete numbers | ✅ | `docs/threshold_and_fairness.md` (counts + named runs + run IDs in `gate_decisions.csv`) |
| Check whether the registered model's error is evenly distributed across Income_Bracket | ✅ | `reports/fairness/subgroup_breakdown.md` (test split + out-of-fold + counterfactual, bootstrap CIs) |
| Decide whether any disparity is acceptable | ✅ | "Decision" in `docs/threshold_and_fairness.md` |
| Defense memo: recommend one rule, state the trade-off accepted | ✅ | `docs/gate_defense_memo.md` |
| Memo states what new evidence would change the recommendation, ≤ 1 page | ✅ | final section of the memo (≈570 words) |

## Deliverables (§6)

| Deliverable | Status | Where |
|---|---|---|
| Repository link with code, configuration, tests, workflow files, README | 🔗 | GitHub repository |
| Experiment-tracking evidence (screenshots or export) with ≥ 10 runs and the registered model | ✅ | `reports/tracking/screenshots/`, `all_runs.csv`, `registry.md` |
| Lifecycle mapping diagram or table | ✅ | both |
| CI history showing the three required PRs | 🔗 | GitHub + `reports/ci/pr_history.md` + `reports/ci/screenshots/` |
| Feature documentation table and one-page metric justification | ✅ | `docs/features.md`, `docs/metric_justification.md` |
| Threshold comparison table, subgroup breakdown, fairness-defense memo | ✅ | `docs/threshold_and_fairness.md`, `docs/gate_defense_memo.md` |
| Short report (≤ 4 pages): design decisions + v1 vs v2 comparison | ✅ | `reports/report.pdf`, `reports/report.docx` (source `docs/report.md`) |
| 5-minute demonstration | ✅ script | `docs/demo_script.md` (recording is done by the team) |

## Rules and constraints (§7)

| Rule | Status | How |
|---|---|---|
| Only free tools; no paid cloud | ✅ | Git, DVC, MLflow, scikit-learn, GitHub Actions (free tier) |
| Reproducible from a fresh clone via the README | ✅ | `make setup data reproduce`; restore demo clones to a temp directory |
| No secrets, passwords or tokens committed | ✅ | only `GITHUB_TOKEN` (built in); credential-free DVC remote |
| Every reported result traceable to a tracked run | ✅ | run IDs in every generated table; provenance tags |
| Cite external code or tutorials | ✅ | `docs/references.md` |
| Substitute tools justified | ✅ | local venv vs Colab, orphan-branch DVC remote (`docs/references.md`, `docs/data_versioning.md`) |

## Beyond the brief (extras)

- Data-contract validation (schema, ranges, duplicates, categories, missingness) that fails the pipeline.
- Drift report (KS, PSI, chi²) of new customers.
- Paired bootstrap and a hash-based split that keeps v1 customers on the same side of the split across versions.
- Feature ablations and a screen of alternative own features.
- A fairness counterfactual (model without `Income_Bracket`).
- A gate time budget and a reproducibility check of the baseline itself.
- Self-contained logged models (code + pinned requirements bundled).
- A full transitive lock file.
- A Makefile with one-command entry points.
- MLflow UI screenshots captured automatically (headless Chrome).
