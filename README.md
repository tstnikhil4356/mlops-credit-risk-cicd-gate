# Credit Risk Model with a CI/CD Quality Gate

*Elective II: Machine Learning Operations (MLOps), Project 1. Brief by Dr. Santosh Kumar.*
*Team (pair): Nikhil · Harsh*

A retail bank retrains its credit-card default model every month. This repository implements the workflow the risk team asked for:

- **every change** to code, configuration, features or data is **versioned**, **tested automatically**, and **tracked** in an experiment log;
- a change is **accepted only if the model is not worse than the baseline**;
- the gate's rule is **justified with evidence** from our own run history.

| | |
|---|---|
| Data versioning | Git + **DVC** (remote on the orphan branch `dvc-storage`) |
| Configuration | **YAML** + **Pydantic** validation (`configs/`) |
| Modelling | pandas, **scikit-learn** (logistic regression, random forest, gradient boosting) |
| Tracking & registry | **MLflow** (SQLite backend, Model Registry, alias `@champion`) |
| CI/CD | **GitHub Actions** + **pytest** (`.github/workflows/quality-gate.yml`) |

All tools are free and open source; no cloud account or secret is needed.

## Results at a glance

| | |
|---|---|
| Registered model | `credit-risk-classifier` v1 `@champion`: tuned logistic regression, CV PR-AUC **0.873**, test PR-AUC **0.876**, recall **0.97** at the F2-tuned threshold ([registry record](reports/registry/registered_model.json)) |
| Tracked runs | 60+ runs across selection, ablation, gate history, drill and baseline ([all runs](reports/tracking/all_runs.csv)) |
| Gate rule | `PR-AUC ≥ baseline − 0.02 AND recall ≥ 0.85`: the only one of 8 candidate rules with 0 false blocks and 0 false accepts on our history ([memo](docs/gate_defense_memo.md)) |
| v1 → v2 | +0.0235 on the CI seed, but −0.0004 ± 0.013 on identical customers over 5 seeds: **accept as "not worse", not as "better"** ([drill](docs/data_change_drill.md)) |
| Fairness | Income-group error gaps of 5–7 points, inside overlapping CIs, unchanged without the income feature: acceptable, monitored ([analysis](docs/threshold_and_fairness.md)) |

## Quick start (fresh clone)

Requires Python 3.11 and Git.

```bash
git clone <repo-url> && cd <repo>
make setup        # .venv + locked dependencies (requirements.lock) + package
make data         # fetch the dvc-storage branch and `dvc pull` the data of this commit
make test         # 58 unit tests
make gate         # exactly what CI does: tests + train (ci.yaml) + compare with baseline
```

Without `make`:

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.lock && .venv/bin/pip install -e . --no-deps
DVC=.venv/bin/dvc scripts/dvc_storage.sh pull
.venv/bin/pytest
.venv/bin/python -m credit_risk.train --config configs/ci.yaml --out reports/metrics/ci_metrics.json
.venv/bin/python -m credit_risk.gate --config configs/ci.yaml --candidate reports/metrics/ci_metrics.json
```

### Reproduce every reported result

```bash
make reproduce    # fresh MLflow store -> experiments -> register -> gate analysis -> fairness -> drill -> evidence
make ui           # browse runs and the registry at http://127.0.0.1:5000
make restore-demo # Task 2: fresh clone of tag baseline-v1 + its data -> identical result
```

All runs are deterministic (pinned lock file, seeded split, CV and models), so `make reproduce` regenerates the same numbers. Run IDs differ because they are new runs.

### Score new customers with the registered model

```bash
.venv/bin/python -m credit_risk.predict --input new_customers.csv --output scored.csv
```

## Repository layout

```
.
├── configs/                    # Task 3 - all tunable values (validated on load)
│   ├── base.yaml               #   defaults: paths, data contract, features, search spaces, seed, tracking, gate
│   ├── dev.yaml                #   fast local loop
│   ├── ci.yaml                 #   trained by the quality gate on every PR
│   └── full.yaml               #   full model-selection campaign
├── data/
│   ├── raw/credit_card_customers.csv.dvc   # Task 2 - DVC pointer (CSV itself is NOT in Git)
│   └── versions.yaml           #   md5 -> v1 / v2 labels
├── src/credit_risk/
│   ├── config.py               #   YAML loading, extends/overrides, Pydantic schema, config hash
│   ├── data.py                 #   validation, DVC md5 versioning, ID-hash split
│   ├── features.py             # Task 4 - CreditFeatureBuilder + encoders
│   ├── models.py               #   model families and full pipeline
│   ├── evaluate.py             #   imbalanced metrics, F-beta threshold, subgroup metrics + bootstrap
│   ├── train.py                # Task 5 - CV tuning, threshold, MLflow logging with provenance
│   ├── register.py             #   select by CV PR-AUC, register, lineage tags, @champion alias
│   ├── gate.py                 # Task 6 - baseline comparison, reviewer summary, exit code
│   ├── predict.py              #   batch inference with the registered pipeline
│   └── tracking.py             #   git provenance, MLflow setup
├── tests/                      # 58 tests: features, config, data contract, gate, pipeline parity
│   └── fixtures/broken_*.yaml  #   deliberately broken configs
├── baseline/baseline_metrics.json   # gate reference - CODEOWNERS + baseline-guard protected
├── scripts/
│   ├── run_experiments.py      # Task 5/8 - selection, ablation, gate-history campaign
│   ├── gate_rule_analysis.py   # Task 8 - eight candidate gate rules vs run history
│   ├── fairness_analysis.py    # Task 8 - Income_Bracket breakdown + counterfactual
│   ├── data_drill.py           # Task 7 - v1 vs v2 over 5 seeds, paired bootstrap, drift
│   ├── update_baseline.py      # Task 6 - the only way to change the baseline (+ --check)
│   ├── restore_demo.sh         # Task 2 - restore code + data and reproduce
│   ├── dvc_storage.sh          #   DVC remote on the dvc-storage branch (pull / push)
│   ├── export_tracking.py      #   tracking evidence: runs CSV, comparison, registry
│   ├── screenshot_mlflow.sh    #   MLflow UI screenshots (headless Chrome)
│   ├── make_diagrams.py        #   lifecycle map and CI flow figures
│   └── build_report.py         #   report.md -> PDF + DOCX
├── .github/
│   ├── workflows/quality-gate.yml   # quality-gate + baseline-guard jobs
│   ├── CODEOWNERS
│   └── pull_request_template.md
├── docs/                       # one document per task + report + memo (index below)
├── reports/                    # generated evidence (tables, figures, logs, screenshots)
├── requirements.txt            # direct dependencies (pinned)
├── requirements.lock           # full transitive lock used by CI and `make setup`
└── Makefile
```

## Documentation and deliverables

| Deliverable (brief §6) | Where |
|---|---|
| Repository with code, configuration, tests, workflow files and README | this repository |
| Experiment-tracking evidence: ≥ 10 runs + registered model | [`reports/tracking/`](reports/tracking) (runs CSV, comparison, registry, **screenshots**) |
| Lifecycle mapping diagram / table | [`docs/lifecycle.md`](docs/lifecycle.md), [`reports/figures/lifecycle_map.png`](reports/figures/lifecycle_map.png) |
| CI history with the three required PRs | [`docs/ci_quality_gate.md`](docs/ci_quality_gate.md), [`reports/ci/pr_history.md`](reports/ci/pr_history.md), GitHub *Pull requests* / *Actions* tabs |
| Feature documentation table + one-page metric justification | [`docs/features.md`](docs/features.md), [`docs/metric_justification.md`](docs/metric_justification.md) |
| Threshold comparison, subgroup breakdown, fairness-defense memo | [`docs/threshold_and_fairness.md`](docs/threshold_and_fairness.md), [`docs/gate_defense_memo.md`](docs/gate_defense_memo.md) |
| Short report (≤ 4 pages) | [`docs/report.md`](docs/report.md) → [`reports/report.pdf`](reports/report.pdf) / [`.docx`](reports/report.docx) |
| 5-minute demonstration | [`docs/demo_script.md`](docs/demo_script.md) |
| Requirement-by-requirement traceability | [`docs/requirements_checklist.md`](docs/requirements_checklist.md) |

Task-specific design documents: [configuration](docs/configuration.md) · [data versioning & restore](docs/data_versioning.md) · [data-change drill](docs/data_change_drill.md) · [references & citations](docs/references.md).

## Rules respected

- **Free tools only**: everything runs on a laptop and free GitHub Actions minutes.
- **Reproducible from a fresh clone**: `make setup data reproduce`; verified by `scripts/restore_demo.sh`, which clones into a temporary directory.
- **No secrets committed**: CI uses only the built-in `GITHUB_TOKEN`; the DVC remote needs no credentials.
- **Every reported result is traceable to a tracked run**: generated tables carry MLflow run IDs; every run is tagged with git commit, data md5/version and config hash.
- **External code and tutorials cited**: [`docs/references.md`](docs/references.md).
