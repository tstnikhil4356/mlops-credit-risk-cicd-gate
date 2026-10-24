# Task 1: Lifecycle map

![Lifecycle map](../reports/figures/lifecycle_map.png)

## DevOps stage → MLOps activity → artefact in this repository

| DevOps stage | What happens in this project (ML view) | Concrete artefact / workflow step |
|---|---|---|
| **Plan** | Define the business problem (monthly default model), choose the decision metric for imbalanced data, define what "not worse" means (gate rule) and the fairness question. | [`docs/metric_justification.md`](metric_justification.md), [`docs/gate_defense_memo.md`](gate_defense_memo.md), [`.github/pull_request_template.md`](../.github/pull_request_template.md) |
| **Code** | Feature logic, training/tuning code, *configuration* and *data* are all versioned. Data enters through DVC pointer files, not Git blobs. | [`src/credit_risk/`](../src/credit_risk), [`configs/*.yaml`](../configs), [`data/raw/credit_card_customers.csv.dvc`](../data/raw/credit_card_customers.csv.dvc), [`data/versions.yaml`](../data/versions.yaml) |
| **Build** | Reproducible environment from a full lock file; data for the exact commit is pulled from DVC storage; every YAML config is validated (Pydantic) before anything runs. | [`requirements.lock`](../requirements.lock), [`scripts/dvc_storage.sh`](../scripts/dvc_storage.sh), CI steps *Install*, *Pull data*, *Validate configuration files* |
| **Test** | Two levels: (1) software tests for feature edge cases, config/data contracts and pipeline parity; (2) **model tests**: retrain with `configs/ci.yaml` and compare to the stored baseline. | [`tests/`](../tests) (58 tests), [`src/credit_risk/gate.py`](../src/credit_risk/gate.py), [`.github/workflows/quality-gate.yml`](../.github/workflows/quality-gate.yml) |
| **Release** | A model is "released" by (a) being tracked with full provenance, (b) being registered with alias `@champion`, (c) moving the gate baseline only through the protected baseline-update rule. | MLflow runs + Model Registry ([`register.py`](../src/credit_risk/register.py)), [`baseline/baseline_metrics.json`](../baseline/baseline_metrics.json), [`scripts/update_baseline.py`](../scripts/update_baseline.py), [`CODEOWNERS`](../.github/CODEOWNERS) |
| **Deploy** | Batch scoring with the registered pipeline (identical transformations); the threshold travels with the run. *No online endpoint* (see below). | [`src/credit_risk/predict.py`](../src/credit_risk/predict.py) (`models:/credit-risk-classifier@champion`) |
| **Operate** | Monthly retraining = a data-version change entering through a PR and the same gate; demonstrated by the v1 → v2 drill. | PR "data: dataset v2", [`scripts/data_drill.py`](../scripts/data_drill.py), [`docs/data_change_drill.md`](data_change_drill.md) |
| **Monitor** | Offline monitoring at each data version: input drift (KS/PSI) of new customers, subgroup error by `Income_Bracket`, seed-variance of the metric. | [`reports/drill/drift_new_customers.csv`](../reports/drill/drift_new_customers.csv), [`scripts/fairness_analysis.py`](../scripts/fairness_analysis.py), [`reports/fairness/`](../reports/fairness) |

### ML-specific lifecycle steps (beyond classic DevOps)

| MLOps step | Where |
|---|---|
| Data versioning & validation | DVC + `data/versions.yaml`; `credit_risk.data.validate` (schema, duplicates, ranges, categories, missingness) |
| Feature engineering as code | `CreditFeatureBuilder`, fitted inside the sklearn `Pipeline` |
| Experiment tracking | MLflow: params, metrics, tags `git_commit`, `data_md5`, `data_version`, `config_sha256`, resolved config artefact, model artefact |
| Model evaluation / validation | CV selection on train only; held-out test; gate rule; noise study |
| Model registry | MLflow Model Registry, alias `@champion`, lineage tags |
| Continuous training | Data PR → same workflow → drill → baseline update rule |

## Where ML delivery differs from ordinary software delivery

1. **Behaviour depends on data, not only code.** The same commit gives a different model when the data changes, so the data must be versioned and tied to the commit (DVC pointer in Git), and the dataset id is logged with every run. In ordinary software, the commit alone determines behaviour.
2. **"Tests pass" does not mean "the change is fine".** A change can pass every unit test and still make the model worse: dropping the payment-delay features passed all tests but lost 0.093 PR-AUC. So CI needs a *statistical* acceptance test (retrain + compare to a baseline) in addition to deterministic tests.
3. **Results are noisy, so acceptance needs tolerances that are justified.** Changing only the random seed moved test PR-AUC by sd 0.025 ([gate analysis](../reports/gate/gate_rule_comparison.md)). An exact-equality "test" is either useless or always red, so the gate threshold has to be derived from measured variation (Task 8).
4. **The artefact is trained, not compiled.** The release unit is a fitted pipeline plus its decision threshold, produced by a search over hyperparameters. It needs an experiment log and a registry with lineage (commit + data + config), not just a build number.
5. **Quality decays without code changes.** Customer populations drift (the v2 newcomers are older: KS p = 0.007), so the loop needs retraining and monitoring stages. Fairness across subgroups is also a release criterion with no equivalent in typical software.

## Stages intentionally left out (and why)

| Left out | Reason |
|---|---|
| Online serving endpoint (REST/containers, canary, rollback) | The bank retrains monthly and scores in batch; the brief scopes delivery to a gated, traceable model. `predict.py` covers batch deployment. Adding a server would add cost and infrastructure without serving the risk team's question. |
| Live production monitoring (dashboards, alerting on live predictions and outcomes) | No production traffic or delayed default labels exist in a course project. Monitoring is done offline per data version (drift + subgroup reports); the same scripts would run on a schedule in production. |
| Automated retraining trigger / scheduler | Retraining is triggered by a data-version PR so that a human reviews the gate summary. A cron-triggered workflow is a one-line addition (`on: schedule`) once a data feed exists. |
| Cloud infrastructure (SageMaker, S3, CodePipeline) | Replaced with free tools as required by the brief: Git+DVC, MLflow, GitHub Actions. |
| Hyperparameter search at scale (Optuna, distributed) | 800–1,100 rows; full grids finish in under a minute. |
