# References and citations

No third-party code was copied into this repository. The following official documentation and well-known methods informed the design; usage patterns adapted from documentation examples are noted.

| Topic | Source | Used for |
|---|---|---|
| DVC data versioning, `dvc add/push/pull/get --rev`, local remotes | DVC documentation, https://dvc.org/doc (Iterative.ai) | Data tracking (Task 2), drill materialisation (Task 7) |
| MLflow tracking, `mlflow.sklearn.log_model`, Model Registry aliases | MLflow 2.17 documentation, https://mlflow.org/docs/2.17.2/ | Run logging, registry, `@champion` alias (Task 5) |
| scikit-learn `Pipeline`, `ColumnTransformer`, custom transformers, `GridSearchCV`, `cross_val_predict` | scikit-learn 1.5 user guide, https://scikit-learn.org/1.5/ | Feature pipeline, tuning, OOF threshold (Tasks 4–5); custom transformer follows the "developing estimators" guide |
| Pydantic v2 models and `ValidationError` | https://docs.pydantic.dev/2.9/ | Config schema (Task 3) |
| GitHub Actions: `actions/checkout`, `setup-python` caching, `upload-artifact`, `$GITHUB_STEP_SUMMARY`, `gh pr comment` | https://docs.github.com/actions | Quality-gate workflow (Task 6) |
| Branch protection & CODEOWNERS | https://docs.github.com/repositories | Baseline protection (Task 6) |
| Precision-recall vs ROC under class imbalance | Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLoS ONE 10(3) | Metric justification (Task 5) |
| Population Stability Index thresholds (0.1 / 0.25) | Common credit-scoring practice; e.g. Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley | Drift check (Task 7) |
| Bootstrap confidence intervals | Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap.* Chapman & Hall | Paired bootstrap (Task 7), subgroup CIs (Task 8) |
| Fairness metrics (error-rate balance, FNR/FPR parity) | Hardt, M., Price, E., Srebro, N. (2016). *Equality of Opportunity in Supervised Learning.* NeurIPS | Subgroup analysis framing (Task 8) |
| MLOps lifecycle / ML-specific technical debt | Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS; Google Cloud, *MLOps: Continuous delivery and automation pipelines in ML* | Lifecycle map and ML-vs-software differences (Task 1) |

Tooling note: the brief suggests Google Colab. We substituted a local Python 3.11 virtual environment (free) because CI, DVC and Git hooks need a persistent checkout; everything still runs in Colab with `!make setup data reproduce`.
