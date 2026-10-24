# 5-minute demonstration script

Setup before recording: `make setup data`, `make reproduce` done; MLflow UI open (`make ui`); browser tabs open on the repository's *Pull requests* and *Actions* pages; a terminal in the repo root.

| Time | Show | Say (key points) | Command / screen |
|---|---|---|---|
| 0:00–0:30 | README "Results at a glance" + lifecycle figure | The problem: an untested change lowered quality and nobody knew which code/data/config made it. Our answer: version everything, test automatically, track, gate, and defend the gate. | `README.md`, `reports/figures/lifecycle_map.png` |
| 0:30–1:15 | Versioning & restore (Task 2) | Data is a DVC pointer in Git, not a CSV. Restore an old commit with its exact data and get the identical number. | `cat data/raw/credit_card_customers.csv.dvc`; `git log --oneline -8`; `make restore-demo` (≈30 s: "recorded AP=0.876207 / reproduced AP=0.876207") |
| 1:15–1:45 | Config (Task 3) | No hard-coded values; environments extend base; a broken config fails immediately. | `python -m credit_risk.train --config tests/fixtures/broken_bad_values.yaml` (4 clear errors); `--set features.utilisation_risk_cutoff=0.75` |
| 1:45–2:30 | Features & tracking (Tasks 4–5) | Feature table; pipeline shared by training and inference; PR-AUC not accuracy (dummy has 81.7% accuracy and catches nobody). 49 tracked runs; registered model traced to commit, data and config. | `docs/features.md`; MLflow UI: experiment list, compare runs, model `credit-risk-classifier` v1 tags |
| 2:30–3:45 | CI gate (Task 6) | Three PRs: pass, blocked because the model is worse (−0.09 PR-AUC), blocked because a test fails. The summary is readable without logs. Baseline only moves via a labelled, reproducible PR. | GitHub PR list → open each PR's gate comment; *Actions* run summary; `baseline/baseline_metrics.json` |
| 3:45–4:30 | Drill (Task 7) | v2 looks +0.024 better in CI, but over 5 seeds on identical customers it is +0.0002 ± 0.014: not better, not worse → accept, and do not claim an improvement. | `reports/figures/drill_v1_v2.png`, `docs/data_change_drill.md` |
| 4:30–5:00 | Gate & fairness defense (Task 8) | 8 rules vs our history: only R8 has 0 false blocks and 0 false accepts. Trade-off: up to 0.02 PR-AUC loss can pass; the recall floor is coarse. Income-group gaps are within noise and not caused by the income feature. What would change our mind. | `reports/figures/gate_rules.png`, `docs/gate_defense_memo.md` |

Backup if the network fails: `make gate` locally shows the same summary; PR screenshots are in `reports/ci/screenshots/`.
