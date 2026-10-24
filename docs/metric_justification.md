# Task 5: Why plain accuracy is not sufficient (metric justification, one page)

**The data.** Only 18.9% of customers are high risk (151 of 800). On the held-out test split, 34 of 186 customers are defaulters.

**Accuracy rewards ignoring the problem.** A model that labels *every* customer "not high risk" scores **81.7% accuracy** on our test split while catching **zero** defaulters. Our registered model scores **81.2%** accuracy, which is *lower*, even though it catches **33 of 34** defaulters. In the gate run history, moving the decision threshold from 0.13 to 0.7 *raised* accuracy from 0.812 to 0.909 while recall collapsed from 0.97 to 0.62 ([gate history](../reports/gate/gate_rule_comparison.md)). An accuracy-based gate (rule R1) accepted 5 of 7 known-harmful changes. Accuracy is dominated by the majority class and treats a missed defaulter (a credit loss of up to the full limit) the same as a false alarm (a manual review).

**What we use instead.**

| Role | Metric | Why |
|---|---|---|
| **Model selection & primary gate metric** | **PR-AUC** (average precision), cross-validated on the training split | Measures how well the model *ranks* defaulters above non-defaulters across all thresholds, focused on the positive (minority) class. Its no-skill value equals the prevalence (≈0.19, matching our dummy model's 0.19), so the gain is easy to read. Unlike ROC-AUC, it is not inflated by the large number of easy true negatives. It is also threshold-free, so it is stable for selecting between model families. |
| **Operating point** | **Recall** at a threshold chosen to maximise **F2** on out-of-fold training scores | The bank's costs are asymmetric: missing a defaulter costs much more than reviewing a good customer. F2 weights recall four times as heavily as precision in its harmonic mean. Recall is the number the risk team acts on: the share of defaulters caught. |
| Reported for context | ROC-AUC, precision, F2, balanced accuracy, Brier score, FPR/FNR | Precision and FPR show the cost of the high recall (34 false flags out of 152 good customers). Brier shows calibration. Balanced accuracy averages the per-class recalls. |
| Not used for decisions | Accuracy | Reported only to demonstrate the point above. |

**Model comparison** (dataset v1, seed 42; [all runs](../reports/tracking/model_comparison.md))

| Model (tracked run) | CV PR-AUC ± sd | Test PR-AUC | Test ROC-AUC | Test recall | Test precision | Test accuracy |
|---|---|---|---|---|---|---|
| Dummy (prior) baseline | 0.191 ± 0.004 | 0.183 | 0.500 | 1.000* | 0.183 | 0.183* |
| Logistic regression, default C=1 | 0.866 ± 0.036 | 0.884 | 0.970 | 0.971 | 0.635 | 0.892 |
| **Logistic regression, tuned (C=3, no class weight) - selected** | **0.873 ± 0.040** | 0.876 | 0.969 | 0.971 | 0.493 | 0.812 |
| Random forest, tuned (48 settings) | 0.852 ± 0.053 | 0.788 | 0.947 | 0.824 | 0.596 | 0.866 |
| Hist. gradient boosting, tuned (48 settings) | 0.852 ± 0.051 | 0.781 | 0.934 | 0.853 | 0.580 | 0.860 |

\*The F2-tuned threshold makes the constant dummy flag everyone.

**Selection.** We choose by **CV PR-AUC on the training split**, never by the test split. Picking the test winner (default LR, 0.884) would be selection on the test set. The tuned and default logistic regressions are within one CV sd of each other. The tree ensembles are clearly worse out of sample: with 600 training rows and largely monotonic risk drivers, the regularised linear model with engineered ratios generalises best, and it is interpretable for the risk team. The selected run is registered as `credit-risk-classifier` version 1, alias `@champion`, with tags tracing it to its git commit, data md5/version and config hash ([`reports/registry/registered_model.json`](../reports/registry/registered_model.json)).

**Caveat stated up front.** With only 34 test defaulters, one customer is worth 2.9 points of recall. A seed change alone moves test PR-AUC by sd 0.025. Differences smaller than that are not evidence (Tasks 7 and 8 build on this).
