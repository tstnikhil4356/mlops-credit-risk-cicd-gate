# Task 4: Cleaning and feature engineering

All logic lives in [`CreditFeatureBuilder`](../src/credit_risk/features.py), the first step of the scikit-learn `Pipeline`, followed by the encoder ([`build_encoder`](../src/credit_risk/features.py)) and the classifier. The fitted pipeline is what MLflow logs and the registry serves. CV folds, the final fit, the CI gate and `predict.py` therefore all apply identical transformation logic, and imputation statistics are learned from training rows only. Every threshold below comes from the `features:` section of the YAML config.

## Cleaning decisions

| Issue (found in the data) | Decision | Justification |
|---|---|---|
| `Credit_Limit`, `Current_Balance`, `Payment_Delay_Days` missing for 24 of 800 rows (33 of 1,100 in v2) | Median imputation learned in `fit` (config `impute_strategy`), plus the flag `record_incomplete` | All three are missing **in the same rows**, so this is record-level incompleteness, not three independent gaps. Those rows have the normal default rate (18.2% vs 18.9%), so dropping them loses data without benefit. The median is robust to the right-skewed money columns. The flag lets the model discount imputed values (coefficient +0.86 in the registered model). Mean imputation was tested as a benign change: AP moved by only −0.0006. |
| `Customer_ID` | Never a feature; the builder ignores it. It is used only for the deterministic hash split | It is an identifier, not a property of the customer. A test proves predictions do not change when the ID changes. |
| Target `High_Risk_Customer` | The builder raises an error if the target column reaches it | Guards against target leakage by construction (tested). |
| Invalid values (negative, out-of-range, unknown category, duplicate IDs, missingness above 10%) | `credit_risk.data.validate` stops the pipeline with a message listing every problem | These are data errors, not modelling problems; silently "fixing" them would hide upstream faults. |
| Division edge cases (zero/negative limit, zero transactions) | `safe_ratio`: limit ≤ 0 with a positive balance → cap (3.0); limit ≤ 0 with zero balance → 0; ratios clipped to [0, 3]; transactions clipped at ≥ 1 | Owing money with no available limit is maximal utilisation risk, and infinities would break linear models. No zero limits exist in v1/v2, but new monthly data can contain them, so the behaviour is defined and tested. |

## Engineered features

| # | Name | Kind | Formula | Reason (domain) |
|---|---|---|---|---|
| 1 | `utilisation` | **ratio** | `safe_ratio(Current_Balance, Credit_Limit)` | The credit-utilisation ratio is the strongest bureau-style default predictor: a customer near their limit has no buffer. It has the largest coefficient in the model (+4.25 standardised). |
| 2 | `spend_to_limit` | ratio | `safe_ratio(Monthly_Spend, Credit_Limit)` | Monthly "burn rate" relative to the limit: how fast the customer would reach the limit. |
| 3 | `high_utilisation_flag` | **binary flag** | `utilisation ≥ features.utilisation_risk_cutoff` (0.80) | Mirrors the bank's "over 80% utilised" policy rule, which the risk team can audit. The cut-off is configurable (the 0.75 variant was tested). |
| 4 | `late_payer_flag` | binary flag | `Payment_Delay_Days > features.payment_delay_flag_days` (7) | More than a week late on average marks the start of delinquency in collections practice. |
| 5 | `age_band` | **binned / categorical** | `cut(Age, [0,30,45,60,120))` → `under_30`, `30_44`, `45_59`, `60_plus`, one-hot encoded | Risk is not linear in age: the registered model gives `under_30` −2.19 and `60_plus` +1.63 on top of linear `Age`, a U/J-shaped effect that one linear term cannot express. |
| 6 | `credit_stress_index` | **own design** | `utilisation × log(1 + Payment_Delay_Days)` | An interaction: risk compounds when a customer is both stretched (high utilisation) **and** already paying late; either alone is far less alarming. The log damps extreme delays. See the honest ablation result below. |
| 7 | `spend_per_transaction` | ratio | `Monthly_Spend / max(Num_Transactions_Monthly, 1)` | Few, large transactions (cash-like, big-ticket) behave differently from many small everyday ones. |
| 8 | `record_incomplete` | flag | any of the raw numeric columns missing | See cleaning decisions. |

### Encoding

| Column | Encoding | Why |
|---|---|---|
| `Income_Bracket` | Ordinal `Low=0 < Medium=1 < High=2` (categories fixed in config; unknown → −1) | Naturally ordered; one column keeps the linear model simple. |
| `age_band` | One-hot (categories fixed in config, unknown ignored) | No meaningful order in the *effect*; we want separate coefficients. |
| Numeric + engineered | `StandardScaler` for logistic regression; passthrough for tree models | Regularised linear models need comparable scales; trees are scale-invariant. |

## Evidence that the features matter (tracked ablations, logistic regression, CV PR-AUC on train)

| Variant | CV PR-AUC | test PR-AUC | Conclusion |
|---|---|---|---|
| All features (selected model) | **0.8732** | 0.8762 | reference |
| Raw columns only (no engineered features) | 0.7946 | 0.8210 | engineered features add +0.079 CV PR-AUC |
| Without the ratio features | 0.8242 | 0.8260 | the ratios carry most of the lift |
| Without `record_incomplete` | 0.8625 | 0.8767 | small CV gain (+0.011), within fold noise |
| Without `credit_stress_index` (own feature) | 0.8752 | 0.8790 | **no measurable lift** (+0.002, far inside the fold-to-fold sd of ≈0.04) |

We report the last row honestly. The own feature is domain-motivated, but on this data a linear model already captures the additive utilisation and delay effects. Seven alternative domain features were screened by CV (balance-to-spend months, headroom, projected utilisation, delay per tenure, etc.), and none beat the others by more than ±0.005. We keep `credit_stress_index` because it is interpretable for the risk team and costs nothing. The ablation infrastructure is exactly what lets a future PR remove it with evidence (`features.drop_features` in YAML, no code change). Exact runs: [`reports/tracking/model_comparison.md`](../reports/tracking/model_comparison.md).

## Leakage controls

- The target is rejected by the feature builder; `Customer_ID` is ignored.
- The train/test split is made **before** any fitting; medians, scaler and encoder statistics are fitted inside the pipeline, so they are re-fitted inside every CV fold.
- Model selection and threshold tuning use cross-validation on the training split only; the test split is scored once per run.
- No feature uses information from the future or from other customers (each row is transformed independently; tested).

## Tests covering this logic ([`tests/test_features.py`](../tests/test_features.py))

Zero credit limit (with and without balance), negative limit, zero transactions, missing values imputed with *training* medians, clipping and no infinities, flag boundaries read from config, age-band edges, target rejection, `Customer_ID` invariance, missing required column, `drop_features`, config-driven behaviour change, row independence, and train vs pickled-pipeline parity (`tests/test_pipeline.py`).
