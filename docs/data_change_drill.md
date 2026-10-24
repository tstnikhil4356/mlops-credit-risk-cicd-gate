# Task 7: Data-change drill (v1 → v2)

**Procedure.** Dataset v2 (all 800 v1 rows unchanged + 300 new customers) entered on branch `data/v2` as a DVC pointer change. It was retrained through **the same code and `configs/ci.yaml`** as the gate. [`scripts/data_drill.py`](../scripts/data_drill.py) materialises both versions from history (`dvc get . data/raw/credit_card_customers.csv --rev data-v1` and `--rev <v2 PR head>`) and trains each **with 5 seeds** (42, 1, 2, 3, 4). Every one of the 10 runs is tracked in MLflow (`credit-risk-ci`, `run_label = drill/v1 | drill/v2`) with tags `data_version`, `data_md5` and `drill_rev`. Full output: [`reports/drill/drill_results.md`](../reports/drill/drill_results.md). The drill was re-run on the v2 PR branch after PR #1 widened the CI grid, so it uses exactly the configuration the gate runs; seed-42 results are identical to the first pass and the conclusion is unchanged.

Because the split is a hash of (seed, `Customer_ID`), every v1 customer is on the same side of the split in both versions. For each seed we can compare:

- **own split**: each model on its own test split (what the CI gate sees);
- **common rows**: both models on the *identical* v1 test customers (a paired, fair comparison);
- **new rows**: both models on test customers that exist only in v2.

![v1 vs v2 per seed](../reports/figures/drill_v1_v2.png)

| Comparison (v2 − v1, PR-AUC) | Mean | SD across seeds | Seeds v2 better | Paired t p |
|---|---|---|---|---|
| Own split (what CI sees) | +0.0068 | 0.0280 | 3/5 | 0.62 |
| **Common v1 test rows (paired)** | **+0.0002** | 0.0144 | 2/5 | 0.97 |
| New-customer test rows | +0.0108 | 0.0079 | 5/5 | 0.04 |
| Recall, own split | −0.0438 | 0.0544 | 0/5 | 0.15 |

Paired bootstrap of the PR-AUC difference on common rows: every seed's 95% interval contains 0 (for example seed 42: +0.012 [−0.012, +0.043]).

**Drift check.** The new customers match v1 on every financial feature (KS p ≥ 0.47, PSI ≤ 0.06) and on the default rate (19.3% vs 18.9%). They are slightly **older** (mean age 46.2 vs 43.9, KS p = 0.007, PSI 0.095, below the usual 0.1 "investigate" line).

## Conclusion

1. **The CI run alone would overstate the gain.** With the gate's fixed seed (42), v2 reaches PR-AUC 0.8997 against the baseline's 0.8762 (+0.0235), and the gate passes it. But across seeds the own-split change is +0.007 ± 0.028, well within the 0.025 seed-to-seed sd measured in Task 8. On the identical customers the two models are indistinguishable (+0.0002, p = 0.97). The apparent +0.0235 comes mostly from **a different test set**: the new customers are easier to rank (the v1 model already scores 0.93 PR-AUC on them at seed 42).
2. **The only consistent effect is small and expected.** v2 is slightly better on the new customers (+0.011, 5/5 seeds), since it has seen customers like them.
3. **Recall is a watch item.** Mean recall at the tuned threshold is 0.044 lower with v2 (not significant, p = 0.15). Two seeds fall below the 0.85 floor (0.833, 0.797), although the gate's own seed gives 0.965. The F2-tuned threshold moves with the data, and with 34–64 test defaulters per split, one customer is worth 1.6–2.9 points of recall.

**Should the gate accept the v2 model? Yes, as "not worse", not as "better".** It is statistically equivalent on the customers both versions know, at least as good on the new segment, trained on 37.5% more data, and representative of the current (slightly older) population. We accept it through the gate and then move the baseline with a separate `baseline-update` PR whose reason cites this drill. The new baseline is *not* recorded as a +0.0235 improvement. Follow-up: watch recall on the next monthly data; if it falls below 0.85 on the gate seed, recalibrate the threshold using pooled out-of-fold scores.
