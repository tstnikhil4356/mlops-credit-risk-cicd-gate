# Quality-gate defense memo

**To:** Risk team · **From:** MLOps engineering (Nikhil Singh B042, Harsh Jain B059) · **Re:** Which rule should block a monthly model change

**Recommendation.** Block a pull request unless **PR-AUC ≥ baseline − 0.02 AND recall at the tuned threshold ≥ 0.85**, with tests passing (rule **R8**, `configs/base.yaml → gate`).

**Evidence (our own 18 tracked gate-history runs; [full table](threshold_and_fairness.md)).** We replayed eight candidate rules against changes of known effect.

- **R8 is the only rule with zero false blocks (5/5 benign changes accepted) and zero false accepts (7/7 harmful changes rejected).**
- Accuracy (R1) accepted 5 of 7 harmful changes, because a model that misses more defaulters looks *more* accurate.
- Strict PR-AUC (R3) blocked 2 harmless refactors.
- Every ranking-only rule (R2–R5) accepted the two threshold regressions, where recall fell to 0.76 and 0.62 with PR-AUC exactly unchanged.
- A 0.70 recall floor (R7) still let recall fall from 0.97 to 0.76.

**Why these numbers.**

- **0.02 tolerance:** benign changes on the gate's fixed split moved PR-AUC by ≤ 0.0095, while the smallest real regression was 0.060. The tolerance sits 2× above jitter and 3× below harm.
- **Recall floor 0.85:** this is the weakest recall any benign change produced (0.94) minus a margin for the test set's granularity (one defaulter = 2.9 points). It also sits above every threshold regression we observed.
- **No compounding drift:** each PR is compared with the *fixed* baseline, not with the previous PR, so a chain of small drops cannot add up.

**Trade-offs we accept by choosing R8 over the alternatives.**

1. **A real loss of up to 0.02 PR-AUC can pass unnoticed** (R3 would catch it but blocks 40% of harmless changes). We prefer that engineers trust the gate over a gate that cries wolf.
2. **The recall floor is coarse and seed-sensitive.** On different splits (noise probes) recall ranged 0.83–0.97, and R8 accepted only 2 of 5 seed changes. A change that alters the evaluation split (a seed change, or a new data version) can be blocked by noise rather than harm. That is why data changes go through the multi-seed drill and a protected baseline update, not the single CI number.
3. **R8 does not price precision.** Recall can be bought with more false alarms (FPR 22% today). We accept this because the bank values catching defaulters; F2 (R6) priced it better but let a 0.06 PR-AUC regression through.
4. **Fairness is monitored, not gated.** The Income_Bracket gaps (FPR 5.2 points, FNR 6.5 points) are within noise and do not come from the income input, so we did not add a subgroup gate that would be mostly noise on 21 High-income defaulters.

**What new evidence would change this recommendation.**

- **Cost data from the risk team.** If the cost of a false alarm is close to that of a missed defaulter, we would move to an expected-cost or F2 rule.
- **A larger or pooled test set** (v2 and later months). A smaller seed sd would justify tightening the tolerance to 0.01 and replacing the fixed floor with a relative one (recall ≥ baseline − 0.05).
- **Benign changes that legitimately move PR-AUC by more than 0.02** (e.g. a new data source). We would then widen the tolerance or add a multi-seed paired check in CI.
- **A subgroup gap above 10 points with a CI excluding 0**, or regulatory guidance treating income as a protected attribute. We would then add a per-group FNR/FPR guard.
