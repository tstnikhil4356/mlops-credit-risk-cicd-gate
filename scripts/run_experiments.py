"""Tracked experiment campaign (Task 5) and gate run history (Task 8).

Every run below is logged to MLflow with parameters, metrics, data version,
resolved config and model artefact. Groups (tag ``run_label``):

  selection           full.yaml, all model families (baseline -> tuned)
  ablation/*          feature ablations of the selected family
  gate-history/*      CI-config runs simulating pull requests of known kind
                      (tag ``change_kind`` = benign | harmful), used to compare
                      candidate gate rules in scripts/gate_rule_analysis.py

Usage:  python scripts/run_experiments.py [--only selection|ablation|gate-history]
Output: reports/experiments/campaign_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from credit_risk.config import load_config
from credit_risk.features import ENGINEERED
from credit_risk.train import run_training

OUT = Path("reports/experiments")

# (label, overrides, change_kind, description)
GATE_HISTORY = [
    ("reference", {}, "benign", "CI config as committed (baseline reproduction)"),
    ("seed-1", {"seed": 1}, "benign", "Different random seed: split + CV folds"),
    ("seed-2", {"seed": 2}, "benign", "Different random seed"),
    ("seed-3", {"seed": 3}, "benign", "Different random seed"),
    ("seed-4", {"seed": 4}, "benign", "Different random seed"),
    ("seed-5", {"seed": 5}, "benign", "Different random seed"),
    ("cutoff-0.75", {"features.utilisation_risk_cutoff": 0.75}, "benign", "Utilisation flag cut-off 0.80 -> 0.75"),
    ("delay-flag-5", {"features.payment_delay_flag_days": 5}, "benign", "Late-payer flag 7 -> 5 days"),
    ("impute-mean", {"features.impute_strategy": "mean"}, "benign", "Median -> mean imputation"),
    ("cv-10", {"training.cv_folds": 10}, "benign", "10-fold instead of 5-fold CV"),
    ("drop-tenure", {"features.drop_features": ["Tenure_Months"]}, "benign", "Remove a near-useless raw column"),
    ("drop-delay-signal", {"features.drop_features": ["Payment_Delay_Days", "late_payer_flag",
                                                     "credit_stress_index"]},
     "harmful", "Pipeline bug drops every payment-delay feature"),
    ("drop-balance-signal", {"features.drop_features": ["Current_Balance", "utilisation",
                                                       "high_utilisation_flag", "credit_stress_index"]},
     "harmful", "Balance column lost upstream"),
    ("over-regularised", {"training.candidates": [{"name": "logreg_ci", "family": "logistic_regression",
                                                   "params": {"max_iter": 2000, "C": 0.0005}}]},
     "harmful", "Hyper-parameter typo: C=0.0005"),
    ("threshold-0.5", {"training.threshold_strategy": "fixed", "training.fixed_threshold": 0.5},
     "harmful", "Decision threshold reset to 0.5 (ranking unchanged, recall falls)"),
    ("threshold-0.7", {"training.threshold_strategy": "fixed", "training.fixed_threshold": 0.7},
     "harmful", "Decision threshold 0.7 (ranking unchanged, recall collapses)"),
    ("raw-features-only", {"features.drop_features": [f for f in ENGINEERED if f != "record_incomplete"]},
     "harmful", "Feature engineering step accidentally disabled"),
    ("shallow-tree-model", {"training.candidates": [{"name": "tree_ci", "family": "random_forest",
                                                     "params": {"n_estimators": 10, "max_depth": 2}}]},
     "harmful", "Model swapped for a tiny random forest"),
]

ABLATIONS = [
    ("ablation/raw-features-only", {"features.drop_features": [f for f in ENGINEERED if f != "record_incomplete"]}),
    ("ablation/no-own-feature", {"features.drop_features": ["credit_stress_index"]}),
    ("ablation/no-missing-indicator", {"features.add_missing_indicator": False}),
    ("ablation/no-ratios", {"features.drop_features": ["utilisation", "spend_to_limit", "credit_stress_index"]}),
]


def _record(summary: dict, **extra) -> dict:
    return {**extra, **{k: v for k, v in summary.items() if not k.startswith("_")}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["selection", "ablation", "gate-history"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records: dict[str, list] = {"selection": [], "ablation": [], "gate_history": []}

    full = load_config("configs/full.yaml")
    ci = load_config("configs/ci.yaml")

    if args.only in (None, "selection"):
        print("== selection (full.yaml, all families)")
        records["selection"].append(_record(run_training(full, run_label="selection")))

    if args.only in (None, "ablation"):
        selected = [c.model_dump() for c in full.training.candidates if c.name == "logreg_tuned"]
        for label, overrides in ABLATIONS:
            print(f"== {label}")
            cfg = full.with_overrides(**{"training.candidates": selected, **overrides})
            records["ablation"].append(_record(run_training(cfg, run_label=label), label=label))

    if args.only in (None, "gate-history"):
        for label, overrides, kind, desc in GATE_HISTORY:
            print(f"== gate-history/{label} [{kind}]")
            seed = overrides.get("seed")
            cfg = ci.with_overrides(**{k: v for k, v in overrides.items() if k != "seed"}) if overrides else ci
            s = run_training(cfg, seed=seed, run_label=f"gate-history/{label}",
                             extra_tags={"change_kind": kind, "change_description": desc})
            records["gate_history"].append(_record(s, label=label, change_kind=kind, description=desc))

    path = OUT / (f"campaign_{args.only}.json" if args.only else "campaign_summary.json")
    path.write_text(json.dumps(records, indent=2, default=str) + "\n")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
