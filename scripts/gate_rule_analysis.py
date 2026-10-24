"""Task 8: compare candidate quality-gate rules against the tracked run history.

Reads the ``gate-history/*`` runs from MLflow (each labelled benign/harmful),
replays every candidate rule against the ``gate-history/reference`` run as
baseline, and reports what each rule would have accepted or rejected.

Runs are split into
  * benign, same split  - harmless code/config changes evaluated on the CI split
                          (what the gate actually sees on a pull request)
  * noise probe         - seed changes; the test split itself moves, so these
                          measure run-to-run variation rather than a code change
  * harmful             - changes known to degrade the model

Output: reports/gate/gate_rule_comparison.{csv,md}, reports/gate/gate_decisions.csv,
        reports/figures/gate_rules.png
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from credit_risk.config import load_config
from credit_risk.gate import GateRule

OUT = Path("reports/gate")
FIG = Path("reports/figures")

METRICS = ["average_precision", "roc_auc", "recall", "precision", "f2", "accuracy", "balanced_accuracy"]


def candidate_rules(baseline: dict) -> list[tuple[str, GateRule]]:
    return [
        ("R1 accuracy >= baseline", GateRule("accuracy", 0.0)),
        ("R2 ROC-AUC >= baseline - 0.01", GateRule("roc_auc", 0.01)),
        ("R3 PR-AUC >= baseline (strict)", GateRule("average_precision", 0.0)),
        ("R4 PR-AUC >= baseline - 0.02", GateRule("average_precision", 0.02)),
        ("R5 PR-AUC >= baseline - 0.05", GateRule("average_precision", 0.05)),
        ("R6 F2 >= baseline - 0.02", GateRule("f2", 0.02)),
        ("R7 PR-AUC >= b - 0.02 AND recall >= 0.70", GateRule("average_precision", 0.02, "recall", 0.70)),
        ("R8 PR-AUC >= b - 0.02 AND recall >= 0.85", GateRule("average_precision", 0.02, "recall", 0.85)),
    ]


def load_history(cfg) -> pd.DataFrame:
    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(cfg.tracking.experiment_name)
    runs = client.search_runs([exp.experiment_id], "tags.run_label LIKE 'gate-history/%'", max_results=1000,
                              order_by=["attributes.start_time DESC"])
    rows, seen = [], set()
    for r in runs:  # newest run per label wins
        label = r.data.tags["run_label"].removeprefix("gate-history/")
        if label in seen:
            continue
        seen.add(label)
        rows.append({
            "label": label,
            "run_id": r.info.run_id,
            "change_kind": r.data.tags.get("change_kind"),
            "description": r.data.tags.get("change_description"),
            "seed": r.data.tags.get("seed"),
            **{m: r.data.metrics.get(f"test_{m}") for m in METRICS},
        })
    df = pd.DataFrame(rows)
    df["group"] = np.where(df["label"].str.startswith("seed-"), "noise probe",
                           np.where(df["change_kind"] == "benign", "benign, same split", "harmful"))
    return df.sort_values(["group", "label"]).reset_index(drop=True)


def main() -> int:
    cfg = load_config("configs/ci.yaml")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    hist = load_history(cfg)
    ref = hist.loc[hist["label"] == "reference"].iloc[0]
    baseline = {m: ref[m] for m in METRICS}
    others = hist[hist["label"] != "reference"].copy()
    for m in METRICS:
        others[f"delta_{m}"] = others[m] - baseline[m]

    decisions = others[["label", "group", "description", "run_id", *METRICS]].copy()
    summary = []
    for name, rule in candidate_rules(baseline):
        acc = others.apply(lambda r: rule.evaluate({m: r[m] for m in METRICS}, baseline).passed, axis=1)
        decisions[name] = np.where(acc, "accept", "REJECT")
        benign = others["group"] == "benign, same split"
        noise = others["group"] == "noise probe"
        harmful = others["group"] == "harmful"
        summary.append({
            "rule": name,
            "benign accepted": f"{int(acc[benign].sum())}/{int(benign.sum())}",
            "noise probes accepted": f"{int(acc[noise].sum())}/{int(noise.sum())}",
            "harmful rejected": f"{int((~acc[harmful]).sum())}/{int(harmful.sum())}",
            "false blocks (benign)": int((~acc[benign]).sum()),
            "false accepts (harmful)": int(acc[harmful].sum()),
            "harmful accepted": ", ".join(others.loc[harmful & acc, "label"]) or "-",
            "benign blocked": ", ".join(others.loc[benign & ~acc, "label"]) or "-",
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "gate_rule_comparison.csv", index=False)
    decisions.to_csv(OUT / "gate_decisions.csv", index=False)

    noise = others[others["group"] == "noise probe"]
    benign = others[others["group"] == "benign, same split"]
    harmful = others[others["group"] == "harmful"]
    stats = {
        "baseline_run_id": ref["run_id"],
        "baseline": baseline,
        "seed_ap_values": hist.loc[hist["label"].isin(["reference", *noise["label"]]), "average_precision"].tolist(),
        "seed_ap_std": float(hist.loc[hist["label"].isin(["reference", *noise["label"]]), "average_precision"].std()),
        "benign_paired_max_abs_ap_delta": float(benign["delta_average_precision"].abs().max()),
        "harmful_min_ap_drop": float(-harmful.loc[harmful["delta_average_precision"].abs() > 1e-9,
                                                 "delta_average_precision"].max()),
        "harmful_ap_unchanged": harmful.loc[harmful["delta_average_precision"].abs() < 1e-9, "label"].tolist(),
    }
    (OUT / "gate_noise_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    def md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
        return "\n".join(lines)

    hist_view = others[["label", "group", "average_precision", "delta_average_precision", "recall", "f2",
                        "accuracy", "run_id"]].copy()
    hist_view["run_id"] = hist_view["run_id"].str[:8]
    md = [
        "# Gate rule comparison (generated by scripts/gate_rule_analysis.py)",
        "",
        f"Baseline = run `{ref['run_id'][:8]}` (gate-history/reference): AP={baseline['average_precision']:.4f}, "
        f"recall={baseline['recall']:.4f}, F2={baseline['f2']:.4f}, accuracy={baseline['accuracy']:.4f}",
        "",
        "## Run history replayed",
        "",
        md_table(hist_view),
        "",
        "## What each rule would have done",
        "",
        md_table(summary_df),
        "",
        "## Noise statistics",
        "",
        f"- AP across 6 seeds (split changes): std = {stats['seed_ap_std']:.4f}",
        f"- Largest |AP delta| of a benign change on the same split: {stats['benign_paired_max_abs_ap_delta']:.4f}",
        f"- Smallest AP drop among harmful changes that move AP: {stats['harmful_min_ap_drop']:.4f}",
        f"- Harmful changes with unchanged AP (ranking-preserving): {', '.join(stats['harmful_ap_unchanged'])}",
    ]
    (OUT / "gate_rule_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"benign, same split": "#2a9d8f", "noise probe": "#8d99ae", "harmful": "#e76f51"}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for g, sub in others.groupby("group"):
        ax.scatter(sub["delta_average_precision"], sub["recall"], s=60, c=colors[g], label=g, edgecolor="white")
        for _, r in sub.iterrows():
            ax.annotate(r["label"], (r["delta_average_precision"], r["recall"]), fontsize=7, xytext=(4, 3),
                        textcoords="offset points", color="#333")
    ax.axvline(-0.02, color="#264653", ls="--", lw=1)
    ax.axhline(0.85, color="#264653", ls=":", lw=1)
    ax.axhline(0.70, color="#999", ls=":", lw=1)
    ax.text(-0.019, 0.56, "PR-AUC tolerance -0.02", fontsize=7, color="#264653")
    ax.text(-0.30, 0.855, "recall floor 0.85 (R8)", fontsize=7, color="#264653")
    ax.text(-0.30, 0.705, "recall floor 0.70 (R7)", fontsize=7, color="#777")
    ax.set_xlabel("PR-AUC change vs baseline (test split)")
    ax.set_ylabel("Recall at tuned threshold")
    ax.set_title("Gate run history: each point is a tracked run")
    ax.legend(fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "gate_rules.png", dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
