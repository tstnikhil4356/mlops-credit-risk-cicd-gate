"""Draw the lifecycle map (Task 1) and the CI quality-gate flow (Task 6).

Output: reports/figures/lifecycle_map.png, reports/figures/ci_gate_flow.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path("reports/figures")

DONE, PART, OUT = "#2a9d8f", "#e9c46a", "#c9ccd1"

STAGES = [
    ("PLAN", "Scope, metric & gate\nrule; risk-team brief",
     "docs/metric_justification.md\ndocs/gate_defense_memo.md", DONE),
    ("CODE", "Features, training,\nconfig (YAML), data (DVC)",
     "src/credit_risk/*.py\nconfigs/*.yaml  data/*.dvc", DONE),
    ("BUILD", "Locked env, data pull,\nconfig validation",
     "requirements.lock\nscripts/dvc_storage.sh", DONE),
    ("TEST", "Unit tests + model test:\ntrain & compare to baseline",
     "tests/  credit_risk/gate.py\n.github/workflows/", DONE),
    ("RELEASE", "Experiment log, model\nregistry, baseline update",
     "MLflow runs + registry\nbaseline/  update_baseline.py", DONE),
    ("DEPLOY", "Batch scoring with the\n@champion model",
     "credit_risk/predict.py\n(no serving endpoint)", PART),
    ("OPERATE", "Monthly retrain through\nthe same gated workflow",
     "data-change drill\nscripts/data_drill.py", PART),
    ("MONITOR", "Drift & subgroup checks\n(offline, per data version)",
     "drift_new_customers.csv\nfairness_analysis.py", PART),
]

ML_BAND = [
    (0, 1, "DATA: versioned by DVC md5,\nschema-validated, ID-hash split"),
    (2, 2, "FEATURES fitted\ninside the pipeline"),
    (3, 3, "MODEL TEST:\nmetrics vs baseline"),
    (4, 5, "TRACK & REGISTER: params, metrics,\ndata version, config, model"),
    (6, 7, "CONTINUOUS TRAINING: new data\nretrained + re-evaluated (drill)"),
]


def lifecycle() -> None:
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.4)
    ax.axis("off")
    w, gap = 1.8, 0.2
    for i, (stage, activity, artefact, color) in enumerate(STAGES):
        x = 0.1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 6.6), w, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=color, ec="none"))
        ax.text(x + w / 2, 7.1, stage, ha="center", va="center", fontsize=11, weight="bold",
                color="white" if color == DONE else "#333")
        ax.add_patch(FancyBboxPatch((x, 4.9), w, 1.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc="white", ec=color, lw=1.6))
        ax.text(x + w / 2, 5.65, activity, ha="center", va="center", fontsize=8.2)
        ax.add_patch(FancyBboxPatch((x, 3.4), w, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc="#f4f5f7", ec="none"))
        ax.text(x + w / 2, 4.05, artefact, ha="center", va="center", fontsize=7, family="monospace",
                color="#264653")
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 7.1), (x + w + gap, 7.1), arrowstyle="-|>", mutation_scale=10,
                                         color="#555"))
    ax.add_patch(FancyArrowPatch((0.1 + 7 * (w + gap) + w / 2, 7.65), (0.1 + w / 2, 7.65),
                                 connectionstyle="arc3,rad=0.12", arrowstyle="-|>", mutation_scale=12,
                                 color="#888", lw=1.2))
    ax.text(8, 8.25, "feedback: monitoring findings & new data re-enter PLAN", ha="center", fontsize=8,
            color="#666", style="italic")
    ax.text(0.1, 6.45 - 1.62, "", fontsize=1)
    ax.text(-0.05, 5.65, "activity", rotation=90, va="center", ha="right", fontsize=8, color="#666")
    ax.text(-0.05, 4.05, "artefact", rotation=90, va="center", ha="right", fontsize=8, color="#666")

    ax.text(0.1, 2.95, "ML-specific additions to the DevOps loop", fontsize=9.5, weight="bold", color="#264653")
    for a, b, text in ML_BAND:
        x0 = 0.1 + a * (w + gap)
        x1 = 0.1 + b * (w + gap) + w
        ax.add_patch(FancyBboxPatch((x0, 1.55), x1 - x0, 1.15, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc="#e8f4f2", ec=DONE, lw=1))
        ax.text((x0 + x1) / 2, 2.12, text, ha="center", va="center", fontsize=7.8)
    for j, (c, lab) in enumerate([(DONE, "implemented end-to-end"), (PART, "implemented offline / batch only"),
                                  (OUT, "intentionally out of scope")]):
        ax.add_patch(FancyBboxPatch((0.1 + j * 3.6, 0.45), 0.35, 0.35, boxstyle="round,pad=0.01", fc=c, ec="none"))
        ax.text(0.55 + j * 3.6, 0.62, lab, va="center", fontsize=8)
    ax.text(10.9, 0.62, "Out of scope: online serving endpoint, live production\nmonitoring, automated retrain "
            "scheduler (docs/lifecycle.md)", va="center", fontsize=7.5, color="#666")
    fig.tight_layout()
    fig.savefig(FIG / "lifecycle_map.png", dpi=170)
    plt.close(fig)


def ci_flow() -> None:
    steps = [
        ("Pull request\nopened / updated", "#264653"),
        ("Checkout code +\nDVC storage", "#2a9d8f"),
        ("Install locked\nenvironment", "#2a9d8f"),
        ("dvc pull\n(data of this commit)", "#2a9d8f"),
        ("Validate all\nYAML configs", "#2a9d8f"),
        ("pytest\n(58 tests)", "#2a9d8f"),
        ("Train with\nconfigs/ci.yaml", "#2a9d8f"),
        ("Gate:\nPR-AUC >= base-0.02\nAND recall >= 0.85", "#e76f51"),
        ("Summary:\nstep summary\n+ PR comment", "#e9c46a"),
        ("Required check\npass / BLOCK merge", "#264653"),
    ]
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    w, gap = 1.3, 0.17
    for i, (text, color) in enumerate(steps):
        x = 0.05 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.4), w, 1.2, boxstyle="round,pad=0.02,rounding_size=0.1", fc=color,
                                    ec="none"))
        ax.text(x + w / 2, 2.0, text, ha="center", va="center", fontsize=7.0,
                color="#222" if color == "#e9c46a" else "white", weight="bold")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 2.0), (x + w + gap, 2.0), arrowstyle="-|>", mutation_scale=9,
                                         color="#555"))
    ax.text(0.05, 0.95, "Parallel job baseline-guard: if baseline/baseline_metrics.json changed -> require label "
            "'baseline-update' + CODEOWNER review, retrain and verify the new baseline reproduces exactly (1e-6).",
            fontsize=8, color="#333")
    ax.text(0.05, 0.5, "Evidence uploaded per run: JUnit XML, metrics JSON, gate decision JSON, MLflow store.",
            fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(FIG / "ci_gate_flow.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    lifecycle()
    ci_flow()
    print("written:", FIG / "lifecycle_map.png", FIG / "ci_gate_flow.png")
