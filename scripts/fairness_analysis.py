"""Task 8: error breakdown of the registered model by Income_Bracket.

Two views, both with bootstrap 95% intervals:
  1. registered model (models:/<name>@champion) on its held-out test split
  2. 5-fold out-of-fold scores over the whole dataset with the same pipeline and
     hyper-parameters (more rows per group -> tighter intervals)

Both use the decision threshold stored with the registered run.
Output: reports/fairness/subgroup_{test,oof}.csv, subgroup_breakdown.md, figures/subgroup_error.png
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from credit_risk.config import load_config
from credit_risk.data import file_md5, load_raw, train_test_split_by_id
from credit_risk.evaluate import subgroup_metrics

OUT = Path("reports/fairness")
FIG = Path("reports/figures")
COLS = ["group", "n", "positives", "base_rate", "flagged_rate", "error_rate", "false_negative_rate",
        "false_negative_rate_ci_low", "false_negative_rate_ci_high", "false_positive_rate",
        "false_positive_rate_ci_low", "false_positive_rate_ci_high", "recall", "precision", "average_precision"]


def _parse(value: str):
    import yaml

    return yaml.safe_load(value) if value not in ("None",) else None


def main() -> int:
    cfg = load_config("configs/full.yaml")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    client = mlflow.MlflowClient()
    name = cfg.tracking.registered_model_name
    mv = client.get_model_version_by_alias(name, "champion")
    run = client.get_run(mv.run_id)
    seed = int(run.data.tags["seed"])
    threshold = float(run.data.metrics["test_threshold"])
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")

    df = load_raw(cfg)
    if run.data.tags["data_md5"] != file_md5(cfg.paths.raw_data):
        raise SystemExit("Checked-out data is not the data the registered model was trained on: "
                         "`git checkout <commit> && dvc checkout` first")
    X_tr, X_te, y_tr, y_te = train_test_split_by_id(df, cfg, seed)

    test_scores = model.predict_proba(X_te)[:, 1]
    test = subgroup_metrics(X_te["Income_Bracket"], y_te, test_scores, threshold, n_boot=2000, seed=0)

    X_all = df.drop(columns=[cfg.data.target])
    y_all = df[cfg.data.target].astype(int)
    oof_scores = cross_val_predict(clone(model), X_all, y_all, method="predict_proba",
                                   cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
    oof = subgroup_metrics(X_all["Income_Bracket"], y_all, oof_scores, threshold, n_boot=2000, seed=0)

    # Counterfactual: same model family and hyper-parameters without Income_Bracket as an input.
    from credit_risk.models import build_pipeline

    params = {k.removeprefix("model."): _parse(v) for k, v in run.data.params.items() if k.startswith("model.")}
    cfg_wo = cfg.with_overrides(**{"data.categorical_columns": []})
    no_income = build_pipeline(cfg_wo, run.data.tags["model_family"], params, seed)
    oof_wo_scores = cross_val_predict(no_income, X_all, y_all, method="predict_proba",
                                      cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
    oof_wo = subgroup_metrics(X_all["Income_Bracket"], y_all, oof_wo_scores, threshold, n_boot=2000, seed=0)

    order = {"Low": 0, "Medium": 1, "High": 2, "ALL": 3}
    for frame in (test, oof, oof_wo):
        frame.sort_values("group", key=lambda s: s.map(order), inplace=True)
    test[COLS].to_csv(OUT / "subgroup_test.csv", index=False)
    oof[COLS].to_csv(OUT / "subgroup_oof.csv", index=False)
    oof_wo[COLS].to_csv(OUT / "subgroup_oof_without_income_feature.csv", index=False)

    def table(frame: pd.DataFrame) -> str:
        lines = ["| Income | n | defaulters | base rate | flagged | error rate | FNR [95% CI] | FPR [95% CI] | "
                 "precision | PR-AUC |", "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in frame.iterrows():
            lines.append(
                f"| {r['group']} | {r['n']} | {r['positives']} | {r['base_rate']:.1%} | {r['flagged_rate']:.1%} | "
                f"{r['error_rate']:.1%} | {r['false_negative_rate']:.1%} [{r['false_negative_rate_ci_low']:.0%}, "
                f"{r['false_negative_rate_ci_high']:.0%}] | {r['false_positive_rate']:.1%} "
                f"[{r['false_positive_rate_ci_low']:.0%}, {r['false_positive_rate_ci_high']:.0%}] | "
                f"{r['precision']:.2f} | {r['average_precision']:.3f} |")
        return "\n".join(lines)

    g = oof.set_index("group")
    fpr_gap = g.loc[["Low", "Medium", "High"], "false_positive_rate"].max() - \
        g.loc[["Low", "Medium", "High"], "false_positive_rate"].min()
    fnr_gap = g.loc[["Low", "Medium", "High"], "false_negative_rate"].max() - \
        g.loc[["Low", "Medium", "High"], "false_negative_rate"].min()
    facts = {
        "registered_model": name, "version": mv.version, "run_id": mv.run_id, "threshold": threshold, "seed": seed,
        "data_version": run.data.tags["data_version"], "oof_fpr_gap": float(fpr_gap), "oof_fnr_gap": float(fnr_gap),
    }
    (OUT / "subgroup_facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    md = [
        "# Subgroup error breakdown by Income_Bracket (generated by scripts/fairness_analysis.py)",
        "",
        f"Model: `models:/{name}@champion` (version {mv.version}, run `{mv.run_id[:8]}`, data "
        f"{run.data.tags['data_version']}), decision threshold {threshold:.4f}.",
        "",
        "FNR = share of actual defaulters the model misses (bank's credit-loss risk). "
        "FPR = share of good customers wrongly flagged (customer-harm risk: extra review, lower limits).",
        "",
        "## 1. Registered model on its held-out test split",
        "",
        table(test),
        "",
        "## 2. Out-of-fold over all rows (same pipeline and hyper-parameters, 5 folds)",
        "",
        table(oof),
        "",
        f"Largest between-group gap (out-of-fold): FPR {fpr_gap:.1%}, FNR {fnr_gap:.1%}.",
        "",
        "## 3. Counterfactual: same model without Income_Bracket as an input (out-of-fold)",
        "",
        table(oof_wo),
    ]
    (OUT / "subgroup_breakdown.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ["Low", "Medium", "High"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=False)
    for ax, metric, title, color in [
        (axes[0], "false_negative_rate", "Missed defaulters (FNR)", "#e76f51"),
        (axes[1], "false_positive_rate", "Good customers flagged (FPR)", "#2a9d8f"),
    ]:
        vals = g.loc[groups, metric].to_numpy()
        lo = vals - g.loc[groups, f"{metric}_ci_low"].to_numpy()
        hi = g.loc[groups, f"{metric}_ci_high"].to_numpy() - vals
        ax.bar(groups, vals, color=color, alpha=0.85, yerr=[lo, hi], capsize=5, width=0.55)
        ax.axhline(g.loc["ALL", metric], color="#333", ls="--", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Income_Bracket")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Registered model error by income group (out-of-fold, 95% bootstrap CI; dashed = overall)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "subgroup_error.png", dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
