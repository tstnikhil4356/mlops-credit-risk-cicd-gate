"""Task 7: data-change drill - is the model trained on v2 really better than v1?

Both dataset versions are materialised from Git+DVC history with ``dvc get --rev``
and trained through the same code path and CI configuration as the quality gate,
once per seed. Every run is tracked in MLflow with its data-version tags.

Comparisons per seed (the seed moves the split and the CV folds):
  own split    each model on its own held-out split (what the CI gate sees)
  common rows  both models on the identical v1 test customers (paired, fair)
  new rows     both models on the test customers that exist only in v2
Plus a paired bootstrap of the AP difference on the common rows, and a drift
check of the 300 new customers against the original 800.

Usage: python scripts/data_drill.py --rev-a data-v1 --rev-b data/v2 --seeds 42 1 2 3 4
Output: reports/drill/*.csv|json|md, reports/figures/drill_v1_v2.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from credit_risk.config import load_config
from credit_risk.data import data_version, file_md5
from credit_risk.evaluate import classification_metrics
from credit_risk.train import run_training

OUT = Path("reports/drill")
FIG = Path("reports/figures")
DATA = "data/raw/credit_card_customers.csv"


def materialise(rev: str, dest: Path, dvc: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    subprocess.run([dvc, "get", ".", DATA, "--rev", rev, "-o", str(dest), "-q"], check=True)
    return dest


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(v1: pd.DataFrame, v2: pd.DataFrame, cfg) -> pd.DataFrame:
    new = v2[~v2[cfg.data.id_column].isin(v1[cfg.data.id_column])]
    rows = []
    for col in [*cfg.data.numeric_columns, cfg.data.target]:
        a, b = v1[col].dropna().to_numpy(), new[col].dropna().to_numpy()
        ks = stats.ks_2samp(a, b)
        rows.append({"feature": col, "v1_mean": a.mean(), "new_rows_mean": b.mean(),
                     "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue, "psi": psi(a, b),
                     "v1_missing": v1[col].isna().mean(), "new_missing": new[col].isna().mean()})
    ct = pd.crosstab(pd.concat([v1, new], keys=["v1", "new"]).index.get_level_values(0),
                     pd.concat([v1, new])["Income_Bracket"])
    chi = stats.chi2_contingency(ct)
    rows.append({"feature": "Income_Bracket (chi2)", "v1_mean": np.nan, "new_rows_mean": np.nan,
                 "ks_stat": chi.statistic, "ks_pvalue": chi.pvalue, "psi": np.nan,
                 "v1_missing": 0.0, "new_missing": 0.0})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev-a", default="data-v1")
    ap.add_argument("--rev-b", default="data/v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--config", default="configs/ci.yaml")
    ap.add_argument("--dvc", default=".venv/bin/dvc")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    tmp = Path("artifacts/drill")
    paths = {"A": materialise(args.rev_a, tmp / "data_a.csv", args.dvc),
             "B": materialise(args.rev_b, tmp / "data_b.csv", args.dvc)}
    # Each revision carries its own md5 -> label registry (v2 is only registered on the v2 branch).
    versions = {}
    for key, rev in (("A", args.rev_a), ("B", args.rev_b)):
        registry = subprocess.run(["git", "show", f"{rev}:{cfg.paths.data_versions}"], capture_output=True,
                                  text=True, check=True).stdout
        reg_path = tmp / f"versions_{key}.yaml"
        reg_path.write_text(registry)
        versions[key] = data_version(cfg.with_overrides(**{"paths.data_versions": str(reg_path)}), paths[key])
    frames = {k: pd.read_csv(p) for k, p in paths.items()}
    la, lb = versions["A"].label, versions["B"].label
    print(f"A = {args.rev_a}: {la} {versions['A'].md5[:8]} ({versions['A'].rows} rows)")
    print(f"B = {args.rev_b}: {lb} {versions['B'].md5[:8]} ({versions['B'].rows} rows)")
    ids_a = set(frames["A"][cfg.data.id_column])

    per_seed, boot_rows = [], []
    for seed in args.seeds:
        res = {}
        for key in ("A", "B"):
            run_cfg = cfg.with_overrides(**{"paths.data_versions": str(tmp / f"versions_{key}.yaml")})
            s = run_training(run_cfg, data_path=str(paths[key]), seed=seed, run_label=f"drill/{versions[key].label}",
                             extra_tags={"drill_rev": args.rev_a if key == "A" else args.rev_b})
            res[key] = s
        objs = {k: res[k]["_objects"] for k in res}
        # Common rows: v1 test customers (present in both test splits thanks to the hash split).
        te_b = objs["B"]["X_test"][cfg.data.id_column]
        common_b = te_b.isin(ids_a).to_numpy()
        model_a, thr_a = objs["A"]["best_model"], res["A"]["metrics"]["threshold"]
        X_b, y_b = objs["B"]["X_test"], objs["B"]["y_test"].to_numpy()
        scores_a_on_b = model_a.predict_proba(X_b)[:, 1]
        scores_b = objs["B"]["test_scores"]
        thr_b = res["B"]["metrics"]["threshold"]
        assert set(objs["A"]["X_test"][cfg.data.id_column]) == set(te_b[common_b]), "split not stable"

        row = {"seed": seed,
               "run_id_a": res["A"]["selected_run_id"], "run_id_b": res["B"]["selected_run_id"],
               "test_rows_a": len(objs["A"]["y_test"]), "test_rows_b": len(y_b),
               "common_rows": int(common_b.sum()), "new_rows": int((~common_b).sum()),
               "new_positives": int(y_b[~common_b].sum())}
        for name, mask in (("own", None), ("common", common_b), ("new", ~common_b)):
            if name == "own":
                ma, mb = res["A"]["metrics"], res["B"]["metrics"]
            else:
                ma = classification_metrics(y_b[mask], scores_a_on_b[mask], thr_a)
                mb = classification_metrics(y_b[mask], scores_b[mask], thr_b)
            for m in ("average_precision", "roc_auc", "recall", "precision", "f2", "brier"):
                row[f"{name}_{m}_{la}"] = ma[m]
                row[f"{name}_{m}_{lb}"] = mb[m]
                row[f"{name}_{m}_delta"] = mb[m] - ma[m]
        per_seed.append(row)

        rng = np.random.default_rng(seed)
        idx = np.flatnonzero(common_b)
        ya, sa, sb = y_b[idx], scores_a_on_b[idx], scores_b[idx]
        diffs = []
        for _ in range(args.n_boot):
            b = rng.integers(0, len(idx), len(idx))
            if ya[b].min() == ya[b].max():
                continue
            diffs.append(average_precision_score(ya[b], sb[b]) - average_precision_score(ya[b], sa[b]))
        d = np.asarray(diffs)
        boot_rows.append({"seed": seed, "ap_delta_common": float(average_precision_score(ya, sb) -
                                                                  average_precision_score(ya, sa)),
                          "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
                          "p_b_better": float((d > 0).mean())})

    df = pd.DataFrame(per_seed)
    boot = pd.DataFrame(boot_rows)
    df.to_csv(OUT / "drill_per_seed.csv", index=False)
    boot.to_csv(OUT / "drill_bootstrap.csv", index=False)
    drift = drift_report(frames["A"], frames["B"], cfg)
    drift.to_csv(OUT / "drift_new_customers.csv", index=False)

    def summarise(col: str) -> dict:
        x = df[col].to_numpy()
        t = stats.ttest_1samp(x, 0.0) if len(x) > 1 and x.std() > 0 else None
        return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "min": float(x.min()), "max": float(x.max()),
                "t_pvalue": float(t.pvalue) if t is not None else float("nan"),
                "seeds_b_better": int((x > 0).sum()), "n": int(len(x))}

    summary = {
        "rev_a": args.rev_a, "rev_b": args.rev_b, "label_a": la, "label_b": lb,
        "md5_a": versions["A"].md5, "md5_b": versions["B"].md5,
        "seeds": args.seeds, "config": cfg.name, "config_sha256": cfg.sha256(),
        "own_ap": {la: float(df[f"own_average_precision_{la}"].mean()), lb: float(df[f"own_average_precision_{lb}"].mean()),
                   "seed_std_" + la: float(df[f"own_average_precision_{la}"].std(ddof=1)),
                   "seed_std_" + lb: float(df[f"own_average_precision_{lb}"].std(ddof=1))},
        "delta_own_ap": summarise("own_average_precision_delta"),
        "delta_common_ap": summarise("common_average_precision_delta"),
        "delta_new_ap": summarise("new_average_precision_delta"),
        "delta_common_recall": summarise("common_recall_delta"),
        "delta_own_recall": summarise("own_recall_delta"),
        "bootstrap_common": boot.to_dict(orient="records"),
        "drift_max_psi": float(drift["psi"].max()),
        "drift_min_ks_pvalue": float(drift["ks_pvalue"].min()),
    }
    (OUT / "drill_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def f(v):
        return f"{v:+.4f}" if isinstance(v, float) else str(v)

    lines = [
        "# Data-change drill: v1 vs v2 (generated by scripts/data_drill.py)",
        "",
        f"A = `{args.rev_a}` ({la}, md5 `{versions['A'].md5[:8]}`, {versions['A'].rows} rows); "
        f"B = `{args.rev_b}` ({lb}, md5 `{versions['B'].md5[:8]}`, {versions['B'].rows} rows). "
        f"Config `{cfg.name}` (`{cfg.sha256()[:10]}`), seeds {args.seeds}.",
        "",
        "## Per seed",
        "",
        f"| seed | AP own {la} | AP own {lb} | delta own | AP common {la} | AP common {lb} | delta common | "
        f"AP new-rows {la} model | AP new-rows {lb} model | recall own {la} | recall own {lb} | runs ({la} / {lb}) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['seed']} | {r[f'own_average_precision_{la}']:.4f} | {r[f'own_average_precision_{lb}']:.4f} | "
            f"{r['own_average_precision_delta']:+.4f} | {r[f'common_average_precision_{la}']:.4f} | "
            f"{r[f'common_average_precision_{lb}']:.4f} | {r['common_average_precision_delta']:+.4f} | "
            f"{r[f'new_average_precision_{la}']:.4f} | {r[f'new_average_precision_{lb}']:.4f} | "
            f"{r[f'own_recall_{la}']:.3f} | {r[f'own_recall_{lb}']:.3f} | "
            f"`{str(r['run_id_a'])[:8]}` / `{str(r['run_id_b'])[:8]}` |")
    lines += ["", "## Across seeds", "", "| comparison | mean delta (B - A) | sd | min | max | seeds B better | "
              "paired t p-value |", "|---|---|---|---|---|---|---|"]
    for key, label in (("delta_own_ap", "AP, own split"), ("delta_common_ap", "AP, common v1 rows (paired)"),
                       ("delta_new_ap", "AP, new-customer rows"), ("delta_own_recall", "recall, own split"),
                       ("delta_common_recall", "recall, common rows")):
        s = summary[key]
        lines.append(f"| {label} | {s['mean']:+.4f} | {s['std']:.4f} | {s['min']:+.4f} | {s['max']:+.4f} | "
                     f"{s['seeds_b_better']}/{s['n']} | {s['t_pvalue']:.3f} |")
    lines += ["", "## Paired bootstrap of AP difference on common rows (per seed)", "",
              "| seed | delta | 95% CI | P(B better) |", "|---|---|---|---|"]
    for r in boot_rows:
        lines.append(f"| {r['seed']} | {r['ap_delta_common']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                     f"{r['p_b_better']:.2f} |")
    lines += ["", "## Drift: 300 new customers vs original 800", "",
              "| feature | v1 mean | new mean | KS / chi2 | p-value | PSI |", "|---|---|---|---|---|---|"]
    for _, r in drift.iterrows():
        lines.append(f"| {r['feature']} | {r['v1_mean']:.2f} | {r['new_rows_mean']:.2f} | {r['ks_stat']:.3f} | "
                     f"{r['ks_pvalue']:.3f} | {r['psi']:.3f} |")
    (OUT / "drill_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(df))
    for ax, prefix, title in ((axes[0], "own", "Own held-out split"), (axes[1], "common", "Identical v1 test rows")):
        ax.plot(x, df[f"{prefix}_average_precision_{la}"], "o-", color="#8d99ae", label=la)
        ax.plot(x, df[f"{prefix}_average_precision_{lb}"], "s-", color="#2a9d8f", label=lb)
        ax.set_xticks(x, [str(s) for s in df["seed"]])
        ax.set_xlabel("seed")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("PR-AUC (test)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{la} vs {lb}: PR-AUC per seed (same code, CI config)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "drill_v1_v2.png", dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
