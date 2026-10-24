"""Train, tune and track every candidate defined in a configuration.

Usage:
    python -m credit_risk.train --config configs/full.yaml
    python -m credit_risk.train --config configs/ci.yaml --out reports/metrics/ci_metrics.json
    python -m credit_risk.train --config configs/full.yaml --seed 7 --set features.drop_features='[late_payer_flag]'

Model selection uses the cross-validated ``training.selection_metric`` on the
training split only; the held-out test split is scored once per candidate and
never used to choose between them.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score

from credit_risk.config import Config, load_config
from credit_risk.data import data_version, load_raw, train_test_split_by_id
from credit_risk.evaluate import best_fbeta_threshold, classification_metrics
from credit_risk.models import build_pipeline, search_grid
from credit_risk.tracking import flatten, git_provenance, setup_mlflow

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Runtime requirements recorded with every logged model (subset of requirements.txt).
MODEL_REQUIREMENTS = ["scikit-learn==1.5.2", "scipy==1.13.1", "pandas==2.2.3", "numpy==1.26.4", "pyyaml==6.0.2",
                      "pydantic==2.9.2"]


def _parse_overrides(pairs: list[str]) -> dict[str, Any]:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got '{pair}'")
        key, value = pair.split("=", 1)
        out[key] = yaml.safe_load(value)
    return out


def fit_candidate(cfg: Config, cand, X_train, y_train, X_test, y_test, seed: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    cv = StratifiedKFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=seed)
    pipe = build_pipeline(cfg, cand.family, cand.params, seed)
    n_settings = 1
    cv_results = None
    if cand.search_space:
        grid = search_grid(cand.search_space)
        n_settings = len(ParameterGrid(grid))
        common = dict(scoring=cfg.training.selection_metric, cv=cv, n_jobs=cfg.training.n_jobs, refit=True)
        if cfg.training.search == "random" and n_settings > cfg.training.n_iter:
            search = RandomizedSearchCV(pipe, grid, n_iter=cfg.training.n_iter, random_state=seed, **common)
            n_settings = cfg.training.n_iter
        else:
            search = GridSearchCV(pipe, grid, **common)
        search.fit(X_train, y_train)
        best_params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        cv_score = float(search.best_score_)
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        cv_results = pd.DataFrame(search.cv_results_).drop(columns=["params"], errors="ignore")
        model = search.best_estimator_
    else:
        best_params = {}
        model = pipe

    tuned = build_pipeline(cfg, cand.family, {**cand.params, **best_params}, seed)
    oof = cross_val_predict(tuned, X_train, y_train, cv=cv, method="predict_proba", n_jobs=cfg.training.n_jobs)[:, 1]
    if not cand.search_space:
        fold_scores = cross_val_score(tuned, X_train, y_train, cv=cv, scoring=cfg.training.selection_metric,
                                      n_jobs=cfg.training.n_jobs)
        cv_score, cv_std = float(fold_scores.mean()), float(fold_scores.std())
        model.fit(X_train, y_train)

    if cfg.training.threshold_strategy == "max_fbeta":
        threshold = best_fbeta_threshold(y_train, oof, cfg.training.threshold_beta)
    else:
        threshold = cfg.training.fixed_threshold
    cv_at_t = classification_metrics(y_train, oof, threshold)

    test_scores = model.predict_proba(X_test)[:, 1]
    test = classification_metrics(y_test, test_scores, threshold)
    return {
        "name": cand.name,
        "family": cand.family,
        "params": {**cand.params, **best_params},
        "best_params": best_params,
        "n_settings": n_settings,
        "cv_selection_score": cv_score,
        "cv_selection_std": cv_std,
        "cv": {k: v for k, v in cv_at_t.items() if isinstance(v, float)},
        "test": test,
        "threshold": threshold,
        "train_seconds": time.perf_counter() - t0,
        "model": model,
        "cv_results": cv_results,
        "test_scores": test_scores,
    }


def _log_run(mlflow, cfg: Config, res: dict, tags: dict[str, str], X_test, run_name: str) -> str:
    from mlflow.models import infer_signature

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**tags, "candidate": res["name"], "model_family": res["family"]})
        params = {f"model.{k}": v for k, v in res["params"].items()}
        params.update({f"cfg.{k}": v for k, v in flatten(cfg.to_dict()).items() if not k.startswith("training.candidates")})
        params["n_settings_searched"] = res["n_settings"]
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.log_metric("cv_selection_score", res["cv_selection_score"])
        if not np.isnan(res["cv_selection_std"]):
            mlflow.log_metric("cv_selection_std", res["cv_selection_std"])
        mlflow.log_metrics({f"cv_{k}": v for k, v in res["cv"].items() if not np.isnan(v)})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in res["test"].items() if not np.isnan(v)})
        mlflow.log_metric("train_seconds", res["train_seconds"])
        mlflow.log_dict(cfg.to_dict(), "config/resolved_config.yaml")
        mlflow.log_dict({k: v for k, v in res.items() if k in ("name", "family", "params", "cv", "test", "threshold")},
                        "metrics/summary.json")
        if res["cv_results"] is not None:
            mlflow.log_text(res["cv_results"].to_csv(index=False), "search/cv_results.csv")
        if cfg.tracking.log_model:
            example = X_test.head(5)
            signature = infer_signature(example, res["model"].predict_proba(example)[:, 1])
            # Ship the package source with the model so it loads without `pip install credit-risk`.
            mlflow.sklearn.log_model(
                res["model"], artifact_path="model", signature=signature, input_example=example,
                code_paths=[str(Path(__file__).resolve().parent)], pip_requirements=MODEL_REQUIREMENTS,
            )
        return run.info.run_id


def run_training(
    cfg: Config,
    data_path: str | None = None,
    seed: int | None = None,
    run_label: str | None = None,
    extra_tags: dict[str, str] | None = None,
    tracking: bool | None = None,
) -> dict[str, Any]:
    seed = cfg.seed if seed is None else seed
    t_start = time.perf_counter()
    df = load_raw(cfg, data_path)
    dv = data_version(cfg, data_path)
    X_train, X_test, y_train, y_test = train_test_split_by_id(df, cfg, seed)
    tracking = cfg.tracking.enabled if tracking is None else tracking
    mlflow = setup_mlflow(cfg) if tracking else None
    tags = {
        **git_provenance(),
        **dv.as_tags(),
        "config_name": cfg.name,
        "config_sha256": cfg.sha256(),
        "seed": str(seed),
        "run_label": run_label or cfg.name,
        "train_rows": str(len(X_train)),
        "test_rows": str(len(X_test)),
        **(extra_tags or {}),
    }

    results = []
    for cand in cfg.training.candidates:
        res = fit_candidate(cfg, cand, X_train, y_train, X_test, y_test, seed)
        if mlflow is not None:
            res["run_id"] = _log_run(mlflow, cfg, res, tags, X_test, f"{run_label or cfg.name}/{cand.name}/s{seed}")
        results.append(res)
        print(
            f"  {cand.name:<22} cv_{cfg.training.selection_metric}={res['cv_selection_score']:.4f}  "
            f"test_AP={res['test']['average_precision']:.4f}  test_recall={res['test']['recall']:.3f}  "
            f"({res['train_seconds']:.1f}s)"
        )

    best = max(results, key=lambda r: -np.inf if np.isnan(r["cv_selection_score"]) else r["cv_selection_score"])
    summary = {
        "config": cfg.name,
        "config_sha256": cfg.sha256(),
        "seed": seed,
        "run_label": run_label or cfg.name,
        **{k: v for k, v in tags.items() if k.startswith(("git_", "data_"))},
        "selection_metric": cfg.training.selection_metric,
        "selected": best["name"],
        "selected_run_id": best.get("run_id"),
        "metrics": {k: v for k, v in best["test"].items()},
        "cv_selection_score": best["cv_selection_score"],
        "candidates": [
            {
                "name": r["name"],
                "family": r["family"],
                "run_id": r.get("run_id"),
                "params": r["params"],
                "cv_selection_score": r["cv_selection_score"],
                "cv_selection_std": r["cv_selection_std"],
                "test": r["test"],
                "train_seconds": round(r["train_seconds"], 2),
            }
            for r in results
        ],
        "total_seconds": round(time.perf_counter() - t_start, 2),
    }
    summary["_objects"] = {"best_model": best["model"], "X_test": X_test, "y_test": y_test,
                           "test_scores": best["test_scores"], "results": results}
    return summary


def write_summary(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in summary.items() if not k.startswith("_")}
    path.write_text(json.dumps(clean, indent=2, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-path", help="Override paths.raw_data (e.g. a file from `dvc get --rev`)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a config value, e.g. --set gate.max_drop=0.01")
    ap.add_argument("--run-label")
    ap.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--out", help="Write the run summary JSON here")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.set:
        cfg = cfg.with_overrides(**_parse_overrides(args.set))
    tags = {k: str(v) for k, v in _parse_overrides(args.tag).items()}
    print(f"[train] config={cfg.name} sha={cfg.sha256()[:10]} seed={args.seed if args.seed is not None else cfg.seed}")
    summary = run_training(cfg, args.data_path, args.seed, args.run_label, tags,
                           tracking=False if args.no_tracking else None)
    print(f"[train] data={summary['data_version']} ({summary['data_md5'][:8]}) commit={summary['git_commit'][:8]} "
          f"selected={summary['selected']} test_AP={summary['metrics']['average_precision']:.4f} "
          f"in {summary['total_seconds']}s")
    if args.out:
        write_summary(summary, args.out)
        print(f"[train] summary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
