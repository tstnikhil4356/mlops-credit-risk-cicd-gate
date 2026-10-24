"""Select the final model from tracked runs and register it in the MLflow registry.

Selection uses the cross-validated selection metric (never the test split) among
runs of the model-selection campaign. The registered version carries tags that
trace it back to its code commit, data version and configuration, and receives
the ``champion`` alias used by predict.py.

Usage:
    python -m credit_risk.register --config configs/full.yaml --run-label selection
"""

from __future__ import annotations

import argparse
import json

from credit_risk.config import load_config


def select_run(client, experiment_id: str, run_label: str, metric: str = "cv_selection_score", data_version=None):
    filt = f"tags.run_label = '{run_label}' and attributes.status = 'FINISHED'"
    if data_version:
        filt += f" and tags.data_version = '{data_version}'"
    runs = client.search_runs([experiment_id], filter_string=filt, order_by=[f"metrics.{metric} DESC"], max_results=500)
    runs = [r for r in runs if r.data.tags.get("model_family") != "dummy" and metric in r.data.metrics]
    if not runs:
        raise SystemExit(f"No finished runs with run_label={run_label!r} found")
    return runs[0], runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--run-label", default="selection")
    ap.add_argument("--data-version")
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--out", default="reports/registry/registered_model.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    import mlflow

    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(cfg.tracking.experiment_name)
    if exp is None:
        raise SystemExit(f"Experiment {cfg.tracking.experiment_name!r} not found; run training first")
    best, ranked = select_run(client, exp.experiment_id, args.run_label, data_version=args.data_version)
    tags = best.data.tags
    mv = mlflow.register_model(f"runs:/{best.info.run_id}/model", cfg.tracking.registered_model_name)
    trace = {
        "git_commit": tags.get("git_commit"),
        "git_dirty": tags.get("git_dirty"),
        "data_version": tags.get("data_version"),
        "data_md5": tags.get("data_md5"),
        "config_name": tags.get("config_name"),
        "config_sha256": tags.get("config_sha256"),
        "seed": tags.get("seed"),
        "candidate": tags.get("candidate"),
        "selection_metric": f"cv {cfg.training.selection_metric}",
    }
    for k, v in trace.items():
        client.set_model_version_tag(cfg.tracking.registered_model_name, mv.version, k, str(v))
    client.set_registered_model_alias(cfg.tracking.registered_model_name, args.alias, mv.version)
    client.update_model_version(
        cfg.tracking.registered_model_name, mv.version,
        description=(f"{tags.get('candidate')} selected by cv {cfg.training.selection_metric}="
                     f"{best.data.metrics['cv_selection_score']:.4f}; data {tags.get('data_version')} "
                     f"({tags.get('data_md5', '')[:8]}); commit {tags.get('git_commit', '')[:10]}; "
                     f"config {tags.get('config_name')} ({tags.get('config_sha256', '')[:10]})"),
    )
    record = {
        "registered_model": cfg.tracking.registered_model_name,
        "version": mv.version,
        "alias": args.alias,
        "run_id": best.info.run_id,
        "model_uri": f"models:/{cfg.tracking.registered_model_name}@{args.alias}",
        **trace,
        "cv_selection_score": best.data.metrics["cv_selection_score"],
        "test_metrics": {k.removeprefix("test_"): v for k, v in best.data.metrics.items() if k.startswith("test_")},
        "ranking": [
            {"run_id": r.info.run_id, "candidate": r.data.tags.get("candidate"),
             "cv_selection_score": r.data.metrics.get("cv_selection_score")}
            for r in ranked[:10]
        ],
    }
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("registered_model", "version", "run_id", "candidate",
                                              "data_version", "git_commit")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
