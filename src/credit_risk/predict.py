"""Batch inference with the registered model.

The registered artefact is the whole fitted pipeline (feature builder, encoder,
classifier), so new data goes through exactly the transformations used in training.

Usage:
    python -m credit_risk.predict --input new_customers.csv --output scored.csv
    python -m credit_risk.predict --model-uri runs:/<run_id>/model --input ... --output ...
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from credit_risk.config import load_config


def load_model(model_uri: str):
    import mlflow.sklearn

    return mlflow.sklearn.load_model(model_uri)


def resolve_threshold(model_uri: str, default: float) -> float:
    """Read the decision threshold stored with the run that produced the model."""
    import mlflow

    client = mlflow.MlflowClient()
    try:
        if model_uri.startswith("models:/") and "@" in model_uri:
            name, alias = model_uri.removeprefix("models:/").split("@", 1)
            run_id = client.get_model_version_by_alias(name, alias).run_id
        elif model_uri.startswith("runs:/"):
            run_id = model_uri.removeprefix("runs:/").split("/", 1)[0]
        else:
            return default
        return float(client.get_run(run_id).data.metrics.get("test_threshold", default))
    except Exception:  # noqa: BLE001 - fall back to the configured threshold
        return default


def score(model, df: pd.DataFrame, threshold: float, id_column: str) -> pd.DataFrame:
    proba = model.predict_proba(df)[:, 1]
    out = pd.DataFrame({id_column: df[id_column]}) if id_column in df else pd.DataFrame(index=df.index)
    out["risk_score"] = proba
    out["high_risk_pred"] = (proba >= threshold).astype(int)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--model-uri", help="Defaults to models:/<registered_model_name>@champion")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, help="Override the threshold stored with the run")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    import mlflow

    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    uri = args.model_uri or f"models:/{cfg.tracking.registered_model_name}@champion"
    model = load_model(uri)
    threshold = args.threshold if args.threshold is not None else resolve_threshold(uri, cfg.training.fixed_threshold)
    df = pd.read_csv(args.input)
    if cfg.data.target in df:
        df = df.drop(columns=[cfg.data.target])
    result = score(model, df, threshold, cfg.data.id_column)
    result.to_csv(args.output, index=False)
    print(json.dumps({"model_uri": uri, "threshold": threshold, "rows": len(result),
                      "flagged": int(result["high_risk_pred"].sum())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
