"""Experiment-tracking helpers: provenance tags and MLflow setup."""

from __future__ import annotations

import os
import subprocess
from typing import Any


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_provenance() -> dict[str, str]:
    """Commit, branch and dirty state of the code that produced a run."""
    commit = os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD") or "unknown"
    branch = os.environ.get("GITHUB_HEAD_REF") or _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    # Untracked files (reports, scratch) do not change the code that ran; modified tracked files do.
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"git_commit": commit, "git_branch": branch, "git_dirty": str(dirty).lower()}


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def setup_mlflow(cfg):
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg.tracking.tracking_uri))
    mlflow.set_experiment(cfg.tracking.experiment_name)
    return mlflow
