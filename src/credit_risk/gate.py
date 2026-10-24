"""CI quality gate: compare a fresh CI training run against the committed baseline.

Usage (as run by .github/workflows/quality-gate.yml):
    python -m credit_risk.gate --config configs/ci.yaml \
        --candidate reports/metrics/ci_metrics.json \
        --tests-junit reports/ci/junit.xml --summary reports/ci/gate_summary.md

Exit code 0 = merge allowed, 1 = blocked (model worse, guard violated, tests
failed or time budget exceeded). The Markdown summary is written for reviewers
and appended to $GITHUB_STEP_SUMMARY when present.
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from credit_risk.config import ConfigError, load_config


@dataclass
class GateRule:
    """``metric`` may drop by at most ``max_drop``; optional absolute floor on ``guard_metric``."""

    metric: str
    max_drop: float
    guard_metric: str | None = None
    guard_min: float = 0.0
    name: str = ""

    def describe(self) -> str:
        text = f"{self.metric} >= baseline - {self.max_drop:g}"
        if self.guard_metric:
            text += f" AND {self.guard_metric} >= {self.guard_min:g}"
        return text

    def evaluate(self, candidate: dict[str, float], baseline: dict[str, float]) -> "GateDecision":
        checks = []
        new, ref = candidate[self.metric], baseline[self.metric]
        checks.append(
            Check(
                f"{self.metric} vs baseline",
                passed=new >= ref - self.max_drop - 1e-12,
                detail=f"{new:.4f} vs {ref:.4f} (delta {new - ref:+.4f}, allowed -{self.max_drop:g})",
            )
        )
        if self.guard_metric:
            g = candidate[self.guard_metric]
            checks.append(
                Check(
                    f"{self.guard_metric} floor",
                    passed=g >= self.guard_min - 1e-12,
                    detail=f"{g:.4f} vs floor {self.guard_min:g}",
                )
            )
        return GateDecision(rule=self, checks=checks)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateDecision:
    rule: GateRule
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def rule_from_config(cfg) -> GateRule:
    g = cfg.gate
    return GateRule(metric=g.metric, max_drop=g.max_drop, guard_metric=g.guard_metric, guard_min=g.guard_min,
                    name="configured")


def read_junit(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {"available": False, "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "failed_cases": []}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    failed = [
        f"{case.get('classname', '')}::{case.get('name')}"
        for case in root.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    ]
    return {"available": True, **totals, "failed_cases": failed}


def _fmt(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def render_summary(
    decision: GateDecision | None,
    candidate: dict | None,
    baseline: dict,
    tests: dict,
    tests_outcome: str,
    time_ok: bool,
    budget_minutes: float,
) -> str:
    tests_ok = tests_outcome == "success" and tests.get("failures", 0) == 0 and tests.get("errors", 0) == 0
    model_ok = decision is not None and decision.passed
    overall = tests_ok and model_ok and time_ok
    icon = lambda ok: "✅" if ok else "❌"  # noqa: E731
    lines = [
        f"## {icon(overall)} Model quality gate: {'PASSED, merge allowed' if overall else 'FAILED, merge blocked'}",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    if tests.get("available"):
        detail = f"{tests['tests']} tests, {tests['failures']} failures, {tests['errors']} errors"
    else:
        detail = f"pytest step outcome: {tests_outcome}"
    lines.append(f"| Unit tests | {icon(tests_ok)} | {detail} |")
    if decision is None:
        lines.append("| Model training | ❌ | CI training did not produce metrics (see logs) |")
    else:
        for c in decision.checks:
            lines.append(f"| {c.name} | {icon(c.passed)} | {c.detail} |")
    if candidate is not None:
        lines.append(
            f"| CI time budget | {icon(time_ok)} | training {candidate.get('total_seconds', 0):.0f}s "
            f"(budget {budget_minutes:g} min) |"
        )
    if tests.get("failed_cases"):
        lines += ["", "**Failing tests:**", *[f"- `{c}`" for c in tests["failed_cases"][:20]]]
    if candidate is not None:
        bm, cm = baseline["metrics"], candidate["metrics"]
        lines += [
            "",
            f"**Gate rule:** `{decision.rule.describe() if decision else 'n/a'}`",
            "",
            "| Metric (held-out test split) | Baseline | This PR | Delta |",
            "|---|---|---|---|",
        ]
        for m in ("average_precision", "roc_auc", "recall", "precision", "f2", "balanced_accuracy", "brier", "accuracy"):
            if m in bm and m in cm:
                lines.append(f"| {m} | {bm[m]:.4f} | {cm[m]:.4f} | {cm[m] - bm[m]:+.4f} |")
        lines += [
            "",
            "<details><summary>Provenance</summary>",
            "",
            "| | Baseline | This PR |",
            "|---|---|---|",
            f"| git commit | `{baseline.get('git_commit', '?')[:10]}` | `{candidate.get('git_commit', '?')[:10]}` |",
            f"| data version | {baseline.get('data_version', '?')} (`{baseline.get('data_md5', '?')[:8]}`) | "
            f"{candidate.get('data_version', '?')} (`{candidate.get('data_md5', '?')[:8]}`) |",
            f"| config | {baseline.get('config', '?')} (`{baseline.get('config_sha256', '?')[:10]}`) | "
            f"{candidate.get('config', '?')} (`{candidate.get('config_sha256', '?')[:10]}`) |",
            f"| selected model | {baseline.get('selected', '?')} | {candidate.get('selected', '?')} |",
            f"| decision threshold | {_fmt(bm.get('threshold'))} | {_fmt(cm.get('threshold'))} |",
            "",
            "</details>",
        ]
        if baseline.get("data_md5") != candidate.get("data_md5"):
            lines += [
                "",
                "> ⚠️ **Data version differs from the baseline.** The comparison is on different test rows; "
                "see docs/data_change_drill.md. Accepting it also requires a baseline update PR "
                "(docs/ci_quality_gate.md, 'Baseline update rule').",
            ]
    lines += ["", f"_Baseline file: `{baseline.get('_path', 'baseline/baseline_metrics.json')}` "
                  f"(updated {baseline.get('updated_at', '?')}, reason: {baseline.get('reason', '?')})_"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--candidate", required=True, help="Run summary JSON written by credit_risk.train --out")
    ap.add_argument("--baseline", help="Defaults to paths.baseline from the config")
    ap.add_argument("--tests-junit")
    ap.add_argument("--tests-outcome", default="success", help="Outcome of the pytest step (success/failure)")
    ap.add_argument("--summary", default="reports/ci/gate_summary.md")
    ap.add_argument("--decision-json", default="reports/ci/gate_decision.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    baseline_path = Path(args.baseline or cfg.paths.baseline)
    baseline = json.loads(baseline_path.read_text())
    baseline["_path"] = str(baseline_path)
    candidate = json.loads(Path(args.candidate).read_text()) if Path(args.candidate).exists() else None
    tests = read_junit(args.tests_junit)

    rule = rule_from_config(cfg)
    decision = rule.evaluate(candidate["metrics"], baseline["metrics"]) if candidate else None
    time_ok = candidate is not None and candidate.get("total_seconds", 0) <= cfg.gate.max_minutes * 60
    text = render_summary(decision, candidate, baseline, tests, args.tests_outcome, time_ok, cfg.gate.max_minutes)

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(text)
    tests_ok = args.tests_outcome == "success" and tests.get("failures", 0) == 0 and tests.get("errors", 0) == 0
    passed = bool(decision and decision.passed and tests_ok and time_ok)
    Path(args.decision_json).write_text(json.dumps({
        "passed": passed,
        "tests_ok": tests_ok,
        "model_ok": bool(decision and decision.passed),
        "time_ok": time_ok,
        "rule": rule.describe(),
        "checks": [c.__dict__ for c in decision.checks] if decision else [],
    }, indent=2) + "\n")
    print(text)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
