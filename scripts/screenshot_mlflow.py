"""Capture MLflow UI screenshots as experiment-tracking evidence (Task 5 deliverable).

Starts `mlflow ui` on the local SQLite store, drives the installed Google Chrome
with Playwright (no browser download), dismisses promotional pop-ups, and saves
reports/tracking/screenshots/*.png.

Usage: python scripts/screenshot_mlflow.py   (needs requirements-docs.txt + Chrome)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = 5055
BASE = f"http://127.0.0.1:{PORT}"
OUT = Path("reports/tracking/screenshots")


def api(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}/api/2.0/mlflow/{path}") as r:
        return json.load(r)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([str(Path(sys.executable).parent / "mlflow"), "ui", "--backend-store-uri",
                               "sqlite:///mlflow.db", "--port", str(PORT)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{BASE}/health")
                break
            except OSError:
                time.sleep(1)
        exp = api("experiments/get-by-name?experiment_name=credit-risk")["experiment"]["experiment_id"]
        ci = api("experiments/get-by-name?experiment_name=credit-risk-ci")["experiment"]["experiment_id"]
        reg = json.loads(Path("reports/registry/registered_model.json").read_text())
        run = reg["run_id"]
        sel = [r["run_id"] for r in reg["ranking"][:4]]
        compare = "#/compare-runs?runs=" + json.dumps(sel).replace(" ", "") + f"&experiments=[\"{exp}\"]"
        shots = [
            ("01_experiment_runs", f"#/experiments/{exp}", (1900, 900)),
            ("02_ci_gate_history_and_drill_runs", f"#/experiments/{ci}", (1900, 1500)),
            ("03_run_comparison", compare, (1900, 1500)),
            ("04_registered_model", "#/models/credit-risk-classifier", (1600, 700)),
            ("05_registered_version_lineage", f"#/models/credit-risk-classifier/versions/{reg['version']}",
             (1600, 1300)),
            ("06_selected_run_overview", f"#/experiments/{exp}/runs/{run}", (1600, 1500)),
            ("07_selected_run_artifacts", f"#/experiments/{exp}/runs/{run}/artifacts", (1600, 1000)),
        ]
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True,
                                        args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            for name, path, (w, h) in shots:
                page = browser.new_page(viewport={"width": w, "height": h})
                page.goto(f"{BASE}/{path}")
                page.wait_for_timeout(4000)
                for label in ("Close", "Got it", "Dismiss"):
                    for btn in page.get_by_role("button", name=label).all():
                        try:
                            btn.click(timeout=1000)
                        except Exception:  # noqa: BLE001 - pop-up may already be gone
                            pass
                page.keyboard.press("Escape")
                page.wait_for_timeout(800)
                page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
                print(f"  {OUT / name}.png")
                page.close()
            browser.close()
    finally:
        server.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
