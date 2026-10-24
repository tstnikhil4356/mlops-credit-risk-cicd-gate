"""Screenshot the public GitHub CI evidence (PR list, each PR with its gate comment, gate run summaries).

Usage: python scripts/screenshot_github.py   (needs `gh` auth for URLs, Playwright + installed Chrome)
Output: reports/ci/screenshots/*.png
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("reports/ci/screenshots")


def gh(*args: str):
    return json.loads(subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    repo = gh("repo", "view", "--json", "url")["url"]
    prs = sorted(gh("pr", "list", "--state", "all", "--limit", "50", "--json", "number,statusCheckRollup"),
                 key=lambda p: p["number"])
    shots = [("00_pull_requests", f"{repo}/pulls?q=is%3Apr", 1400)]
    for p in prs:
        n = p["number"]
        shots.append((f"pr{n}_conversation", f"{repo}/pull/{n}", 1400))
        gate = [c for c in p["statusCheckRollup"] or [] if c.get("name") == "quality-gate"]
        if gate and gate[-1].get("detailsUrl"):
            run_url = gate[-1]["detailsUrl"].split("/job/")[0]
            shots.append((f"pr{n}_gate_run_summary", run_url, 1400))
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        for name, url, width in shots:
            page = browser.new_page(viewport={"width": width, "height": 1000})
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            print(f"  {OUT / name}.png  <- {url}")
            page.close()
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
