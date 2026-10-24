#!/usr/bin/env bash
# Capture MLflow UI screenshots as tracking evidence (headless Chrome; free, local).
# Usage: scripts/screenshot_mlflow.sh   -> reports/tracking/screenshots/*.png
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PORT="${PORT:-5055}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
command -v google-chrome >/dev/null && CHROME=google-chrome
OUT=reports/tracking/screenshots; mkdir -p "$OUT"
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port "$PORT" >/dev/null 2>&1 &
PID=$!; trap 'kill $PID 2>/dev/null' EXIT
for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
exp_id() { curl -s "http://127.0.0.1:$PORT/api/2.0/mlflow/experiments/get-by-name?experiment_name=$1" | .venv/bin/python -c "import sys,json;print(json.load(sys.stdin)['experiment']['experiment_id'])"; }
shot() { "$CHROME" --headless=new --disable-gpu --hide-scrollbars --window-size="${3:-1800,1100}" --virtual-time-budget=15000 --screenshot="$OUT/$1.png" "http://127.0.0.1:$PORT/$2" >/dev/null 2>&1; echo "  $OUT/$1.png"; }
E=$(exp_id credit-risk); C=$(exp_id credit-risk-ci)
RUN=$(.venv/bin/python -c "import json;print(json.load(open('reports/registry/registered_model.json'))['run_id'])")
echo "Screenshots:"
shot 01_experiment_runs "#/experiments/$E" 1900,1200
shot 02_ci_gate_history_runs "#/experiments/$C" 1900,1200
shot 03_registered_model "#/models/credit-risk-classifier" 1600,800
shot 04_registered_version "#/models/credit-risk-classifier/versions/1" 1600,1100
shot 05_selected_run_overview "#/experiments/$E/runs/$RUN" 1600,1400
shot 06_selected_run_artifacts "#/experiments/$E/runs/$RUN/artifacts" 1600,1000
