#!/usr/bin/env bash
# Task 2 demonstration: restore an earlier commit together with the exact data it
# used, retrain, and show the result is identical to the one recorded at that commit.
#
#   scripts/restore_demo.sh [git-ref]      (default: baseline-v1)
#
# Steps: fresh clone -> checkout ref -> fetch DVC storage -> dvc pull (data pinned by
# the .dvc pointer of that commit) -> train CI config -> compare with the baseline
# file committed at that ref (exact match required, tolerance 1e-6).
set -euo pipefail
REF="${1:-baseline-v1}"
SRC="$(git rev-parse --show-toplevel)"
PY="${PYTHON:-$SRC/.venv/bin/python}"
DVC="${DVC:-$SRC/.venv/bin/dvc}"
WORK="$(mktemp -d)/restore"
LOG="$SRC/reports/restore/restore_${REF//\//_}.log"
mkdir -p "$(dirname "$LOG")"

{
  echo "== restore demo: ref=$REF  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  git clone -q "$SRC" "$WORK"
  cd "$WORK"
  git checkout -q "$REF"
  echo "commit     : $(git rev-parse HEAD)"
  echo "data .dvc  : $(grep md5 data/raw/credit_card_customers.csv.dvc)"
  git fetch -q origin dvc-storage:dvc-storage && git worktree add -q .dvc-storage dvc-storage
  "$DVC" pull -q
  echo "data md5   : $(md5 -q data/raw/credit_card_customers.csv 2>/dev/null || md5sum data/raw/credit_card_customers.csv | cut -d' ' -f1)"
  PYTHONPATH=src "$PY" -m credit_risk.train --config configs/ci.yaml --no-tracking --out /tmp/restore_metrics.json 2>&1 | grep '\[train\]'
  PYTHONPATH=src "$PY" scripts/update_baseline.py --check --candidate /tmp/restore_metrics.json
  echo "recorded   : $(PYTHONPATH=src "$PY" -c "import json;b=json.load(open('baseline/baseline_metrics.json'));print('AP=%.6f recall=%.6f data=%s' % (b['metrics']['average_precision'], b['metrics']['recall'], b['data_version']))")"
  echo "reproduced : $(PYTHONPATH=src "$PY" -c "import json;b=json.load(open('/tmp/restore_metrics.json'));print('AP=%.6f recall=%.6f data=%s' % (b['metrics']['average_precision'], b['metrics']['recall'], b['data_version']))")"
} 2>&1 | grep -v -i "warning" | tee "$LOG"
rm -rf "$(dirname "$WORK")"
