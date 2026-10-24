# One-command entry points. Every target is reproducible from a fresh clone.
PY  ?= .venv/bin/python
DVC ?= .venv/bin/dvc
V2_REV ?= data-v2
export PYTHONPATH := src

.PHONY: help setup data test lint-config train-dev train-ci train-full gate experiments register \
        gate-analysis fairness drill evidence diagrams screenshots reproduce restore-demo ui report clean-tracking

help:            ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup:           ## Create .venv and install the locked dependencies
	python3.11 -m venv .venv
	.venv/bin/pip install -r requirements.lock
	.venv/bin/pip install -e . --no-deps

data:            ## Fetch DVC storage branch and pull the data version of this commit
	DVC=$(DVC) scripts/dvc_storage.sh pull

test:            ## Unit tests
	$(PY) -m pytest

lint-config:     ## Validate every YAML config
	@for f in configs/*.yaml; do $(PY) -c "from credit_risk.config import load_config as l; c=l('$$f'); print('valid', '$$f', c.sha256()[:10])"; done

train-dev:       ## Fast development training run
	$(PY) -m credit_risk.train --config configs/dev.yaml

train-ci:        ## Training exactly as the CI quality gate runs it
	$(PY) -m credit_risk.train --config configs/ci.yaml --out reports/metrics/ci_metrics.json

train-full:      ## Full model-selection run (all families)
	$(PY) -m credit_risk.train --config configs/full.yaml --run-label selection

gate: train-ci   ## Run the quality gate locally (tests + train + compare)
	$(PY) -m pytest --junitxml=reports/ci/junit.xml -q
	$(PY) -m credit_risk.gate --config configs/ci.yaml --candidate reports/metrics/ci_metrics.json \
	    --tests-junit reports/ci/junit.xml --summary reports/ci/gate_summary.md

experiments:     ## Task 5/8: selection, ablation and gate-history campaign (tracked)
	$(PY) scripts/run_experiments.py

register:        ## Register the selected model (alias @champion)
	$(PY) -m credit_risk.register --config configs/full.yaml --run-label selection

gate-analysis:   ## Task 8: compare candidate gate rules on the run history
	$(PY) scripts/gate_rule_analysis.py

fairness:        ## Task 8: subgroup error breakdown by Income_Bracket
	$(PY) scripts/fairness_analysis.py

drill:           ## Task 7: v1 vs v2 over five seeds
	$(PY) scripts/data_drill.py --rev-a data-v1 --rev-b $(V2_REV) --dvc $(DVC)

evidence:        ## Export tracking evidence (runs CSV, comparison, registry)
	$(PY) scripts/export_tracking.py

diagrams:        ## Lifecycle and CI-flow figures
	$(PY) scripts/make_diagrams.py

screenshots:     ## MLflow UI screenshots (needs Chrome)
	scripts/screenshot_mlflow.sh

reproduce: clean-tracking experiments register gate-analysis fairness drill evidence diagrams  ## Rebuild every reported result

restore-demo:    ## Task 2: restore an earlier commit + data and obtain the same result
	scripts/restore_demo.sh baseline-v1

ui:              ## Open the MLflow UI on http://127.0.0.1:5000
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

report:          ## Build docs/report PDF + DOCX from docs/report.md
	$(PY) scripts/build_report.py

clean-tracking:  ## Start a fresh local MLflow store (results are fully reproducible)
	rm -rf mlflow.db mlruns mlartifacts
