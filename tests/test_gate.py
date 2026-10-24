import json

from credit_risk.gate import GateRule, main, read_junit

BASE = {"average_precision": 0.85, "recall": 0.90, "roc_auc": 0.95}


def test_equal_model_passes():
    assert GateRule("average_precision", 0.02).evaluate(BASE, BASE).passed


def test_small_drop_within_tolerance_passes():
    assert GateRule("average_precision", 0.02).evaluate({**BASE, "average_precision": 0.835}, BASE).passed


def test_drop_beyond_tolerance_fails():
    d = GateRule("average_precision", 0.02).evaluate({**BASE, "average_precision": 0.82}, BASE)
    assert not d.passed and "delta -0.0300" in d.checks[0].detail


def test_guard_metric_blocks_even_if_primary_improves():
    d = GateRule("average_precision", 0.02, "recall", 0.70).evaluate(
        {**BASE, "average_precision": 0.90, "recall": 0.60}, BASE)
    assert not d.passed and d.checks[0].passed and not d.checks[1].passed


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def _summary(ap, recall=0.9):
    return {"metrics": {"average_precision": ap, "recall": recall, "roc_auc": 0.9, "precision": 0.5,
                        "f2": 0.7, "balanced_accuracy": 0.8, "brier": 0.1, "accuracy": 0.8, "threshold": 0.3},
            "total_seconds": 12, "git_commit": "abc", "data_md5": "x", "data_version": "v1",
            "config": "ci", "config_sha256": "s", "selected": "logreg"}


def test_cli_blocks_worse_model_and_writes_summary(tmp_path):
    base = _write(tmp_path, "b.json", _summary(0.88))
    cand = _write(tmp_path, "c.json", _summary(0.80))
    out = tmp_path / "s.md"
    rc = main(["--config", "configs/ci.yaml", "--candidate", cand, "--baseline", base,
               "--summary", str(out), "--decision-json", str(tmp_path / "d.json")])
    assert rc == 1
    assert "FAILED, merge blocked" in out.read_text()


def test_cli_blocks_when_tests_failed(tmp_path):
    base = _write(tmp_path, "b.json", _summary(0.88))
    cand = _write(tmp_path, "c.json", _summary(0.88))
    rc = main(["--config", "configs/ci.yaml", "--candidate", cand, "--baseline", base, "--tests-outcome",
               "failure", "--summary", str(tmp_path / "s.md"), "--decision-json", str(tmp_path / "d.json")])
    assert rc == 1


def test_cli_passes_equal_model(tmp_path):
    base = _write(tmp_path, "b.json", _summary(0.88))
    cand = _write(tmp_path, "c.json", _summary(0.88))
    rc = main(["--config", "configs/ci.yaml", "--candidate", cand, "--baseline", base,
               "--summary", str(tmp_path / "s.md"), "--decision-json", str(tmp_path / "d.json")])
    assert rc == 0


def test_cli_blocks_when_training_produced_nothing(tmp_path):
    base = _write(tmp_path, "b.json", _summary(0.88))
    rc = main(["--config", "configs/ci.yaml", "--candidate", str(tmp_path / "missing.json"), "--baseline", base,
               "--summary", str(tmp_path / "s.md"), "--decision-json", str(tmp_path / "d.json")])
    assert rc == 1


def test_read_junit(tmp_path):
    p = tmp_path / "j.xml"
    p.write_text('<testsuites><testsuite tests="2" failures="1" errors="0" skipped="0">'
                 '<testcase classname="t" name="ok"/><testcase classname="t" name="bad"><failure/></testcase>'
                 '</testsuite></testsuites>')
    r = read_junit(p)
    assert r["tests"] == 2 and r["failures"] == 1 and r["failed_cases"] == ["t::bad"]
