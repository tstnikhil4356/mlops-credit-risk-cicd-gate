"""End-to-end smoke tests on synthetic data: train -> save -> reload -> identical predictions."""

import pickle

import numpy as np
import pytest

from credit_risk.data import train_test_split_by_id
from credit_risk.models import build_pipeline
from credit_risk.train import fit_candidate


@pytest.mark.parametrize("family, params", [
    ("logistic_regression", {"max_iter": 500}),
    ("random_forest", {"n_estimators": 20, "max_depth": 3}),
    ("hist_gradient_boosting", {"max_iter": 20}),
])
def test_pipeline_trains_and_predicts_probabilities(cfg, customers, family, params):
    X_tr, X_te, y_tr, _ = train_test_split_by_id(customers, cfg)
    pipe = build_pipeline(cfg, family, params, seed=0).fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    assert p.shape == (len(X_te),) and ((0 <= p) & (p <= 1)).all()


def test_training_and_inference_use_identical_transformations(cfg, customers):
    X_tr, X_te, y_tr, _ = train_test_split_by_id(customers, cfg)
    pipe = build_pipeline(cfg, "logistic_regression", {"max_iter": 500}, seed=0).fit(X_tr, y_tr)
    reloaded = pickle.loads(pickle.dumps(pipe))
    np.testing.assert_array_equal(pipe.predict_proba(X_te), reloaded.predict_proba(X_te))
    np.testing.assert_array_equal(
        pipe.named_steps["features"].transform(X_te), reloaded.named_steps["features"].transform(X_te))


def test_training_is_reproducible(dev_cfg, customers):
    X_tr, X_te, y_tr, y_te = train_test_split_by_id(customers, dev_cfg)
    cand = dev_cfg.training.candidates[0]
    a = fit_candidate(dev_cfg, cand, X_tr, y_tr, X_te, y_te, seed=1)
    b = fit_candidate(dev_cfg, cand, X_tr, y_tr, X_te, y_te, seed=1)
    assert a["test"] == b["test"] and a["threshold"] == b["threshold"]


def test_model_beats_chance_on_signal(dev_cfg, customers):
    X_tr, X_te, y_tr, y_te = train_test_split_by_id(customers, dev_cfg)
    res = fit_candidate(dev_cfg, dev_cfg.training.candidates[0], X_tr, y_tr, X_te, y_te, seed=0)
    assert res["test"]["roc_auc"] > 0.6
