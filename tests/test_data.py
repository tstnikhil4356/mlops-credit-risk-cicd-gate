import numpy as np
import pandas as pd
import pytest

from credit_risk.data import DataValidationError, split_mask, train_test_split_by_id, validate


def test_valid_data_passes(cfg, customers):
    validate(customers, cfg)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.drop(columns=["Age"]), "Missing columns"),
        (lambda d: d.assign(Customer_ID="CC-1"), "duplicate"),
        (lambda d: d.assign(High_Risk_Customer=2), "must be 0/1"),
        (lambda d: d.assign(Income_Bracket="Unknown"), "unknown categories"),
        (lambda d: d.assign(Age=-4), "outside"),
        (lambda d: d.assign(Monthly_Spend=np.nan), "missing fraction"),
    ],
)
def test_invalid_data_is_rejected(cfg, customers, mutate, message):
    with pytest.raises(DataValidationError, match=message):
        validate(mutate(customers.copy()), cfg)


def test_split_is_deterministic_and_seed_dependent(customers):
    a = split_mask(customers["Customer_ID"], 0.25, 42)
    assert (a == split_mask(customers["Customer_ID"], 0.25, 42)).all()
    assert not (a == split_mask(customers["Customer_ID"], 0.25, 7)).all()
    assert 0.15 < a.mean() < 0.35


def test_split_is_stable_when_new_rows_are_appended(cfg, customers):
    """v2 = v1 + new rows: every v1 customer must keep its train/test side."""
    v1 = customers.iloc[:200]
    v2 = customers
    m1 = split_mask(v1["Customer_ID"], cfg.data.test_size, 42)
    m2 = split_mask(v2["Customer_ID"], cfg.data.test_size, 42)
    assert (m1 == m2[:200]).all()


def test_train_and_test_are_disjoint(cfg, customers):
    X_tr, X_te, y_tr, y_te = train_test_split_by_id(customers, cfg)
    assert set(X_tr["Customer_ID"]).isdisjoint(X_te["Customer_ID"])
    assert len(X_tr) + len(X_te) == len(customers)
    assert "High_Risk_Customer" not in X_tr
