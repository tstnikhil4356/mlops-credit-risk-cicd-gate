import numpy as np
import pandas as pd
import pytest

from credit_risk.features import RATIO_CAP, CreditFeatureBuilder, build_feature_builder, safe_ratio


def _row(**overrides):
    base = {
        "Customer_ID": "CC-1", "Age": 40, "Income_Bracket": "Medium", "Credit_Limit": 10000.0,
        "Current_Balance": 5000.0, "Monthly_Spend": 600.0, "Num_Transactions_Monthly": 12,
        "Payment_Delay_Days": 3.0, "Tenure_Months": 24,
    }
    base.update(overrides)
    return pd.DataFrame([base])


@pytest.fixture
def builder(cfg, customers):
    return build_feature_builder(cfg).fit(customers.drop(columns=["High_Risk_Customer"]))


# ---- safe_ratio edge cases -----------------------------------------------------
def test_ratio_plain():
    assert safe_ratio(pd.Series([500.0]), pd.Series([1000.0])).iloc[0] == pytest.approx(0.5)


def test_ratio_zero_limit_with_balance_is_capped():
    assert safe_ratio(pd.Series([100.0]), pd.Series([0.0])).iloc[0] == RATIO_CAP


def test_ratio_zero_limit_zero_balance_is_zero():
    assert safe_ratio(pd.Series([0.0]), pd.Series([0.0])).iloc[0] == 0.0


def test_ratio_negative_limit_treated_as_no_limit():
    assert safe_ratio(pd.Series([10.0]), pd.Series([-5.0])).iloc[0] == RATIO_CAP


def test_ratio_is_clipped_and_never_infinite():
    r = safe_ratio(pd.Series([1e9, -5.0]), pd.Series([1.0, 100.0]))
    assert r.tolist() == [RATIO_CAP, 0.0]
    assert np.isfinite(r).all()


def test_ratio_missing_stays_missing():
    assert np.isnan(safe_ratio(pd.Series([np.nan]), pd.Series([100.0])).iloc[0])


# ---- engineered features -------------------------------------------------------
def test_utilisation_and_flags(builder):
    out = builder.transform(_row(Current_Balance=8500.0, Payment_Delay_Days=10.0))
    assert out["utilisation"].iloc[0] == pytest.approx(0.85)
    assert out["high_utilisation_flag"].iloc[0] == 1
    assert out["late_payer_flag"].iloc[0] == 1
    assert out["credit_stress_index"].iloc[0] == pytest.approx(0.85 * np.log1p(10))


def test_flag_boundaries_follow_config(builder, cfg):
    cut = cfg.features.utilisation_risk_cutoff
    at = builder.transform(_row(Current_Balance=10000.0 * cut))
    below = builder.transform(_row(Current_Balance=10000.0 * cut - 1))
    assert at["high_utilisation_flag"].iloc[0] == 1 and below["high_utilisation_flag"].iloc[0] == 0
    d = cfg.features.payment_delay_flag_days
    assert builder.transform(_row(Payment_Delay_Days=d))["late_payer_flag"].iloc[0] == 0
    assert builder.transform(_row(Payment_Delay_Days=d + 1))["late_payer_flag"].iloc[0] == 1


def test_zero_credit_limit_row_is_handled(builder):
    out = builder.transform(_row(Credit_Limit=0.0, Current_Balance=1200.0, Monthly_Spend=300.0))
    assert out["utilisation"].iloc[0] == RATIO_CAP
    assert out["spend_to_limit"].iloc[0] == RATIO_CAP
    assert out["high_utilisation_flag"].iloc[0] == 1
    assert np.isfinite(out.select_dtypes("number").to_numpy()).all()


def test_zero_transactions_does_not_divide_by_zero(builder):
    out = builder.transform(_row(Num_Transactions_Monthly=0, Monthly_Spend=450.0))
    assert out["spend_per_transaction"].iloc[0] == pytest.approx(450.0)


def test_age_bands(builder):
    ages = {22: "under_30", 30: "30_44", 44: "30_44", 45: "45_59", 60: "60_plus", 69: "60_plus"}
    for age, band in ages.items():
        assert builder.transform(_row(Age=age))["age_band"].iloc[0] == band


def test_missing_values_imputed_with_training_median_and_flagged(builder, customers):
    medians = customers[["Credit_Limit", "Current_Balance", "Payment_Delay_Days"]].median()
    out = builder.transform(_row(Credit_Limit=np.nan, Current_Balance=np.nan, Payment_Delay_Days=np.nan))
    assert out["record_incomplete"].iloc[0] == 1
    assert out["Credit_Limit"].iloc[0] == pytest.approx(medians["Credit_Limit"])
    assert out["Payment_Delay_Days"].iloc[0] == pytest.approx(medians["Payment_Delay_Days"])
    assert not out.isna().any().any()


def test_imputation_statistics_learned_only_in_fit(cfg, customers):
    X = customers.drop(columns=["High_Risk_Customer"])
    b = build_feature_builder(cfg).fit(X)
    before = dict(b.fill_values_)
    shifted = X.assign(Credit_Limit=X["Credit_Limit"] * 100)
    b.transform(shifted)
    assert b.fill_values_ == before


def test_customer_id_is_never_a_feature(builder):
    out = builder.transform(_row())
    assert "Customer_ID" not in out.columns
    a = builder.transform(_row(Customer_ID="CC-1"))
    b = builder.transform(_row(Customer_ID="ZZ-99999"))
    pd.testing.assert_frame_equal(a, b)


def test_target_is_rejected_to_prevent_leakage(builder):
    with pytest.raises(ValueError, match="must not be passed"):
        builder.transform(_row(High_Risk_Customer=1))


def test_missing_required_column_raises(builder):
    with pytest.raises(ValueError, match="missing required columns"):
        builder.transform(_row().drop(columns=["Monthly_Spend"]))


def test_drop_features_removes_columns(cfg, customers):
    c = cfg.with_overrides(**{"features.drop_features": ["late_payer_flag", "credit_stress_index"]})
    out = build_feature_builder(c).fit_transform(customers.drop(columns=["High_Risk_Customer"]))
    assert "late_payer_flag" not in out and "credit_stress_index" not in out


def test_config_change_changes_behaviour_without_code_change(cfg, customers):
    X = customers.drop(columns=["High_Risk_Customer"])
    low = build_feature_builder(cfg.with_overrides(**{"features.utilisation_risk_cutoff": 0.1})).fit_transform(X)
    high = build_feature_builder(cfg.with_overrides(**{"features.utilisation_risk_cutoff": 1.5})).fit_transform(X)
    assert low["high_utilisation_flag"].sum() > high["high_utilisation_flag"].sum()


def test_transform_is_deterministic_and_row_independent(builder, customers):
    X = customers.drop(columns=["High_Risk_Customer"])
    full = builder.transform(X)
    single = builder.transform(X.iloc[[7]])
    pd.testing.assert_frame_equal(full.iloc[[7]], single)


def test_requires_dataframe(cfg):
    with pytest.raises(TypeError):
        CreditFeatureBuilder(numeric_columns=list(cfg.data.numeric_columns),
                             categorical_columns=["Income_Bracket"]).fit(np.zeros((3, 3)))
