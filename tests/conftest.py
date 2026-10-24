from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "base.yaml")


@pytest.fixture(scope="session")
def dev_cfg():
    return load_config(ROOT / "configs" / "dev.yaml")


def make_customers(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the real schema, so unit tests never depend on DVC data."""
    rng = np.random.default_rng(seed)
    limit = rng.uniform(1000, 25000, n).round(2)
    balance = (limit * rng.beta(2, 3, n)).round(2)
    delay = rng.poisson(4, n).astype(float)
    spend = rng.gamma(2, 500, n).round(2)
    logit = -5 + 6 * balance / limit + 0.3 * delay + spend / 1500
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "Customer_ID": [f"CC-{i:05d}" for i in range(1, n + 1)],
        "Age": rng.integers(21, 70, n),
        "Income_Bracket": rng.choice(["Low", "Medium", "High"], n),
        "Credit_Limit": limit,
        "Current_Balance": balance,
        "Monthly_Spend": spend,
        "Num_Transactions_Monthly": rng.integers(5, 28, n),
        "Payment_Delay_Days": delay,
        "Tenure_Months": rng.integers(1, 120, n),
        "High_Risk_Customer": y,
    })
    miss = rng.choice(n, size=n // 30, replace=False)
    df.loc[miss, ["Credit_Limit", "Current_Balance", "Payment_Delay_Days"]] = np.nan
    return df


@pytest.fixture
def customers():
    return make_customers()
