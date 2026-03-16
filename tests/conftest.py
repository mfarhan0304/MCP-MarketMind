"""
Shared pytest fixtures for MarketMind tests.

Network calls (yfinance, httpx) are always mocked — tests must be fast and
offline. Integration tests that hit real APIs are explicitly marked and skipped
by default.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def fake_close_series() -> pd.Series:
    """90-day close price series with a slight uptrend (deterministic seed)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=90, freq="D")
    prices = 800.0 + np.cumsum(rng.normal(0, 2, 90)) + np.linspace(0, 20, 90)
    return pd.Series(prices, index=dates, name="Close")


@pytest.fixture
def fake_history_df(fake_close_series: pd.Series) -> pd.DataFrame:
    """Realistic OHLCV DataFrame built around fake_close_series."""
    rng = np.random.default_rng(42)
    close = fake_close_series
    n = len(close)
    return pd.DataFrame(
        {
            "Open": close.values * rng.uniform(0.99, 1.0, n),
            "High": close.values * rng.uniform(1.00, 1.02, n),
            "Low": close.values * rng.uniform(0.98, 1.00, n),
            "Close": close.values,
            "Volume": rng.integers(30_000_000, 60_000_000, n),
            "Dividends": 0,
            "Stock Splits": 0,
        },
        index=close.index,
    )


@pytest.fixture
def mock_ticker(mocker, fake_history_df: pd.DataFrame):
    """
    Mock yfinance.Ticker with realistic fast_info and history data.

    Patches yfinance.Ticker globally for the duration of each test.
    """
    mock = MagicMock()
    mock.fast_info.last_price = 875.40
    mock.fast_info.previous_close = 857.25
    mock.fast_info.last_volume = 42_000_000
    mock.fast_info.three_month_average_volume = 45_000_000
    mock.history.return_value = fake_history_df

    mocker.patch("yfinance.Ticker", return_value=mock)
    return mock
