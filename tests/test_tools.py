"""
Tests for MarketMind tools.

Coverage matrix:
  Business     — financial computation correctness, data shape invariants
  Security     — input validation rejects malformed / injection inputs
  Convenience  — sensible defaults, normalisation, readable error messages
  Goal         — Pydantic models are enforced end-to-end (Bloomberg signal)
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from marketmind.schemas import (
    ComputeRSIInput,
    GetHistoricalPricesInput,
    GetStockQuoteInput,
)
from marketmind.tools import (
    _rsi,
    compute_rsi,
    get_historical_prices,
    get_stock_quote,
)


# ===========================================================================
# get_stock_quote
# ===========================================================================


class TestGetStockQuote:

    # --- Business -----------------------------------------------------------

    async def test_returns_valid_quote(self, mock_ticker):
        result = await get_stock_quote(GetStockQuoteInput(symbol="NVDA"))
        assert result.symbol == "NVDA"
        assert result.price == 875.40
        assert result.volume > 0
        assert result.timestamp is not None

    async def test_change_pct_calculation(self, mock_ticker):
        # (875.40 - 857.25) / 857.25 * 100 ≈ 2.1171 %
        result = await get_stock_quote(GetStockQuoteInput(symbol="NVDA"))
        expected = ((875.40 - 857.25) / 857.25) * 100
        assert abs(result.change_pct - expected) < 0.01

    async def test_positive_change_when_price_above_prev_close(self, mock_ticker):
        result = await get_stock_quote(GetStockQuoteInput(symbol="NVDA"))
        assert result.change_pct > 0

    # --- Convenience --------------------------------------------------------

    async def test_symbol_is_uppercased(self, mock_ticker):
        result = await get_stock_quote(GetStockQuoteInput(symbol="nvda"))
        assert result.symbol == "NVDA"

    async def test_mixed_case_symbol_normalised(self, mock_ticker):
        result = await get_stock_quote(GetStockQuoteInput(symbol="Nvda"))
        assert result.symbol == "NVDA"

    # --- Security -----------------------------------------------------------

    def test_empty_symbol_rejected(self):
        with pytest.raises(ValidationError):
            GetStockQuoteInput(symbol="")

    def test_symbol_too_long_rejected(self):
        with pytest.raises(ValidationError):
            GetStockQuoteInput(symbol="A" * 16)

    def test_symbol_with_slash_rejected(self):
        with pytest.raises(ValidationError):
            GetStockQuoteInput(symbol="../../etc/passwd")

    def test_symbol_with_sql_injection_rejected(self):
        with pytest.raises(ValidationError):
            GetStockQuoteInput(symbol="'; DROP TABLE stocks;--")

    def test_symbol_with_whitespace_only_rejected(self):
        with pytest.raises(ValidationError):
            GetStockQuoteInput(symbol="   ")

    # --- Goal: clear error, not raw yfinance exception ----------------------

    async def test_missing_price_raises_value_error_with_message(self, mocker):
        mock = mocker.MagicMock()
        mock.fast_info.last_price = None
        mock.fast_info.previous_close = None
        mocker.patch("yfinance.Ticker", return_value=mock)

        with pytest.raises(ValueError, match="No market data found"):
            await get_stock_quote(GetStockQuoteInput(symbol="FAKE"))


# ===========================================================================
# get_historical_prices
# ===========================================================================


class TestGetHistoricalPrices:

    # --- Business -----------------------------------------------------------

    async def test_returns_bars(self, mock_ticker):
        result = await get_historical_prices(
            GetHistoricalPricesInput(symbol="NVDA", period="1mo")
        )
        assert result.symbol == "NVDA"
        assert len(result.bars) > 0

    async def test_ohlc_high_gte_low(self, mock_ticker):
        result = await get_historical_prices(
            GetHistoricalPricesInput(symbol="NVDA", period="1mo")
        )
        for bar in result.bars:
            assert bar.high >= bar.low, f"high={bar.high} < low={bar.low}"

    async def test_ohlc_volume_non_negative(self, mock_ticker):
        result = await get_historical_prices(
            GetHistoricalPricesInput(symbol="NVDA", period="1mo")
        )
        for bar in result.bars:
            assert bar.volume >= 0

    async def test_period_and_interval_echoed(self, mock_ticker):
        result = await get_historical_prices(
            GetHistoricalPricesInput(symbol="NVDA", period="3mo", interval="1wk")
        )
        assert result.period == "3mo"
        assert result.interval == "1wk"

    # --- Convenience --------------------------------------------------------

    async def test_default_interval_is_daily(self, mock_ticker):
        result = await get_historical_prices(
            GetHistoricalPricesInput(symbol="NVDA", period="1mo")
        )
        assert result.interval == "1d"

    # --- Security -----------------------------------------------------------

    def test_invalid_period_rejected(self):
        with pytest.raises(ValidationError):
            GetHistoricalPricesInput(symbol="NVDA", period="10y")

    def test_invalid_interval_rejected(self):
        with pytest.raises(ValidationError):
            GetHistoricalPricesInput(symbol="NVDA", interval="1s")

    # --- Goal: clear error on empty data ------------------------------------

    async def test_empty_history_raises_value_error(self, mocker):
        mock = mocker.MagicMock()
        mock.history.return_value = pd.DataFrame()
        mocker.patch("yfinance.Ticker", return_value=mock)

        with pytest.raises(ValueError, match="No historical data"):
            await get_historical_prices(GetHistoricalPricesInput(symbol="FAKE"))


# ===========================================================================
# _rsi  (pure function — no mocking needed)
# ===========================================================================


class TestRSICalculation:

    def _make_close(self, prices: list[float]) -> pd.Series:
        dates = pd.date_range(end="2026-01-01", periods=len(prices), freq="D")
        return pd.Series(prices, index=dates)

    # --- Business -----------------------------------------------------------

    def test_rsi_pure_uptrend_near_100(self):
        close = self._make_close(list(range(100, 160)))  # strict uptrend
        rsi = _rsi(close, 14)
        assert rsi >= 90, f"Expected RSI near 100 for pure uptrend, got {rsi}"

    def test_rsi_pure_downtrend_near_0(self):
        close = self._make_close(list(range(160, 100, -1)))  # strict downtrend
        rsi = _rsi(close, 14)
        assert rsi <= 10, f"Expected RSI near 0 for pure downtrend, got {rsi}"

    def test_rsi_always_in_0_100(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            prices = 100 + np.cumsum(rng.normal(0, 1, 60))
            close = self._make_close(prices.tolist())
            rsi = _rsi(close, 14)
            assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"

    def test_rsi_no_down_days_returns_100(self):
        close = self._make_close(list(range(100, 160)))  # monotone up
        assert _rsi(close, 14) == 100.0

    # --- Business: boundary classification ---------------------------------

    async def test_rsi_ge_70_is_overbought(self, mocker):
        dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=60, freq="D")
        close = pd.Series(np.linspace(100, 300, 60), index=dates)
        df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1_000_000})
        mock = mocker.MagicMock()
        mock.history.return_value = df
        mocker.patch("yfinance.Ticker", return_value=mock)

        result = await compute_rsi(ComputeRSIInput(symbol="NVDA"))
        assert result.rsi >= 70
        assert result.signal == "overbought"

    async def test_rsi_le_30_is_oversold(self, mocker):
        dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=60, freq="D")
        close = pd.Series(np.linspace(300, 100, 60), index=dates)
        df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1_000_000})
        mock = mocker.MagicMock()
        mock.history.return_value = df
        mocker.patch("yfinance.Ticker", return_value=mock)

        result = await compute_rsi(ComputeRSIInput(symbol="NVDA"))
        assert result.rsi <= 30
        assert result.signal == "oversold"

    async def test_rsi_between_30_70_is_neutral(self, mock_ticker):
        # The default fake_close_series has a slight uptrend → RSI in neutral zone
        result = await compute_rsi(ComputeRSIInput(symbol="NVDA"))
        assert result.signal == "neutral"
        assert 30 < result.rsi < 70

    # --- Convenience --------------------------------------------------------

    async def test_default_period_is_14(self, mock_ticker):
        result = await compute_rsi(ComputeRSIInput(symbol="NVDA"))
        assert result.period == 14

    # --- Security -----------------------------------------------------------

    def test_period_below_minimum_rejected(self):
        with pytest.raises(ValidationError):
            ComputeRSIInput(symbol="NVDA", period=1)

    def test_period_above_maximum_rejected(self):
        with pytest.raises(ValidationError):
            ComputeRSIInput(symbol="NVDA", period=101)

    def test_insufficient_data_raises_value_error(self):
        close = self._make_close([100.0, 101.0, 102.0])  # only 3 points, period=14
        with pytest.raises(ValueError, match="data points"):
            _rsi(close, 14)
