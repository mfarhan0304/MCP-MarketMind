"""
Tests for the LangGraph research workflow.

Coverage matrix:
  Business     — all nodes execute, state accumulates correctly, report is complete
  Security     — node errors are isolated and do not propagate as unhandled exceptions
  Convenience  — partial data still produces a report; errors are readable strings
  Goal         — LangGraph graph compiles and runs end-to-end (Bloomberg signal)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from marketmind.schemas import (
    HistoricalPricesResult,
    OHLCBar,
    RSIResult,
    ResearchReport,
    StockQuote,
)
from marketmind.workflow import ResearchState, research_graph

from datetime import date as date_type


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_quote():
    return StockQuote(
        symbol="NVDA",
        price=875.40,
        change_pct=2.12,
        volume=45_000_000,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def fake_history():
    return HistoricalPricesResult(
        symbol="NVDA",
        period="1mo",
        interval="1d",
        bars=[
            OHLCBar(
                date=date_type(2026, 3, 12),
                open=860.0,
                high=880.0,
                low=855.0,
                close=875.0,
                volume=40_000_000,
            )
        ],
    )


@pytest.fixture
def fake_rsi():
    return RSIResult(symbol="NVDA", rsi=64.2, signal="neutral", period=14)


@pytest.fixture
def fake_report():
    return ResearchReport(
        symbol="NVDA",
        summary="NVIDIA shows strong momentum driven by AI demand.",
        price_analysis="Price is up 2.1% today, testing resistance near $880.",
        momentum_analysis="RSI at 64.2, elevated but not yet overbought.",
        news_context="Data centre revenue beat expectations last quarter.",
        outlook="bullish",
        generated_at=datetime.now(timezone.utc),
    )


def _initial_state(symbol: str = "NVDA") -> ResearchState:
    return {
        "symbol": symbol,
        "quote": None,
        "history": None,
        "rsi": None,
        "headlines": [],
        "report": None,
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Business
# ---------------------------------------------------------------------------


class TestWorkflowBusiness:

    async def test_all_nodes_populate_state(
        self, mocker, fake_quote, fake_history, fake_rsi, fake_report
    ):
        mocker.patch("marketmind.tools.get_stock_quote", AsyncMock(return_value=fake_quote))
        mocker.patch("marketmind.tools.get_historical_prices", AsyncMock(return_value=fake_history))
        mocker.patch("marketmind.tools.compute_rsi", AsyncMock(return_value=fake_rsi))
        mocker.patch("marketmind.tools.fetch_headlines_safe", AsyncMock(return_value=["NVDA beats estimates"]))
        mocker.patch("marketmind.tools.generate_research_report", AsyncMock(return_value=fake_report))

        result = await research_graph.ainvoke(_initial_state())

        assert result["quote"] is not None
        assert result["history"] is not None
        assert result["rsi"] is not None
        assert result["headlines"] == ["NVDA beats estimates"]
        assert result["report"] is not None

    async def test_report_has_all_required_fields(
        self, mocker, fake_quote, fake_history, fake_rsi, fake_report
    ):
        mocker.patch("marketmind.tools.get_stock_quote", AsyncMock(return_value=fake_quote))
        mocker.patch("marketmind.tools.get_historical_prices", AsyncMock(return_value=fake_history))
        mocker.patch("marketmind.tools.compute_rsi", AsyncMock(return_value=fake_rsi))
        mocker.patch("marketmind.tools.fetch_headlines_safe", AsyncMock(return_value=[]))
        mocker.patch("marketmind.tools.generate_research_report", AsyncMock(return_value=fake_report))

        result = await research_graph.ainvoke(_initial_state())
        r = result["report"]

        assert r.symbol == "NVDA"
        assert r.summary
        assert r.price_analysis
        assert r.momentum_analysis
        assert r.news_context
        assert r.outlook in ("bullish", "neutral", "bearish")
        assert r.disclaimer

    async def test_report_symbol_matches_input(
        self, mocker, fake_quote, fake_history, fake_rsi, fake_report
    ):
        mocker.patch("marketmind.tools.get_stock_quote", AsyncMock(return_value=fake_quote))
        mocker.patch("marketmind.tools.get_historical_prices", AsyncMock(return_value=fake_history))
        mocker.patch("marketmind.tools.compute_rsi", AsyncMock(return_value=fake_rsi))
        mocker.patch("marketmind.tools.fetch_headlines_safe", AsyncMock(return_value=[]))
        mocker.patch("marketmind.tools.generate_research_report", AsyncMock(return_value=fake_report))

        result = await research_graph.ainvoke(_initial_state("NVDA"))
        assert result["report"].symbol == "NVDA"

    async def test_no_errors_on_happy_path(
        self, mocker, fake_quote, fake_history, fake_rsi, fake_report
    ):
        mocker.patch("marketmind.tools.get_stock_quote", AsyncMock(return_value=fake_quote))
        mocker.patch("marketmind.tools.get_historical_prices", AsyncMock(return_value=fake_history))
        mocker.patch("marketmind.tools.compute_rsi", AsyncMock(return_value=fake_rsi))
        mocker.patch("marketmind.tools.fetch_headlines_safe", AsyncMock(return_value=[]))
        mocker.patch("marketmind.tools.generate_research_report", AsyncMock(return_value=fake_report))

        result = await research_graph.ainvoke(_initial_state())
        assert result["errors"] == []


# ---------------------------------------------------------------------------
# Security — node failures must not crash the graph
# ---------------------------------------------------------------------------


class TestWorkflowSecurity:

    async def test_market_node_failure_captured_not_raised(self, mocker):
        """market_node errors are stored in state, not raised as exceptions."""
        mocker.patch("marketmind.tools.get_stock_quote", AsyncMock(side_effect=ValueError("API down")))
        mocker.patch("marketmind.tools.get_historical_prices", AsyncMock(side_effect=ValueError("API down")))
        mocker.patch("marketmind.tools.compute_rsi", AsyncMock(side_effect=ValueError("API down")))
        mocker.patch("marketmind.tools.fetch_headlines_safe", AsyncMock(return_value=[]))
        mocker.patch("marketmind.tools.generate_research_report", AsyncMock(side_effect=ValueError("No data")))

        # Must not raise — graph always returns a result
        result = await research_graph.ainvoke(_initial_state("FAKE"))

        assert isinstance(result["errors"], list)
        assert len(result["errors"]) > 0

    async def test_synthesis_failure_returns_none_report(self, mocker, fake_quote, fake_history, fake_rsi):
        mocker.patch("marketmind.tools.get_stock_quote", AsyncMock(return_value=fake_quote))
        mocker.patch("marketmind.tools.get_historical_prices", AsyncMock(return_value=fake_history))
        mocker.patch("marketmind.tools.compute_rsi", AsyncMock(return_value=fake_rsi))
        mocker.patch("marketmind.tools.fetch_headlines_safe", AsyncMock(return_value=[]))
        mocker.patch("marketmind.tools.generate_research_report", AsyncMock(side_effect=RuntimeError("LLM unavailable")))

        result = await research_graph.ainvoke(_initial_state())

        assert result["report"] is None
        assert any("synthesis_node" in e for e in result["errors"])

    async def test_error_messages_are_strings_not_exceptions(self, mocker):
        mocker.patch("marketmind.tools.get_stock_quote", AsyncMock(side_effect=ValueError("bad symbol")))
        mocker.patch("marketmind.tools.get_historical_prices", AsyncMock(side_effect=ValueError("bad symbol")))
        mocker.patch("marketmind.tools.compute_rsi", AsyncMock(side_effect=ValueError("bad symbol")))
        mocker.patch("marketmind.tools.fetch_headlines_safe", AsyncMock(return_value=[]))
        mocker.patch("marketmind.tools.generate_research_report", AsyncMock(side_effect=ValueError("no data")))

        result = await research_graph.ainvoke(_initial_state())

        for err in result["errors"]:
            assert isinstance(err, str), f"Expected str, got {type(err)}: {err}"


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


class TestWorkflowConvenience:

    async def test_empty_headlines_does_not_break_synthesis(
        self, mocker, fake_quote, fake_history, fake_rsi, fake_report
    ):
        mocker.patch("marketmind.tools.get_stock_quote", AsyncMock(return_value=fake_quote))
        mocker.patch("marketmind.tools.get_historical_prices", AsyncMock(return_value=fake_history))
        mocker.patch("marketmind.tools.compute_rsi", AsyncMock(return_value=fake_rsi))
        mocker.patch("marketmind.tools.fetch_headlines_safe", AsyncMock(return_value=[]))
        mocker.patch("marketmind.tools.generate_research_report", AsyncMock(return_value=fake_report))

        result = await research_graph.ainvoke(_initial_state())
        assert result["report"] is not None

    async def test_partial_data_still_reaches_synthesis(self, mocker, fake_quote, fake_report):
        """If only quote succeeds, synthesis should still be called."""
        mocker.patch("marketmind.tools.get_stock_quote", AsyncMock(return_value=fake_quote))
        mocker.patch("marketmind.tools.get_historical_prices", AsyncMock(side_effect=ValueError("timeout")))
        mocker.patch("marketmind.tools.compute_rsi", AsyncMock(side_effect=ValueError("timeout")))
        mocker.patch("marketmind.tools.fetch_headlines_safe", AsyncMock(return_value=[]))
        mocker.patch("marketmind.tools.generate_research_report", AsyncMock(return_value=fake_report))

        result = await research_graph.ainvoke(_initial_state())

        # Quote made it through
        assert result["quote"] is not None
        # Synthesis still ran
        assert result["report"] is not None


# ---------------------------------------------------------------------------
# Goal — graph structure
# ---------------------------------------------------------------------------


class TestWorkflowGoal:

    def test_graph_compiles(self):
        """research_graph must be a compiled LangGraph — verifies Bloomberg stack is wired."""
        assert research_graph is not None

    def test_graph_has_correct_nodes(self):
        node_names = set(research_graph.nodes.keys())
        assert {"market", "news", "synthesis"}.issubset(node_names)
