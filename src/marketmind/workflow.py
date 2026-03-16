"""
LangGraph research workflow for MarketMind.

Graph:  market_node → news_node → synthesis_node → END

Each node calls tools directly (not via MCP) and accumulates results into
ResearchState. Errors in any node are captured and do not crash the graph —
the synthesis node receives whatever data is available.

Usage (CLI):
    uv run python -m marketmind.workflow NVDA
"""

import asyncio
import operator
import sys
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

import marketmind.tools as tools
from marketmind.schemas import (
    ComputeRSIInput,
    GenerateReportInput,
    GetHistoricalPricesInput,
    GetStockQuoteInput,
    HistoricalPricesResult,
    RSIResult,
    ResearchReport,
    StockQuote,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ResearchState(TypedDict):
    symbol: str
    quote: StockQuote | None
    history: HistoricalPricesResult | None
    rsi: RSIResult | None
    headlines: list[str]
    report: ResearchReport | None
    errors: Annotated[list[str], operator.add]  # LangGraph appends across nodes


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def market_node(state: ResearchState) -> dict:
    """Fetch quote, price history, and RSI concurrently."""
    symbol = state["symbol"]
    quote = history = rsi = None
    errors: list[str] = []

    results = await asyncio.gather(
        tools.get_stock_quote(GetStockQuoteInput(symbol=symbol)),
        tools.get_historical_prices(GetHistoricalPricesInput(symbol=symbol, period="1mo")),
        tools.compute_rsi(ComputeRSIInput(symbol=symbol)),
        return_exceptions=True,
    )

    if isinstance(results[0], Exception):
        errors.append(f"market_node/quote: {results[0]}")
    else:
        quote = results[0]

    if isinstance(results[1], Exception):
        errors.append(f"market_node/history: {results[1]}")
    else:
        history = results[1]

    if isinstance(results[2], Exception):
        errors.append(f"market_node/rsi: {results[2]}")
    else:
        rsi = results[2]

    return {"quote": quote, "history": history, "rsi": rsi, "errors": errors}


async def news_node(state: ResearchState) -> dict:
    """Fetch recent headlines. Always succeeds — returns empty list on failure."""
    headlines = await tools.fetch_headlines_safe(state["symbol"])
    return {"headlines": headlines}


async def synthesis_node(state: ResearchState) -> dict:
    """Call the LLM to synthesise all collected data into a research report."""
    errors: list[str] = []

    try:
        report = await tools.generate_research_report(
            GenerateReportInput(
                symbol=state["symbol"],
                quote=state["quote"],
                history=state["history"],
                rsi=state["rsi"],
                headlines=state["headlines"],
            )
        )
    except Exception as e:
        errors.append(f"synthesis_node: {e}")
        report = None

    return {"report": report, "errors": errors}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


_builder = StateGraph(ResearchState)
_builder.add_node("market", market_node)
_builder.add_node("news", news_node)
_builder.add_node("synthesis", synthesis_node)
_builder.set_entry_point("market")
_builder.add_edge("market", "news")
_builder.add_edge("news", "synthesis")
_builder.add_edge("synthesis", END)

research_graph = _builder.compile()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


async def _run(symbol: str) -> None:
    print(f"\nResearching {symbol}...\n")

    initial: ResearchState = {
        "symbol": symbol.upper(),
        "quote": None,
        "history": None,
        "rsi": None,
        "headlines": [],
        "report": None,
        "errors": [],
    }

    result = await research_graph.ainvoke(initial)

    if result["quote"]:
        q = result["quote"]
        print(f"[market_node]  {q.symbol}  ${q.price}  ({q.change_pct:+.2f}%)")
    if result["rsi"]:
        r = result["rsi"]
        print(f"[market_node]  RSI({r.period}): {r.rsi}  → {r.signal}")
    if result["headlines"]:
        print(f"[news_node]    {len(result['headlines'])} headlines fetched")

    report = result["report"]
    if report:
        sep = "=" * 56
        print(f"\n{sep}")
        print(f"  RESEARCH REPORT: {report.symbol}")
        print(f"  Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        print(sep)
        print(f"\nSummary\n  {report.summary}")
        print(f"\nPrice Action\n  {report.price_analysis}")
        print(f"\nMomentum\n  {report.momentum_analysis}")
        print(f"\nNews Context\n  {report.news_context}")
        print(f"\nOutlook: {report.outlook.upper()}")
        print(f"\n{report.disclaimer}\n")
    else:
        print("\n[!] No report generated.")

    if result["errors"]:
        print("Errors encountered:")
        for err in result["errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    asyncio.run(_run(symbol))
