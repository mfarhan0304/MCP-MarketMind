"""
MarketMind MCP server.

Exposes 4 tools via the Model Context Protocol using FastMCP:
  1. get_stock_quote         — latest price + change
  2. get_historical_prices   — OHLC bars
  3. compute_rsi             — RSI indicator
  4. generate_research_report — full AI-synthesised report (orchestrates 1–3 internally)

Run:
    uv run marketmind                  # stdio transport (Claude Desktop)
    uv run fastmcp dev src/marketmind/server.py   # browser inspector
"""

from typing import Annotated, Literal

from fastmcp import Context, FastMCP
from pydantic import Field

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

mcp = FastMCP(
    name="MarketMind",
    instructions=(
        "Financial research tools. Use these to gather market data, compute "
        "technical indicators, and generate AI research reports for stocks. "
        "All symbols must be valid US equity tickers (e.g. NVDA, AAPL, MSFT)."
    ),
)


# ---------------------------------------------------------------------------
# Tool: get_stock_quote
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_stock_quote(
    symbol: Annotated[str, Field(description="Stock ticker symbol, e.g. NVDA")],
) -> StockQuote:
    """Get the latest price, daily change %, and trading volume for a stock."""
    return await tools.get_stock_quote(GetStockQuoteInput(symbol=symbol))


# ---------------------------------------------------------------------------
# Tool: get_historical_prices
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_historical_prices(
    symbol: Annotated[str, Field(description="Stock ticker symbol")],
    period: Annotated[
        Literal["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        Field(description="How far back to retrieve data (default 1mo)"),
    ] = "1mo",
    interval: Annotated[
        Literal["1d", "1wk", "1mo"],
        Field(description="Bar interval (default 1d)"),
    ] = "1d",
) -> HistoricalPricesResult:
    """Retrieve OHLC price history for a stock over a given period."""
    return await tools.get_historical_prices(
        GetHistoricalPricesInput(symbol=symbol, period=period, interval=interval)
    )


# ---------------------------------------------------------------------------
# Tool: compute_rsi
# ---------------------------------------------------------------------------


@mcp.tool()
async def compute_rsi(
    symbol: Annotated[str, Field(description="Stock ticker symbol")],
    period: Annotated[
        int,
        Field(description="RSI lookback period in days (default 14)", ge=2, le=100),
    ] = 14,
) -> RSIResult:
    """
    Compute the Relative Strength Index (RSI) for a stock.

    Returns the RSI value (0–100) and a signal:
    - overbought: RSI ≥ 70 (potential sell pressure)
    - oversold:   RSI ≤ 30 (potential buy opportunity)
    - neutral:    RSI between 30 and 70
    """
    return await tools.compute_rsi(ComputeRSIInput(symbol=symbol, period=period))


# ---------------------------------------------------------------------------
# Tool: generate_research_report
# ---------------------------------------------------------------------------


@mcp.tool()
async def generate_research_report(
    symbol: Annotated[
        str,
        Field(description="Stock ticker to research, e.g. NVDA"),
    ],
    ctx: Context,
) -> str:
    """
    Generate a streaming AI research report for a stock.

    Fetches the latest quote, RSI(14), price history, and recent news
    concurrently, then streams a narrative analyst report token-by-token via
    MCP progress notifications — so the client sees output as it is written
    rather than waiting for the full response.

    Returns the complete report text when finished.
    """
    import asyncio

    # Step 1: gather all market data concurrently (tolerates partial failures)
    await ctx.report_progress(0, 3, "Gathering market data…")

    quote, history, rsi, headlines = await asyncio.gather(
        tools.get_stock_quote(GetStockQuoteInput(symbol=symbol)),
        tools.get_historical_prices(GetHistoricalPricesInput(symbol=symbol, period="1mo")),
        tools.compute_rsi(ComputeRSIInput(symbol=symbol)),
        tools.fetch_headlines_safe(symbol),
        return_exceptions=True,
    )

    inp = GenerateReportInput(
        symbol=symbol,
        quote=quote if not isinstance(quote, Exception) else None,
        history=history if not isinstance(history, Exception) else None,
        rsi=rsi if not isinstance(rsi, Exception) else None,
        headlines=headlines if not isinstance(headlines, Exception) else [],
    )

    # Step 2: stream synthesis — each token is pushed to the client immediately
    await ctx.report_progress(1, 3, "Generating report…")

    full_report = ""
    async for chunk in tools.stream_research_report(inp):
        full_report += chunk
        await ctx.report_progress(2, 3, chunk)

    await ctx.report_progress(3, 3, "Done.")
    return full_report


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
