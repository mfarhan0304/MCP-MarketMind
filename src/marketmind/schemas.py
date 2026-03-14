"""
Pydantic v2 schemas for all MarketMind tool inputs and outputs.

These models are the single source of truth for:
- MCP tool JSON Schema (FastMCP derives it automatically)
- Runtime validation of all data entering and leaving the system
- LangGraph workflow state typing
"""

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared validator
# ---------------------------------------------------------------------------

_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]{1,15}$")


def _validate_symbol(v: str) -> str:
    """Normalise to uppercase and reject anything that isn't a valid ticker."""
    v = v.strip().upper()
    if not _SYMBOL_RE.match(v):
        raise ValueError(
            "Symbol must be 1–15 characters containing only letters, numbers, "
            "dots, or hyphens (e.g. NVDA, BRK.B, BF-B)."
        )
    return v


# ---------------------------------------------------------------------------
# get_stock_quote
# ---------------------------------------------------------------------------


class GetStockQuoteInput(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol, e.g. NVDA")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return _validate_symbol(v)


class StockQuote(BaseModel):
    symbol: str = Field(..., description="Ticker symbol")
    price: float = Field(..., description="Latest trade price (USD)")
    change_pct: float = Field(..., description="Daily change as a percentage")
    volume: int = Field(..., ge=0, description="Most recent trading volume")
    timestamp: datetime = Field(..., description="Time the quote was fetched (UTC)")


# ---------------------------------------------------------------------------
# get_historical_prices
# ---------------------------------------------------------------------------


class GetHistoricalPricesInput(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol")
    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y"] = Field(
        "1mo", description="How far back to retrieve data"
    )
    interval: Literal["1d", "1wk", "1mo"] = Field(
        "1d", description="Bar interval"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return _validate_symbol(v)


class OHLCBar(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(..., ge=0)


class HistoricalPricesResult(BaseModel):
    symbol: str
    period: str
    interval: str
    bars: list[OHLCBar]


# ---------------------------------------------------------------------------
# compute_rsi
# ---------------------------------------------------------------------------


class ComputeRSIInput(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol")
    period: int = Field(
        14,
        ge=2,
        le=100,
        description="RSI lookback period in days (default 14)",
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return _validate_symbol(v)


class RSIResult(BaseModel):
    symbol: str
    rsi: float = Field(..., ge=0, le=100, description="RSI value (0–100)")
    signal: Literal["overbought", "neutral", "oversold"] = Field(
        ..., description="overbought ≥ 70 | oversold ≤ 30 | neutral otherwise"
    )
    period: int


# ---------------------------------------------------------------------------
# generate_research_report  (internal input — not exposed as flat MCP params)
# ---------------------------------------------------------------------------


class GenerateReportInput(BaseModel):
    symbol: str
    quote: StockQuote | None = None
    history: HistoricalPricesResult | None = None
    rsi: RSIResult | None = None
    headlines: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    symbol: str
    summary: str = Field(..., description="2–3 sentence executive summary")
    price_analysis: str = Field(..., description="Price action commentary")
    momentum_analysis: str = Field(..., description="RSI / momentum commentary")
    news_context: str = Field(..., description="Recent news impact")
    outlook: Literal["bullish", "neutral", "bearish"]
    generated_at: datetime
    disclaimer: str = "For informational purposes only. Not financial advice."
