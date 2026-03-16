"""
Tool implementations for MarketMind MCP.

All functions are async, accept a Pydantic input model, and return a Pydantic
output model. No bare dicts are returned — every boundary is typed and validated.

Data sources:
- yfinance     — market data (free, no API key)
- SEC EDGAR    — not used in MVP; reserved for extension
- Yahoo Finance RSS — news headlines (free, no API key)
- Anthropic    — LLM synthesis for generate_research_report
"""

import json
from datetime import datetime, timezone

import feedparser
import httpx
import pandas as pd
import yfinance as yf

from marketmind.schemas import (
    ComputeRSIInput,
    GenerateReportInput,
    GetHistoricalPricesInput,
    GetStockQuoteInput,
    HistoricalPricesResult,
    OHLCBar,
    RSIResult,
    ResearchReport,
    StockQuote,
)


# ---------------------------------------------------------------------------
# get_stock_quote
# ---------------------------------------------------------------------------


async def get_stock_quote(inp: GetStockQuoteInput) -> StockQuote:
    ticker = yf.Ticker(inp.symbol)
    info = ticker.fast_info

    price: float | None = info.last_price
    prev_close: float | None = info.previous_close

    if price is None or prev_close is None:
        raise ValueError(
            f"No market data found for '{inp.symbol}'. "
            "Check that the symbol is valid and markets are open."
        )

    change_pct = ((price - prev_close) / prev_close) * 100
    volume = int(info.last_volume or info.three_month_average_volume or 0)

    return StockQuote(
        symbol=inp.symbol,
        price=round(price, 2),
        change_pct=round(change_pct, 4),
        volume=volume,
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# get_historical_prices
# ---------------------------------------------------------------------------


async def get_historical_prices(inp: GetHistoricalPricesInput) -> HistoricalPricesResult:
    ticker = yf.Ticker(inp.symbol)
    hist: pd.DataFrame = ticker.history(period=inp.period, interval=inp.interval)

    if hist.empty:
        raise ValueError(
            f"No historical data found for '{inp.symbol}' "
            f"(period={inp.period}, interval={inp.interval})."
        )

    bars = [
        OHLCBar(
            date=idx.date(),
            open=round(float(row["Open"]), 4),
            high=round(float(row["High"]), 4),
            low=round(float(row["Low"]), 4),
            close=round(float(row["Close"]), 4),
            volume=int(row["Volume"]),
        )
        for idx, row in hist.iterrows()
    ]

    return HistoricalPricesResult(
        symbol=inp.symbol,
        period=inp.period,
        interval=inp.interval,
        bars=bars,
    )


# ---------------------------------------------------------------------------
# compute_rsi  (Wilder's smoothing — no external TA library needed)
# ---------------------------------------------------------------------------


def _rsi(close: pd.Series, period: int) -> float:
    """
    Compute RSI using Wilder's EMA smoothing method.

    Returns a float in [0, 100]. Returns 100.0 if there are no down-days
    in the window (pure uptrend), 0.0 if there are no up-days.
    """
    delta = close.diff().dropna()

    if len(delta) < period:
        raise ValueError(
            f"Need at least {period + 1} data points to compute RSI({period}), "
            f"got {len(delta) + 1}."
        )

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    last_avg_loss = float(avg_loss.iloc[-1])
    if last_avg_loss == 0:
        return 100.0

    rs = float(avg_gain.iloc[-1]) / last_avg_loss
    return round(100 - (100 / (1 + rs)), 2)


async def compute_rsi(inp: ComputeRSIInput) -> RSIResult:
    ticker = yf.Ticker(inp.symbol)
    hist: pd.DataFrame = ticker.history(period="3mo", interval="1d")

    if hist.empty:
        raise ValueError(f"No data found for '{inp.symbol}'.")

    rsi_value = _rsi(hist["Close"], inp.period)

    if rsi_value >= 70:
        signal = "overbought"
    elif rsi_value <= 30:
        signal = "oversold"
    else:
        signal = "neutral"

    return RSIResult(
        symbol=inp.symbol,
        rsi=rsi_value,
        signal=signal,
        period=inp.period,
    )


# ---------------------------------------------------------------------------
# _fetch_headlines  (private — not an MCP tool)
# ---------------------------------------------------------------------------


async def _fetch_headlines(symbol: str, max_items: int = 5) -> list[str]:
    """Fetch recent news headlines from Yahoo Finance RSS. Returns up to max_items titles."""
    url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={symbol}&region=US&lang=en-US"
    )
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.get(url)
        response.raise_for_status()
    feed = feedparser.parse(response.text)
    return [entry.title for entry in feed.entries[:max_items]]


async def fetch_headlines_safe(symbol: str, max_items: int = 5) -> list[str]:
    """Public wrapper around _fetch_headlines that swallows network errors."""
    try:
        return await _fetch_headlines(symbol, max_items)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# generate_research_report
# ---------------------------------------------------------------------------


async def stream_research_report(inp: GenerateReportInput):
    """
    Async generator that streams a narrative research report token-by-token
    from OpenAI.

    Used by the MCP server to push progress notifications to the client as the
    report is being written — so the user sees output immediately instead of
    waiting for the full response.

    Yields str chunks. The caller is responsible for assembling the full text.
    """
    import openai

    from marketmind.config import settings

    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    price_line = (
        f"${inp.quote.price} ({inp.quote.change_pct:+.2f}% today)"
        if inp.quote
        else "unavailable"
    )
    rsi_line = (
        f"{inp.rsi.rsi:.1f} — {inp.rsi.signal}" if inp.rsi else "unavailable"
    )
    headlines_block = (
        "\n".join(f"  • {h}" for h in inp.headlines)
        if inp.headlines
        else "  None available"
    )

    prompt = f"""You are a senior equity analyst. Write a concise, professional
research note for {inp.symbol} using the data below. Use plain prose — no JSON,
no markdown headers, no bullet points.

Data:
- Price: {price_line}
- RSI(14): {rsi_line}
- Recent headlines:
{headlines_block}

Structure: 1) price action, 2) momentum, 3) news context, 4) outlook.
Keep it under 150 words."""

    stream = await client.chat.completions.create(
        model="gpt-4o",
        max_tokens=300,
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def generate_research_report(inp: GenerateReportInput) -> ResearchReport:
    # Lazy import so that missing API key only fails when this tool is called.
    import openai

    from marketmind.config import settings

    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    price_line = (
        f"${inp.quote.price} ({inp.quote.change_pct:+.2f}% today)"
        if inp.quote
        else "unavailable"
    )
    rsi_line = (
        f"{inp.rsi.rsi:.1f} ({inp.rsi.signal})" if inp.rsi else "unavailable"
    )
    bars_line = (
        f"{len(inp.history.bars)} daily bars over {inp.history.period}"
        if inp.history
        else "unavailable"
    )
    headlines_block = (
        "\n".join(f"  • {h}" for h in inp.headlines)
        if inp.headlines
        else "  None available"
    )

    prompt = f"""You are a senior equity analyst. Write a concise research note for {inp.symbol}.

Available data:
- Price: {price_line}
- RSI (14): {rsi_line}
- Price history: {bars_line}
- Recent headlines:
{headlines_block}

Respond with a single JSON object — no markdown, no extra text:
{{
  "summary": "<2–3 sentence executive summary>",
  "price_analysis": "<1–2 sentences on price action>",
  "momentum_analysis": "<1–2 sentences on RSI / momentum>",
  "news_context": "<1–2 sentences on news impact>",
  "outlook": "<bullish | neutral | bearish>"
}}"""

    response = await client.chat.completions.create(
        model="gpt-4o",
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    data = json.loads(response.choices[0].message.content)

    return ResearchReport(
        symbol=inp.symbol,
        generated_at=datetime.now(timezone.utc),
        **data,
    )
