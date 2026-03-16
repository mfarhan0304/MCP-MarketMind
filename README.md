# MarketMind MCP

A financial research MCP server built with **FastMCP**, **Pydantic v2**, and **LangGraph**.

Exposes 4 tools over the Model Context Protocol that any MCP-compatible client (Claude Desktop, Cursor, etc.) can call. Includes a LangGraph workflow that chains the tools into a multi-step research pipeline.

## Tools

| Tool | Description |
|---|---|
| `get_stock_quote` | Latest price, change %, and volume for a ticker |
| `get_historical_prices` | OHLC price history for a given period |
| `compute_rsi` | RSI indicator with overbought/oversold classification |
| `generate_research_report` | Streams an AI analyst report token-by-token via MCP progress notifications |

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- OpenAI API key (for `generate_research_report`)

## Setup

```bash
git clone https://github.com/your-username/marketmind-mcp
cd marketmind-mcp
uv sync
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Usage

### Run as MCP server (connect to Claude Desktop)

```bash
uv run marketmind
```

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "marketmind": {
      "command": "uv",
      "args": ["run", "marketmind"],
      "cwd": "/path/to/marketmind-mcp"
    }
  }
}
```

Once connected, Claude Desktop can call all 4 tools. `generate_research_report`
streams the analyst report token-by-token via MCP progress notifications — the
report appears in real-time as it is written.

### Run the LangGraph workflow directly (CLI)

```bash
uv run python -m marketmind.workflow NVDA
```

Example output:

```
Researching NVDA...

[market_node]  NVDA  $924.18  (+2.51%)
[market_node]  RSI(14): 55.61  → neutral
[news_node]    5 headlines fetched

========================================================
  RESEARCH REPORT: NVDA
  Generated: 2026-03-16 05:57 UTC
========================================================

Summary
  NVIDIA continues to exhibit strong bullish momentum, driven by accelerating
  data centre GPU demand and positive earnings revisions.

Price Action
  NVDA is up 2.51% today at $924.18, breaking above the 20-day moving average
  on above-average volume.

Momentum
  RSI(14) at 55.6 signals healthy momentum without approaching overbought territory.

News Context
  Recent headlines highlight record Blackwell GPU shipments and expanded
  hyperscaler contracts.

Outlook: BULLISH

For informational purposes only. Not financial advice.
```

### Inspect tools interactively

```bash
uv run fastmcp inspect src/marketmind/server.py
```

Opens the FastMCP inspector in your browser — lets you call any tool manually
and inspect inputs, outputs, and schemas.

## Run Tests

```bash
uv run pytest -v
```

45 tests covering business correctness, input security, and workflow resilience.

## Stack

| Layer | Technology |
|---|---|
| MCP server | `fastmcp` 3.x |
| Schema validation | `pydantic` v2 |
| Orchestration | `langgraph` |
| LLM | `openai` (GPT-4o) |
| Market data | `yfinance` |
| News | Yahoo Finance RSS via `feedparser` + `httpx` |
