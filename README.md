# MarketMind MCP

A financial research MCP server built with **FastMCP**, **Pydantic v2**, and **LangGraph**.

Exposes 4 tools over the Model Context Protocol that any MCP-compatible client (Claude Desktop, Cursor, etc.) can call. Includes a LangGraph workflow that chains the tools into a single research report.

## Tools

| Tool | Description |
|---|---|
| `get_stock_quote` | Latest price, change %, and volume for a ticker |
| `get_historical_prices` | OHLC price history for a given period |
| `compute_rsi` | RSI indicator with overbought/oversold classification |
| `generate_research_report` | LLM-synthesized research summary from collected data |

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- OpenAI API key (for the synthesis tool)

## Setup

```bash
git clone https://github.com/your-username/marketmind-mcp
cd marketmind-mcp
uv sync
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
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

### Run the research workflow directly

```bash
uv run python -m marketmind.workflow NVDA
```

Example output:

```
Researching NVDA...

[market_node]  quote: $875.40 (+2.1%)  RSI: 64.2 (neutral)
[news_node]    5 headlines fetched
[synthesis_node] generating report...

--- Research Report: NVDA ---
NVIDIA is showing moderate bullish momentum with RSI at 64.2,
below overbought territory. Recent headlines highlight strong
data center demand...
```

### Inspect tools interactively

```bash
uv run fastmcp dev src/marketmind/server.py
```

Opens the FastMCP inspector in your browser — lets you call any tool manually.

## Run Tests

```bash
uv run pytest
```

## Stack

| Layer | Technology |
|---|---|
| MCP server | `fastmcp` |
| Schema validation | `pydantic` v2 |
| Orchestration | `langgraph` |
| LLM | `anthropic` (Claude) |
| Market data | `yfinance` |
| Technical indicators | `pandas-ta` |
