# CLAUDE.md — MarketMind MCP

Instructions for Claude Code working in this project.

## Project Goal

Build a minimal but credible MCP server for financial research to support a Bloomberg Senior GenAI Platform Engineer job application. Scope is intentionally small. Do not expand it.

## Stack

- `fastmcp` — MCP server
- `pydantic` v2 — all tool I/O schemas
- `langgraph` — single 3-node research workflow
- `yfinance` — market data
- `pandas-ta` — technical indicators
- `anthropic` SDK — LLM calls in synthesis node
- `uv` — package manager

## Key Commands

```bash
# Install dependencies
uv sync

# Run the MCP server
uv run marketmind

# Run the LangGraph workflow directly
uv run python -m marketmind.workflow NVDA

# Run tests
uv run pytest

# Inspect MCP tools interactively
uv run fastmcp dev src/marketmind/server.py
```

## Project Structure

```
src/marketmind/
├── server.py      # MCP server entry point, tool registration
├── tools.py       # All 4 tool implementations
├── schemas.py     # All Pydantic input/output models
└── workflow.py    # LangGraph 3-node research graph
tests/
├── test_tools.py
└── test_workflow.py
```

## Coding Conventions

- All tool functions are `async`
- Every tool has a matching Pydantic input model and output model in `schemas.py`
- Tool functions accept the Pydantic input model and return the Pydantic output model
- No bare `dict` returns — always return a typed model
- Use `Field(...)` with a `description` on every schema field — these become the MCP tool descriptions
- Errors surface as raised exceptions, not error fields in the response

## The 4 Tools

| Tool | Input model | Output model |
|---|---|---|
| `get_stock_quote` | `GetStockQuoteInput` | `StockQuote` |
| `get_historical_prices` | `GetHistoricalPricesInput` | `HistoricalPricesResult` |
| `compute_rsi` | `ComputeRSIInput` | `RSIResult` |
| `generate_research_report` | `GenerateReportInput` | `ResearchReport` |

## The LangGraph Workflow

Three nodes, linear, no branching:

```
market_node → news_node → synthesis_node
```

- `market_node` — calls `get_stock_quote`, `get_historical_prices`, `compute_rsi`
- `news_node` — fetches Yahoo Finance RSS headlines for the symbol
- `synthesis_node` — single LLM call that receives all collected data and returns a structured report

State is `ResearchState` (TypedDict) defined in `workflow.py`.

## Constraints — Do Not Violate

- Do not add new tools beyond the 4 defined above
- Do not add portfolio, backtesting, or SEC filing tools
- Do not add a database, vector store, or caching layer
- Do not split tools into multiple files
- Do not add a FastAPI or HTTP layer — MCP is the only interface
- Keep `pyproject.toml` dependencies minimal
