# MarketMind MCP — Design Document

A focused **MCP server** for financial research that demonstrates hands-on experience with Bloomberg's exact GenAI stack: MCP, Pydantic AI, and LangGraph.

Scope is intentionally minimal — built to be completable, demonstrable, and technically credible.

---

## 1. What It Is

Three components, all in one repo:

1. **MCP server** — 6 financial research tools exposed via the MCP protocol
2. **Pydantic models** — strict schema validation on every tool input and output
3. **LangGraph workflow** — one multi-step research pipeline that chains the tools

That's it.

---

## 2. The 6 Tools

All tools use `yfinance` for market data and SEC EDGAR for filings — both free, no API key required.

| Tool | What it does |
|---|---|
| `get_stock_quote` | Latest price, change %, volume |
| `get_historical_prices` | OHLC bars for a given period |
| `compute_rsi` | RSI indicator + overbought/oversold signal |
| `compute_macd` | MACD line, signal line, crossover classification |
| `fetch_company_news` | Recent headlines via RSS (Yahoo Finance feed) |
| `get_recent_sec_filings` | Latest 10-K/10-Q/8-K from SEC EDGAR |

---

## 3. Pydantic Schemas

Every tool has typed input and output models. Example:

```python
class ComputeRSIInput(BaseModel):
    symbol: str
    period: int = Field(14, ge=2, le=100)

class RSIResult(BaseModel):
    symbol: str
    rsi: float = Field(..., ge=0, le=100)
    signal: Literal["overbought", "neutral", "oversold"]
```

The MCP server derives its JSON Schema directly from these models — single source of truth.

---

## 4. LangGraph Research Workflow

One graph with three nodes that run sequentially:

```
User: "Research NVDA"
       │
       ▼
  [market_node]          — calls get_stock_quote, compute_rsi, compute_macd
       │
       ▼
  [news_node]            — calls fetch_company_news
       │
       ▼
  [synthesis_node]       — LLM call: summarize all collected data into a report
       │
       ▼
  Structured research report
```

State is a single `TypedDict` that accumulates results across nodes. The synthesis node receives the full state and generates a concise analyst-style summary.

---

## 5. Tech Stack

| What | Tool |
|---|---|
| MCP server | `fastmcp` |
| Schema validation | `pydantic` v2 |
| Orchestration | `langgraph` |
| LLM calls | `anthropic` SDK (Claude) |
| Market data | `yfinance` |
| Technical indicators | `pandas-ta` |
| SEC filings | `sec-edgar-api` |
| News | `feedparser` (Yahoo Finance RSS) |

---

## 6. Project Structure

```
marketmind-mcp/
├── src/marketmind/
│   ├── server.py          # MCP server, tool registration
│   ├── tools.py           # All 6 tool implementations
│   ├── schemas.py         # All Pydantic input/output models
│   └── workflow.py        # LangGraph research graph
├── tests/
│   ├── test_tools.py
│   └── test_workflow.py
├── pyproject.toml
└── README.md
```

---

## 7. Build Plan (1 week)

| Day | Work |
|---|---|
| 1 | `schemas.py` — all Pydantic models. `tools.py` — market data tools (quote, history) |
| 2 | `tools.py` — technical tools (RSI, MACD). Unit tests |
| 3 | `tools.py` — news + SEC filings. Unit tests |
| 4 | `server.py` — MCP server, register all tools, smoke test with MCP inspector |
| 5 | `workflow.py` — LangGraph graph, wire all nodes, end-to-end test |
| 6 | README, demo output, polish |
| 7 | Buffer / cleanup |
