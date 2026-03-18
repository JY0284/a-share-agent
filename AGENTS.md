# AGENTS.md — A-Share Agent

> This file provides context for AI coding agents (Copilot, Cursor, Claude, etc.) working on this repository.

## Ecosystem Overview

This repository is part of a **three-project ecosystem** for AI-powered Chinese A-share investment analysis:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Deployment Topology                          │
│                                                                     │
│  stock_data (Data Layer)     Duty: ingest, store & serve market data│
│  ─────────────────────────   Parquet + DuckDB, Tushare API          │
│         ▲                    ../stock_data (editable install)        │
│         │ Python API                                                │
│         │                                                           │
│  a-share-agent (AI Layer)    Duty: reasoning, tool-use, portfolio   │
│  ─────────────────────────   LangGraph 1.0, DeepSeek LLM           │
│  THIS REPO — port 2024      28 data tools + 10 util/profile tools  │
│         ▲                                                           │
│         │ HTTP/SSE (LangGraph protocol)                             │
│         │                                                           │
│  a-share-agent-chat-ui       Duty: auth, billing, thread UI, proxy  │
│  ─────────────────────────   Next.js 15, SQLite, SSE streaming      │
│  ../a-share-agent-chat-ui    port 3000 → proxies to 2024            │
│         ▲                                                           │
│         │ HTTPS (Nginx reverse proxy)                               │
│         │                                                           │
│       Browser                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

| Project | Repo Path | Duty | Runtime |
|---------|-----------|------|---------|
| **stock_data** | `../stock_data` | Data ingestion, Parquet storage, DuckDB queries, web API | Editable Python dep |
| **a-share-agent** (this) | `.` | LLM agent: financial reasoning, tool orchestration, memory, backtesting | LangGraph server :2024 |
| **a-share-agent-chat-ui** | `../a-share-agent-chat-ui` | Auth, billing, thread ownership, SSE proxy, React chat UI | Next.js :3000 |
| **a-share-agent-traces** | `../a-share-agent-traces` | JSONL trace logs (written by this repo's `LocalTraceMiddleware`) | Passive storage |

**Data flows**: Browser → chat-ui (auth + billing) → this agent (reasoning + tools) → stock_data (market data).
**Cost flows**: DeepSeek API → this agent (usage_cost.py) → SSE stream → chat-ui (extractCostFromSseEvent) → SQLite deduction.

## What This Project Does

A-Share Agent is a conversational financial analysis assistant for the Chinese A-share stock market. It runs as a **LangGraph 1.0** agent backed by **DeepSeek LLM**, with 28 financial data tools, 10+ utility/profile tools, a Python sandbox, portfolio management, backtesting, and dual-layer memory (structured profile + soft vector memory).

## Quick Reference

| What | Where |
|------|-------|
| Agent entry point | `src/agent/graph.py` → exports `graph` |
| LangGraph config | `langgraph.json` |
| System prompt | `src/agent/prompts.py` |
| Financial data tools | `src/agent/tools.py` (28 tools in `ALL_TOOLS`) |
| Batch/portfolio tools | `src/agent/batch_tools.py` (4 tools) |
| Profile tool wrappers | `src/agent/profile_tools.py` (5 tools) |
| User profile models | `src/agent/user_profile.py` |
| Memory (mem0) | `src/agent/memory.py` (3 tools, optional) |
| Memory middleware | `src/agent/memory_middleware.py` |
| Todo middleware | `src/agent/todo_middleware.py` (InvestmentTodoMiddleware) |
| Python sandbox | `src/agent/sandbox.py` |
| Backtest engine | `src/agent/backtest.py` |
| Skills (24 dirs) | `skills/<name>/index.md` |
| Strategies (11 dirs) | `strategies/<name>/index.md` |
| Tests | `tests/` |
| Dependencies | `pyproject.toml` |
| Docker (prod) | `docker-compose.yml` |
| Dev server script | `scripts/Start-LangGraphServer.ps1` |

## Commands

```bash
uv sync                                          # install all deps
uv run pytest tests/ -v --tb=short               # run tests (always do this after changes)
uv run langgraph dev --port 2024 --no-browser     # start dev server
docker compose up -d                              # start prod (PostgreSQL + Redis)
```

## Architecture

```
Chat UI (Next.js :3000) ──► LangGraph Server (:2024) ──► stock_data (Parquet/DuckDB)
                                    │
                              Middleware Stack:
                              1. MemoryMiddleware (profile + mem0)
                              2. SkillInjectionMiddleware
                              3. LocalTraceMiddleware
                              4. PythonGuardMiddleware
                              5. InvestmentTodoMiddleware
```

- **LLM**: `deepseek-chat` via `langchain-deepseek`, temperature 0.1
- **Data**: `stock_data` package (editable, at `../stock_data`) — Parquet files + DuckDB
- **Memory Layer 1**: `UserProfile` — Pydantic models persisted as JSON in `data/user_profiles/`
- **Memory Layer 2**: `mem0` — Qdrant on-disk vector store in `data/mem0_qdrant/`, Chinese embeddings (`BAAI/bge-small-zh-v1.5`)
- **Figures**: Matplotlib output captured and stored per-thread in `assets/`
- **Traces**: JSONL logs written to `../a-share-agent-traces/`

### Middleware Details

| # | Middleware | Module | Purpose |
|---|-----------|--------|---------|
| 1 | `MemoryMiddleware` | `memory_middleware.py` | Loads `UserProfile` (portfolio, prefs, watchlist) + top mem0 memories into system message |
| 2 | `SkillInjectionMiddleware` | `skill_injection_middleware.py` | Semantic-matches user query to skills, injects relevant skill content |
| 3 | `LocalTraceMiddleware` | `trace_middleware.py` | Records model_start/model_end events + attaches cost via `usage_cost.py` |
| 4 | `PythonGuardMiddleware` | `python_guard_middleware.py` | Validates sandboxed Python execution requests |
| 5 | `InvestmentTodoMiddleware` | `todo_middleware.py` | Investment-tuned task planning (injects `write_todos` tool) |

**InvestmentTodoMiddleware** replaces the stock `TodoListMiddleware`. Key differences:
- Prompt is ~⅓ the token length (investment-specific, not generic)
- Only activates for ≥3-step tasks (avoids wasting tool calls on simple queries)
- Supports `cancelled` status alongside `completed`/`in_progress`/`not_started`
- Same `write_todos` tool name + `PlanningState` schema → full frontend compatibility

### Tool Inventory

| Category | Tools | Module |
|----------|-------|--------|
| Composite (high-level) | `tool_stock_snapshot`, `tool_smart_search`, `tool_peer_comparison`, `tool_search_and_load_skill`, `tool_backtest_strategy` | `tools.py` |
| Discovery | `tool_list_industries`, `tool_get_universe` | `tools.py` |
| Simple Data | `tool_get_daily_prices`, `tool_get_daily_basic`, `tool_get_index_daily_prices`, `tool_get_etf_daily_prices`, `tool_get_fund_nav`, `tool_get_income`, `tool_get_balancesheet`, `tool_get_cashflow`, `tool_get_fina_indicator`, `tool_get_dividend` | `tools.py` |
| Market Extras | `tool_get_moneyflow`, `tool_get_fx_daily` | `tools.py` |
| Macro | `tool_get_lpr`, `tool_get_cpi`, `tool_get_cn_sf`, `tool_get_cn_m` | `tools.py` |
| Calendar | `tool_get_trading_days`, `tool_is_trading_day`, `tool_get_prev_trade_date`, `tool_get_next_trade_date` | `tools.py` |
| Python Execution | `tool_execute_python` | `tools.py` |
| Batch/Portfolio | `tool_batch_quotes`, `tool_portfolio_live_snapshot`, `tool_market_overview`, `tool_compare_stocks` | `batch_tools.py` |
| Profile Management | `tool_update_portfolio`, `tool_update_preferences`, `tool_add_watchlist`, `tool_remove_watchlist`, `tool_add_strategy` | `profile_tools.py` |
| Memory (optional) | `tool_memory_search`, `tool_memory_save`, `tool_memory_list` | `memory.py` |
| Web Search (optional) | `tool_web_search` | `web_search.py` |
| Planning (middleware) | `write_todos` | `todo_middleware.py` |

**`tool_stock_snapshot`** is the most-used tool — it resolves a stock name/code, fetches price data, company info, and valuation. It supports **stocks, ETFs, funds, and indices** via a fallback chain: `search_stocks` → `get_fund_basic` → `get_index_basic`. Pricing and company lookups branch by `_asset_type`.

### System Prompt Rules

The system prompt (`prompts.py`) includes 10 rules:

1. Respond in user's language
2. Use tools before answering financial questions
3. Cite specific numbers with sources
4. Portfolio-aware reasoning
5. Risk warnings for leveraged/volatile products
6. Skill search for complex topics
7. Calendar awareness (trading days, market hours)
8. **Empty-data rule**: if a data tool returns empty/missing data, say so honestly — never fabricate numbers
9. **Freshness rule**: for non-snapshot tools, state the data date range and warn if >5 trading days stale
10. **Confidence degradation**: when data is partial or stale, explicitly lower confidence and suggest alternatives

Rules 8–10 prevent hallucination when data sources are incomplete.

### Portfolio Merge Mode

`tool_update_portfolio` now defaults to `mode="merge"`:
- **merge** (default): adds new positions, updates existing ones; numeric fields only overwrite if non-zero
- **replace**: full replacement (old behavior), used when user explicitly says "set my portfolio to exactly…"

## Billing / Cost Pipeline

Cost tracking flows through these stages:

```
DeepSeek API response (usage_metadata)
  → trace_middleware._attach_usage_cost()
      → usage_cost.extract_usage()        # normalise token counts
      → usage_cost.estimate_cost()        # RMB cost via pricing.json
  → writes cost into AIMessage.additional_kwargs & response_metadata
  → trace_middleware writes model_end event (JSONL trace)
  → LangGraph serialises messages into SSE stream
  → chat-ui extractCostFromSseEvent() sums costs from all AI messages
  → chat-ui deductCost() subtracts from SQLite balance
```

- **Pricing config**: `pricing.json` — DeepSeek cache_hit/cache_miss/output rates in RMB.
- **Cache**: `load_pricing()` reads the file once and caches in `_PRICING_CACHE`. The cache is
  pre-loaded at import time to avoid synchronous file I/O inside async middleware (see Gotchas).
- **Cost attachment**: `_attach_usage_cost()` mutates the AIMessage in-place, adding
  `additional_kwargs.cost.total` and `response_metadata.cost.total` (both in RMB).
- **Overdraft protection**: the chat-ui `deductCost()` uses an atomic SQL guard
  (`WHERE balance >= cost`) so concurrent requests cannot push the balance below zero.
  The proxy also enforces a per-user concurrency limit (1 active run, HTTP 429).

## Source Layout

```
a-share-agent/
├── src/agent/                  # Main package (hatch builds from here)
│   ├── graph.py                # ★ ENTRY POINT — creates the LangGraph agent
│   ├── prompts.py              # System prompt generation (~430 lines, 10 rules)
│   ├── tools.py                # 28 LangChain tools wrapping stock_data
│   ├── batch_tools.py          # Batch data tools (portfolio snapshot, market overview)
│   ├── user_profile.py         # Pydantic models + JSON persistence (merge mode)
│   ├── profile_tools.py        # LangChain tool wrappers for profile CRUD
│   ├── memory.py               # mem0 memory tools (Qdrant vector store)
│   ├── memory_middleware.py     # Dual-layer context injection
│   ├── skills.py               # Loads skill YAML+MD from skills/ directory
│   ├── skill_injection_middleware.py
│   ├── python_guard_middleware.py
│   ├── sandbox.py              # Sandboxed exec() with matplotlib capture
│   ├── backtest.py             # Built-in backtest engine
│   ├── todo_middleware.py      # Investment-tuned task planning (InvestmentTodoMiddleware)
│   ├── trace_middleware.py     # Local JSONL trace recorder
│   ├── trace.py                # Trace data models
│   ├── figures.py              # Thread-scoped figure storage
│   ├── web_search.py           # Tavily web search tools
│   ├── usage_cost.py           # Token usage + cost estimation
│   └── routines/
│       └── daily_briefing.py   # Personalized daily report generator
├── skills/                     # 24 skill directories (YAML frontmatter + MD)
├── strategies/                 # 11 strategy knowledge directories
├── routines/                   # Agent routine definitions (e.g. trace_failure_triage)
├── tests/                      # pytest test files
├── scripts/                    # Utility scripts + PowerShell server launcher
├── langgraph.json              # LangGraph graph entrypoint config
├── pyproject.toml              # Project deps + build config
├── docker-compose.yml          # Production: PostgreSQL + Redis + API
├── pricing.json                # DeepSeek token pricing (RMB)
└── main.py                     # Local test entry point
```

## Key Patterns to Follow

1. **`from __future__ import annotations`** in every module.
2. **`@tool` decorator** for all LangChain tools. Extract `user_id` from `config["configurable"]`.
3. **`logging.getLogger(__name__)`** — never `print()`.
4. **Atomic file writes**: use `tempfile.mkstemp()` + `os.replace()` for any user data.
5. **Mock all I/O in tests**: stock_data functions, mem0, file system. Use `tmp_path`.
6. **Reset `up._profile_dir_ensured = None`** in test fixtures that patch `USER_PROFILE_DIR`.
7. **Decimal ratios** for percentages (0.05 = 5%), not raw percent values.
8. **Bilingual**: agent responds in user's language (Chinese or English).
9. **Tool docs in `prompts.py`** must match actual tool signatures exactly — param names, types, descriptions.
10. **mem0 is best-effort**: always wrap in try/except, never let it crash the agent.
11. **Empty data → honest response**: if a tool returns no data, tell the user; never fabricate numbers.
12. **Portfolio merge by default**: `tool_update_portfolio(mode="merge")` — only use `"replace"` when user explicitly requests a full reset.

## Environment Variables (`.env`)

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `DEEPSEEK_API_KEY` | **Yes** | — | LLM API key |
| `STOCK_DATA_STORE_DIR` | No | `../stock_data/store` | Parquet data store path |
| `TAVILY_API_KEY` | No | — | Enables web search tools |
| `WEB_SEARCH_ENABLED` | No | `false` | Toggle web search |
| `MEM0_ENABLED` | No | `true` | Toggle mem0 soft memory |
| `OPENAI_API_KEY` | No | — | Fallback embedding provider for mem0 |
| `DEFAULT_USER_ID` | No | `dev_user` | Local dev fallback user ID |
| `USER_PROFILE_DIR` | No | auto | Override profile JSON storage directory |
| `AGENT_ASSETS_DIR` | No | `./assets` | Figure storage directory |
| `AGENT_PRICING_PATH` | No | `pricing.json` | LLM cost estimation config |

## Adding Features

### New Tool
1. Define with `@tool` in the appropriate `*_tools.py` module.
2. Add to the module's `*_TOOLS` list.
3. Import in `graph.py` → `get_all_tools()`.
4. Document in `prompts.py` (category, params, usage guidance).
5. Write tests.

### New Skill
1. Create `skills/<name>/index.md` with YAML frontmatter (`name`, `description`, `tags`).
2. Auto-discovered by `skills.py`.

### New Strategy
1. Create `strategies/<name>/index.md` with strategy rules.
2. If backtestable, add to `backtest.py`.

## Common Gotchas

- `stock_data` lives at `../stock_data` (sibling repo, editable install) — data-layer changes go there.
- `prompts.py` is ~430 lines — tool param mismatches cause silent runtime failures.
- `SystemMessage.content_blocks` is read-only in LangChain — mock with `MagicMock()` in tests.
- Dev persistence uses pickle (`.langgraph_api/`), prod uses PostgreSQL — don't assume format.
- `pricing.json` uses RMB (not USD) for DeepSeek token costs.
- **BlockBuster & sync I/O in async middleware**: `langgraph dev` enables BlockBuster, which
  raises exceptions on synchronous file I/O inside `async` functions. `load_pricing()` reads
  `pricing.json` from disk — if the cache is cold and the first call happens inside
  `awrap_model_call`, BlockBuster kills it and `except Exception: pass` silently drops the
  entire cost pipeline + `model_end` trace event. Fix: `usage_cost.py` eagerly calls
  `load_pricing()` at import time, and `LocalTraceMiddleware.__init__` also warms the cache.
  **Any new code that reads files must either pre-load at import or use `asyncio.to_thread()`.**
- **Trace diagnosis**: if `model_end` events stop appearing in trace JSONL files but `model_start`
  events are present, the cost pipeline is crashing silently — check BlockBuster / sync I/O first.
- **ETF/Fund/Index snapshot**: `tool_stock_snapshot` uses a fallback chain (stock → fund → index).
  If you add a new asset type, update the fallback chain and the `_asset_type` branching for pricing.
- **FX data gaps**: Tushare FXCM data stopped mid-2023. `fx_daily` is in the TransientError guard
  set in `stock_data`, and empty dates are skipped. The agent prompt warns about stale data.

## Maintaining This File

`AGENTS.md` is the **single source of truth** — `.github/copilot-instructions.md` is a stub that links here. Only update this file.

| Change Type | What to Update |
|-------------|----------------|
| New tool / module / file | Add to Quick Reference table + Source Layout tree + Tool Inventory |
| New middleware | Update the numbered middleware stack in Architecture + Middleware Details table |
| New env variable | Add to Environment Variables table |
| New skill or strategy dir | Update counts ("24 skills", "11 strategies") |
| Dependency added/removed | Update tech stack in "What This Project Does" |
| New prompt rule | Update System Prompt Rules section |
| Test convention changed | Update Key Patterns |
| New gotcha discovered | Append to Common Gotchas |
| Cross-project change | Update Ecosystem Overview + notify sibling repos' AGENT.md |

**Rule of thumb**: if a future agent would waste time searching for something you just changed, document it here.

Skip updates for: routine bug fixes, internal refactors with no API changes, or prompt wording tweaks.
