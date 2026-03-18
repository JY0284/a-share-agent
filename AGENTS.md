# AGENTS.md — A-Share Agent

> This file provides context for AI coding agents (Copilot, Cursor, Claude, etc.) working on this repository.

## What This Project Does

A-Share Agent is a conversational financial analysis assistant for the Chinese A-share stock market. It runs as a **LangGraph 1.0** agent backed by **DeepSeek LLM**, with 24+ financial data tools, a Python sandbox, portfolio management, backtesting, and dual-layer memory (structured profile + soft vector memory).

## Quick Reference

| What | Where |
|------|-------|
| Agent entry point | `src/agent/graph.py` → exports `graph` |
| LangGraph config | `langgraph.json` |
| System prompt | `src/agent/prompts.py` |
| Financial data tools | `src/agent/tools.py` (24+ tools) |
| Batch/portfolio tools | `src/agent/batch_tools.py` |
| User profile models | `src/agent/user_profile.py` |
| Profile tool wrappers | `src/agent/profile_tools.py` |
| Memory (mem0) | `src/agent/memory.py` |
| Memory middleware | `src/agent/memory_middleware.py` |
| Python sandbox | `src/agent/python_sandbox.py` |
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
                              5. TodoListMiddleware
```

- **LLM**: `deepseek-chat` via `langchain-deepseek`, temperature 0.1
- **Data**: `stock_data` package (editable, at `../stock_data`) — Parquet files + DuckDB
- **Memory Layer 1**: `UserProfile` — Pydantic models persisted as JSON in `data/user_profiles/`
- **Memory Layer 2**: `mem0` — Qdrant on-disk vector store in `data/mem0_qdrant/`, Chinese embeddings (`BAAI/bge-small-zh-v1.5`)
- **Figures**: Matplotlib output captured and stored per-thread in `assets/`
- **Traces**: JSONL logs written to `../a-share-agent-traces/`

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
│   ├── prompts.py              # System prompt generation (large, ~430 lines)
│   ├── tools.py                # 24+ LangChain tools wrapping stock_data
│   ├── batch_tools.py          # Batch data tools (portfolio snapshot, market overview)
│   ├── user_profile.py         # Pydantic models + JSON persistence
│   ├── profile_tools.py        # LangChain tool wrappers for profile CRUD
│   ├── memory.py               # mem0 memory tools (Qdrant vector store)
│   ├── memory_middleware.py     # Dual-layer context injection
│   ├── skill_manager.py        # Loads skill YAML+MD from skills/ directory
│   ├── skill_injection_middleware.py
│   ├── python_sandbox.py       # Sandboxed exec() with matplotlib capture
│   ├── python_guard_middleware.py
│   ├── backtest.py             # Built-in backtest engine
│   ├── trace_middleware.py     # Local JSONL trace recorder
│   ├── local_trace.py
│   ├── figure_service.py       # Thread-scoped figure storage
│   ├── web_search.py           # Tavily web search tools
│   ├── usage_cost.py            # Token usage + cost estimation
│   └── routines/
│       └── daily_briefing.py   # Personalized daily report generator
├── skills/                     # 24 skill directories (YAML frontmatter + MD)
├── strategies/                 # 11 strategy knowledge directories
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
2. Auto-discovered by `skill_manager.py`.

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

## Maintaining This File

`AGENTS.md` is the **single source of truth** — `.github/copilot-instructions.md` is a stub that links here. Only update this file.

| Change Type | What to Update |
|-------------|----------------|
| New tool / module / file | Add to Quick Reference table + Source Layout tree |
| New middleware | Update the numbered middleware stack in Architecture |
| New env variable | Add to Environment Variables table |
| New skill or strategy dir | Update counts ("24 skills", "11 strategies") |
| Dependency added/removed | Update tech stack in "What This Project Does" |
| Test convention changed | Update Key Patterns |
| New gotcha discovered | Append to Common Gotchas |

**Rule of thumb**: if a future agent would waste time searching for something you just changed, document it here.

Skip updates for: routine bug fixes, internal refactors with no API changes, or prompt wording tweaks.
