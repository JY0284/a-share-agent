# A-Share Agent

A professional A-share (Chinese stock market) financial analysis agent powered by LangGraph and DeepSeek.

## Features

- **Stock Information**: Query stock basic info, company profiles, and name change history
- **Price Data**: Daily, weekly, and monthly OHLCV prices with adjustment options
- **Valuation Metrics**: PE, PB, market cap, turnover rates, and more
- **Trading Calendar**: Check trading days, previous/next trading dates
- **Bilingual Support**: Responds in Chinese or English based on user input

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Agent Chat UI  │────▶│  LangGraph Server │────▶│   StockStore    │
│   (Port 3000)   │     │    (Port 2024)    │     │   (Parquet +    │
│                 │◀────│                   │◀────│    DuckDB)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Setup

### 1. Install Dependencies (Backend)

```bash
cd a-share-agent
uv sync
source .venv/bin/activate
```

### 2. Configure Environment (Backend)

Create `a-share-agent/.env`:

```env
DEEPSEEK_API_KEY=your_api_key_here
STOCK_DATA_STORE_DIR=../stock_data/store
```

### 3. Start the Agent Backend (LangGraph server)

```bash
cd a-share-agent
# Recommended (works on Windows/macOS/Linux):
uv run langgraph dev --port 2024
```

This starts the LangGraph server at `http://127.0.0.1:2024`.

If the server prints the banner but the browser shows "not responding", try disabling hot reload:

```bash
uv run langgraph dev --port 2024 --no-reload
```

### 4. Start the Chat UI (agent-chat-ui frontend)

In a separate terminal:

```bash
# Optional: use your local proxy for cloning / npm installs
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

cd ..  # from a-share-agent/ to quant/
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui

# Create agent-chat-ui/.env.local with:
echo 'NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_ASSISTANT_ID=agent' > .env.local

npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

## Usage Examples

### Basic Queries

- "查询贵州茅台的基本信息" (Query Moutai's basic info)
- "600519最近10个交易日的价格" (Last 10 days prices for 600519)
- "比较平安银行和招商银行的PE" (Compare PE of Ping An Bank and CMB)

### Advanced Analysis

- "分析创业板最近一个月的涨幅前10的股票" (Analyze top 10 gainers in ChiNext)
- "300888的复权价格和成交量趋势" (Adjusted prices and volume trend for 300888)

## Development

### Test Locally

```bash
python main.py
```

### Project Structure

```
a-share-agent/
├── pyproject.toml      # Dependencies
├── langgraph.json      # LangGraph CLI config
├── .env                # Environment variables
├── main.py             # Test script
└── src/agent/
    ├── __init__.py
    ├── graph.py        # LangGraph agent definition
    ├── tools.py        # LangChain tool wrappers
    └── prompts.py      # System prompt
```

## Available Tools

| Tool | Description |
|------|-------------|
| `tool_resolve_symbol` | Convert symbol to ts_code |
| `tool_get_stock_basic` | Stock basic information |
| `tool_get_stock_company` | Company profile |
| `tool_get_universe` | Filter stock universe |
| `tool_get_trading_days` | Get trading days in range |
| `tool_is_trading_day` | Check if date is trading day |
| `tool_get_daily_prices` | Daily OHLCV prices |
| `tool_get_daily_adj_prices` | Adjusted daily prices |
| `tool_get_daily_basic` | Daily valuation metrics |
| `tool_get_weekly_prices` | Weekly OHLCV prices |
| `tool_get_monthly_prices` | Monthly OHLCV prices |
| `tool_get_stk_limit` | Limit-up/down prices |
| `tool_get_suspend_d` | Suspension events |
| `tool_get_new_share` | IPO information |
| `tool_get_namechange` | Name change history |

## License

MIT
