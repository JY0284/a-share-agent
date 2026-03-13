"""Backtest engine for common A-share strategies.

This module implements a deterministic backtest pipeline that the agent can invoke
directly via `tool_backtest_strategy`, **replacing hundreds of lines of LLM-generated
Python code** with a single tool call.

Supported strategies (single-asset signal-based):
  - dual_ma       : Dual moving-average crossover (SMA or EMA)
  - bollinger     : Bollinger-band mean-reversion
  - macd          : MACD histogram crossover
  - chandelier    : Chandelier (ATR trailing-stop) exit

Supported strategies (multi-asset):
  - momentum      : Momentum rotation (rank by N-day return, hold top-K)
  - buy_and_hold  : Simple buy-and-hold (benchmark comparison)

Pipeline for single-asset strategies:
  ① Load data   → store.daily_adj(ts_code, how="hfq")
  ② Signals     → strategy-specific entry/exit booleans
  ③ Position    → stateful loop, shift(1) for no-lookahead
  ④ Returns     → pos × ret_mkt − turnover × fee_rate
  ⑤ Metrics     → CAGR, Sharpe, Max-Drawdown, Win-Rate, Trade Count
  ⑥ (Optional)  → Equity curve + drawdown chart via matplotlib
"""

from __future__ import annotations

import io
import os
import base64
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stock_data.store import open_store

STORE_DIR = os.environ.get("STOCK_DATA_STORE_DIR", "../stock_data/store")

_store = None


def _get_store():
    global _store
    if _store is None:
        _store = open_store(STORE_DIR)
    return _store


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_prices(ts_code: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Load hfq-adjusted daily prices for a stock, or raw prices for an ETF."""
    store = _get_store()
    # Determine asset type by ts_code suffix / exchange convention
    code = ts_code.split(".")[0] if "." in ts_code else ts_code
    is_etf = code.startswith("5") and ts_code.endswith(".SH") or code.startswith("1") and ts_code.endswith(".SZ")

    if is_etf:
        df = store.etf_daily(ts_code, start_date=start_date, end_date=end_date)
    else:
        df = store.daily_adj(ts_code, how="hfq", start_date=start_date, end_date=end_date)

    if df is None or df.empty:
        raise ValueError(f"No data for {ts_code} in [{start_date}, {end_date}]")

    # Normalise date column
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").astype(int)

    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def _resolve_name(ts_code: str) -> str:
    """Try to get the stock/ETF name for chart labels."""
    try:
        store = _get_store()
        info = store.stock_basic(ts_code=ts_code)
        if info is not None and not info.empty:
            return str(info.iloc[0].get("name", ts_code))
    except Exception:
        pass
    return ts_code


# ---------------------------------------------------------------------------
# Position & metrics (shared across all signal-based strategies)
# ---------------------------------------------------------------------------

def _build_position(entry_sig: pd.Series, exit_sig: pd.Series) -> pd.Series:
    """Convert entry/exit boolean signals into a position series (0/1) with shift(1)."""
    pos_raw = []
    state = 0
    for ent, ext in zip(entry_sig.fillna(False), exit_sig.fillna(False)):
        if ent:
            state = 1
        elif ext:
            state = 0
        pos_raw.append(state)
    pos = pd.Series(pos_raw, index=entry_sig.index)
    return pos.shift(1).fillna(0).astype(float)


def _compute_returns(df: pd.DataFrame, pos: pd.Series, fee_rate: float) -> pd.DataFrame:
    """Compute market return, strategy return, equity curve, and drawdown."""
    df = df.copy()
    df["pos"] = pos.values
    df["ret_mkt"] = df["close"].pct_change().fillna(0.0)
    df["turnover"] = df["pos"].diff().abs().fillna(0.0)
    df["ret_strat"] = df["pos"] * df["ret_mkt"] - df["turnover"] * fee_rate
    df["equity"] = (1.0 + df["ret_strat"]).cumprod()
    df["equity_bh"] = (1.0 + df["ret_mkt"]).cumprod()  # buy-and-hold benchmark
    peak = df["equity"].cummax()
    df["drawdown"] = df["equity"] / peak - 1.0
    return df


@dataclass
class BacktestMetrics:
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    # buy-and-hold comparison
    bh_total_return: float
    bh_cagr: float
    bh_sharpe: float
    bh_max_drawdown: float

    def to_dict(self) -> dict:
        return {
            "total_return": round(self.total_return, 4),
            "cagr": round(self.cagr, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "trade_count": self.trade_count,
            "bh_total_return": round(self.bh_total_return, 4),
            "bh_cagr": round(self.bh_cagr, 4),
            "bh_sharpe": round(self.bh_sharpe, 4),
            "bh_max_drawdown": round(self.bh_max_drawdown, 4),
        }


def _calc_metrics(df: pd.DataFrame) -> BacktestMetrics:
    """Calculate performance metrics from a DataFrame with equity/equity_bh columns."""
    ann = 252.0
    days = max(len(df), 1)

    # Strategy metrics
    eq = df["equity"]
    ret = df["ret_strat"]
    total_ret = eq.iloc[-1] - 1.0
    cagr = eq.iloc[-1] ** (ann / days) - 1.0
    sharpe = (ret.mean() / (ret.std(ddof=0) + 1e-12)) * (ann ** 0.5)
    mdd = df["drawdown"].min()

    # Win rate & trade count
    pos_diff = df["pos"].diff().fillna(0)
    entries = (pos_diff > 0).sum()
    # For each round-trip, check if return during holding period > 0
    win_count = 0
    in_trade = False
    trade_ret = 0.0
    for _, row in df.iterrows():
        if row.get("pos", 0) > 0 and not in_trade:
            in_trade = True
            trade_ret = 0.0
        if in_trade:
            trade_ret += row.get("ret_strat", 0.0)
        if row.get("pos", 0) == 0 and in_trade:
            in_trade = False
            if trade_ret > 0:
                win_count += 1
    # Count last open trade
    if in_trade and trade_ret > 0:
        win_count += 1
    trade_count = int(entries)
    win_rate = win_count / max(trade_count, 1)

    # Buy-and-hold metrics
    eq_bh = df["equity_bh"]
    ret_bh = df["ret_mkt"]
    bh_total = eq_bh.iloc[-1] - 1.0
    bh_cagr = eq_bh.iloc[-1] ** (ann / days) - 1.0
    bh_sharpe = (ret_bh.mean() / (ret_bh.std(ddof=0) + 1e-12)) * (ann ** 0.5)
    bh_peak = eq_bh.cummax()
    bh_mdd = (eq_bh / bh_peak - 1.0).min()

    return BacktestMetrics(
        total_return=total_ret,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=mdd,
        win_rate=win_rate,
        trade_count=trade_count,
        bh_total_return=bh_total,
        bh_cagr=bh_cagr,
        bh_sharpe=bh_sharpe,
        bh_max_drawdown=bh_mdd,
    )


# ---------------------------------------------------------------------------
# Signal generators (one per strategy type)
# ---------------------------------------------------------------------------

def _signals_dual_ma(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    fast = int(params.get("fast", 5))
    slow = int(params.get("slow", 20))
    ma_type = str(params.get("ma_type", "sma")).lower()

    close = df["close"]
    if ma_type == "ema":
        ma_fast = close.ewm(span=fast, adjust=False).mean()
        ma_slow = close.ewm(span=slow, adjust=False).mean()
    else:
        ma_fast = close.rolling(fast).mean()
        ma_slow = close.rolling(slow).mean()

    entry = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    exit_ = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    return entry, exit_


def _signals_bollinger(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    period = int(params.get("period", 20))
    num_std = float(params.get("num_std", 2.0))

    close = df["close"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std

    # Mean-reversion: buy below lower, sell above upper
    entry = (close < lower) & (close.shift(1) >= lower.shift(1))
    exit_ = (close > upper) & (close.shift(1) <= upper.shift(1))
    return entry, exit_


def _signals_macd(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    signal_period = int(params.get("signal", 9))

    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    entry = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    exit_ = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    return entry, exit_


def _signals_chandelier(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    atr_period = int(params.get("atr_period", 22))
    mult = float(params.get("mult", 3.0))

    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    highest = high.rolling(atr_period).max()
    chandelier_stop = highest - mult * atr

    # Entry: close crosses above chandelier stop (momentum breakout)
    entry = (close > chandelier_stop) & (close.shift(1) <= chandelier_stop.shift(1))
    # Exit: close drops below chandelier stop
    exit_ = (close < chandelier_stop) & (close.shift(1) >= chandelier_stop.shift(1))
    return entry, exit_


def _signals_buy_and_hold(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    """Always in the market from day 1."""
    entry = pd.Series(False, index=df.index)
    exit_ = pd.Series(False, index=df.index)
    if len(df) > 0:
        entry.iloc[0] = True
    return entry, exit_


_SIGNAL_REGISTRY: dict[str, Any] = {
    "dual_ma": _signals_dual_ma,
    "bollinger": _signals_bollinger,
    "macd": _signals_macd,
    "chandelier": _signals_chandelier,
    "buy_and_hold": _signals_buy_and_hold,
}

# Default parameters for each strategy
_DEFAULT_PARAMS: dict[str, dict] = {
    "dual_ma": {"fast": 5, "slow": 20, "ma_type": "sma"},
    "bollinger": {"period": 20, "num_std": 2.0},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "chandelier": {"atr_period": 22, "mult": 3.0},
    "buy_and_hold": {},
    "momentum": {"n_days": 20, "top_k": 1, "rebal_freq": 20},
}


# ---------------------------------------------------------------------------
# Single-asset backtest
# ---------------------------------------------------------------------------

@dataclass
class SingleResult:
    ts_code: str
    name: str
    strategy: str
    params: dict
    metrics: BacktestMetrics
    equity_df: pd.DataFrame  # trade_date, equity, equity_bh, drawdown


def run_single_backtest(
    ts_code: str,
    strategy: str,
    params: dict | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fee_rate: float = 0.0003,
) -> SingleResult:
    """Run a single-asset strategy backtest."""
    if strategy not in _SIGNAL_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Available: {', '.join(_SIGNAL_REGISTRY.keys())}"
        )

    merged_params = dict(_DEFAULT_PARAMS.get(strategy, {}))
    if params:
        merged_params.update(params)

    df = _load_prices(ts_code, start_date, end_date)
    if len(df) < 30:
        raise ValueError(f"{ts_code}: insufficient data ({len(df)} rows, need ≥30)")

    entry_sig, exit_sig = _SIGNAL_REGISTRY[strategy](df, merged_params)
    pos = _build_position(entry_sig, exit_sig)
    df = _compute_returns(df, pos, fee_rate)
    metrics = _calc_metrics(df)
    name = _resolve_name(ts_code)

    eq_df = df[["trade_date", "equity", "equity_bh", "drawdown"]].copy()

    return SingleResult(
        ts_code=ts_code,
        name=name,
        strategy=strategy,
        params=merged_params,
        metrics=metrics,
        equity_df=eq_df,
    )


# ---------------------------------------------------------------------------
# Multi-asset momentum rotation
# ---------------------------------------------------------------------------

@dataclass
class MomentumResult:
    ts_codes: list[str]
    names: list[str]
    params: dict
    # Portfolio-level metrics
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    equal_weight_cagr: float
    equity_df: pd.DataFrame  # trade_date, equity, equity_ew (equal-weight benchmark)


def run_momentum_rotation(
    ts_codes: list[str],
    params: dict | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fee_rate: float = 0.001,
) -> MomentumResult:
    """Run momentum rotation across multiple assets."""
    defaults = dict(_DEFAULT_PARAMS["momentum"])
    if params:
        defaults.update(params)
    n_days = int(defaults.get("n_days", 20))
    top_k = int(defaults.get("top_k", 1))
    rebal_freq = int(defaults.get("rebal_freq", 20))

    # Load all assets
    frames: dict[str, pd.DataFrame] = {}
    names: list[str] = []
    for code in ts_codes:
        try:
            df = _load_prices(code, start_date, end_date)
            frames[code] = df.set_index("trade_date")[["close"]]
            names.append(_resolve_name(code))
        except Exception:
            names.append(code)
            continue

    if len(frames) < 2:
        raise ValueError(f"Need ≥2 assets with data, got {len(frames)}")

    # Build close price panel
    panel = pd.DataFrame({code: fr["close"] for code, fr in frames.items()})
    panel = panel.sort_index().dropna(how="all")
    panel = panel.ffill()

    # Daily returns
    rets = panel.pct_change().fillna(0.0)
    n_assets = len(panel.columns)

    # Momentum signal: rank by N-day return, pick top_k
    mom = panel.pct_change(n_days)

    # Build weight matrix (rebalance every rebal_freq days)
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    last_rebal = -rebal_freq  # force first rebalance

    for i, date in enumerate(panel.index):
        if i - last_rebal >= rebal_freq and i >= n_days:
            ranks = mom.iloc[i].rank(ascending=False)
            top = ranks[ranks <= top_k].index
            w = 1.0 / max(len(top), 1)
            weights.loc[date] = 0.0
            weights.loc[date, top] = w
            last_rebal = i
        elif i > 0:
            weights.iloc[i] = weights.iloc[i - 1]

    # Shift weights by 1 to avoid lookahead
    weights = weights.shift(1).fillna(0.0)

    # Portfolio returns with fee
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    port_ret = (weights * rets).sum(axis=1) - turnover * fee_rate
    equity = (1.0 + port_ret).cumprod()

    # Equal-weight benchmark
    ew_ret = rets.mean(axis=1)
    equity_ew = (1.0 + ew_ret).cumprod()

    # Metrics
    ann = 252.0
    days = max(len(equity), 1)
    total_ret = equity.iloc[-1] - 1.0
    cagr = equity.iloc[-1] ** (ann / days) - 1.0
    sharpe = (port_ret.mean() / (port_ret.std(ddof=0) + 1e-12)) * (ann ** 0.5)
    peak = equity.cummax()
    mdd = (equity / peak - 1.0).min()
    ew_cagr = equity_ew.iloc[-1] ** (ann / days) - 1.0

    eq_df = pd.DataFrame({
        "trade_date": equity.index,
        "equity": equity.values,
        "equity_ew": equity_ew.values,
    }).reset_index(drop=True)

    return MomentumResult(
        ts_codes=list(frames.keys()),
        names=names,
        params=defaults,
        total_return=round(total_ret, 4),
        cagr=round(cagr, 4),
        sharpe=round(sharpe, 4),
        max_drawdown=round(mdd, 4),
        equal_weight_cagr=round(ew_cagr, 4),
        equity_df=eq_df,
    )


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _generate_chart(
    results: list[SingleResult] | MomentumResult,
    strategy: str,
) -> list[dict[str, Any]]:
    """Generate equity curve + drawdown charts, return figure dicts (base64 png)."""
    try:
        import tempfile
        mpl_cfg = os.path.join(tempfile.gettempdir(), "a_share_agent_mplconfig")
        os.makedirs(mpl_cfg, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_cfg)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    figures: list[dict[str, Any]] = []

    if isinstance(results, MomentumResult):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})
        eq = results.equity_df
        ax1.plot(eq["trade_date"], eq["equity"], label="动量轮动策略", linewidth=1.5)
        ax1.plot(eq["trade_date"], eq["equity_ew"], label="等权基准", linewidth=1, alpha=0.7, ls="--")
        ax1.set_ylabel("净值")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        title = f"动量轮动回测 (top_k={results.params.get('top_k')}, n_days={results.params.get('n_days')})"
        ax1.set_title(title)

        # Drawdown
        peak = pd.Series(eq["equity"].values).cummax()
        dd = pd.Series(eq["equity"].values) / peak - 1.0
        ax2.fill_between(range(len(dd)), dd, 0, alpha=0.5, color="red")
        ax2.set_ylabel("回撤")
        ax2.set_xlabel("交易日")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close(fig)
        figures.append({"title": title, "image": img_b64, "format": "png"})
        return figures

    # Single-asset strategy results (possibly multiple stocks)
    if not results:
        return []

    # Chart 1: Equity curves overlay
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    for r in results:
        label = f"{r.name}({r.ts_code})"
        ax1.plot(r.equity_df["trade_date"], r.equity_df["equity"], label=label, linewidth=1.2)

    # Show buy-and-hold for the first stock as benchmark
    if len(results) == 1:
        r0 = results[0]
        ax1.plot(r0.equity_df["trade_date"], r0.equity_df["equity_bh"],
                 label="买入持有", linewidth=1, alpha=0.7, ls="--", color="gray")

    ax1.set_ylabel("净值")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Build strategy description for title
    r0 = results[0]
    param_str = ", ".join(f"{k}={v}" for k, v in r0.params.items())
    strategy_zh = {
        "dual_ma": "双均线", "bollinger": "布林带", "macd": "MACD",
        "chandelier": "吊灯止损", "buy_and_hold": "买入持有",
    }.get(strategy, strategy)
    title = f"{strategy_zh}策略回测 ({param_str})"
    if len(results) > 1:
        title += f" — {len(results)}只标的对比"
    ax1.set_title(title)

    # Drawdown for each stock
    for r in results:
        ax2.fill_between(
            r.equity_df["trade_date"], r.equity_df["drawdown"], 0,
            alpha=0.3, label=r.name,
        )
    ax2.set_ylabel("回撤")
    ax2.set_xlabel("交易日")
    ax2.grid(True, alpha=0.3)
    if len(results) > 1:
        ax2.legend(fontsize=7)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close(fig)
    figures.append({"title": title, "image": img_b64, "format": "png"})

    return figures


# ---------------------------------------------------------------------------
# Public API — called by tool_backtest_strategy
# ---------------------------------------------------------------------------

_STRATEGY_DESCRIPTIONS = {
    "dual_ma": "双均线交叉 (Dual Moving Average Crossover)",
    "bollinger": "布林带均值回归 (Bollinger Band Mean-Reversion)",
    "macd": "MACD交叉 (MACD Histogram Crossover)",
    "chandelier": "吊灯止损 (Chandelier ATR Exit)",
    "buy_and_hold": "买入持有 (Buy & Hold)",
    "momentum": "动量轮动 (Momentum Rotation — multi-asset)",
}


def list_strategies() -> dict:
    """Return available strategies with their default parameters."""
    return {
        name: {
            "description": _STRATEGY_DESCRIPTIONS.get(name, name),
            "default_params": _DEFAULT_PARAMS.get(name, {}),
        }
        for name in list(_SIGNAL_REGISTRY.keys()) + ["momentum"]
    }


def run_backtest(
    ts_codes: list[str],
    strategy: str,
    params: dict | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fee_rate: float | None = None,
    generate_chart: bool = True,
) -> dict[str, Any]:
    """Run a backtest and return structured results + optional charts.

    This is the main entry point called by tool_backtest_strategy.

    Args:
        ts_codes: One or more ts_codes (e.g. ["600519.SH"] or ["600519.SH", "000858.SZ"])
        strategy: Strategy name from list_strategies()
        params: Strategy parameters (merged with defaults)
        start_date: YYYYMMDD (optional)
        end_date: YYYYMMDD (optional)
        fee_rate: Fee per side. Default: 0.0003 (single-asset), 0.001 (momentum)
        generate_chart: Whether to generate equity curve charts

    Returns:
        {
          strategy: str,
          strategy_description: str,
          params: dict,
          ts_codes: [str, ...],
          results: [
            {ts_code, name, metrics: {cagr, sharpe, max_drawdown, ...}},
            ...
          ],
          comparison_table: str,  # pretty-printed markdown table
          figures: [{title, image, format}, ...] | None,
          errors: [str, ...],
        }
    """
    errors: list[str] = []
    figures: list[dict] = []

    if strategy == "momentum":
        default_fee = 0.001
        actual_fee = fee_rate if fee_rate is not None else default_fee
        try:
            mom_result = run_momentum_rotation(
                ts_codes, params=params,
                start_date=start_date, end_date=end_date,
                fee_rate=actual_fee,
            )
            if generate_chart:
                figures = _generate_chart(mom_result, strategy)

            return {
                "strategy": strategy,
                "strategy_description": _STRATEGY_DESCRIPTIONS.get(strategy, ""),
                "params": mom_result.params,
                "ts_codes": mom_result.ts_codes,
                "results": [{
                    "type": "momentum_portfolio",
                    "ts_codes": mom_result.ts_codes,
                    "names": mom_result.names,
                    "metrics": {
                        "total_return": mom_result.total_return,
                        "cagr": mom_result.cagr,
                        "sharpe": mom_result.sharpe,
                        "max_drawdown": mom_result.max_drawdown,
                        "equal_weight_cagr": mom_result.equal_weight_cagr,
                    },
                }],
                "comparison_table": (
                    f"| 指标 | 动量轮动 | 等权基准 |\n"
                    f"|------|---------|----------|\n"
                    f"| 年化收益率 | {mom_result.cagr:.2%} | {mom_result.equal_weight_cagr:.2%} |\n"
                    f"| 夏普比率 | {mom_result.sharpe:.2f} | — |\n"
                    f"| 最大回撤 | {mom_result.max_drawdown:.2%} | — |\n"
                    f"| 总收益 | {mom_result.total_return:.2%} | — |"
                ),
                "figures": figures or None,
                "errors": errors,
            }
        except Exception as e:
            return {
                "strategy": strategy,
                "strategy_description": _STRATEGY_DESCRIPTIONS.get(strategy, ""),
                "params": params or _DEFAULT_PARAMS.get(strategy, {}),
                "ts_codes": ts_codes,
                "results": [],
                "comparison_table": "",
                "figures": None,
                "errors": [f"Momentum backtest failed: {e}"],
            }

    # Single-asset strategies: run for each ts_code
    default_fee = 0.0003
    actual_fee = fee_rate if fee_rate is not None else default_fee

    single_results: list[SingleResult] = []
    result_dicts: list[dict] = []

    for code in ts_codes:
        try:
            sr = run_single_backtest(
                code, strategy, params=params,
                start_date=start_date, end_date=end_date,
                fee_rate=actual_fee,
            )
            single_results.append(sr)
            result_dicts.append({
                "ts_code": sr.ts_code,
                "name": sr.name,
                "metrics": sr.metrics.to_dict(),
            })
        except Exception as e:
            errors.append(f"{code}: {e}")

    if not single_results:
        return {
            "strategy": strategy,
            "strategy_description": _STRATEGY_DESCRIPTIONS.get(strategy, ""),
            "params": params or _DEFAULT_PARAMS.get(strategy, {}),
            "ts_codes": ts_codes,
            "results": [],
            "comparison_table": "",
            "figures": None,
            "errors": errors or ["All backtests failed"],
        }

    # Generate charts
    if generate_chart:
        figures = _generate_chart(single_results, strategy)

    # Build comparison table
    header = "| 标的 | 年化收益(CAGR) | 夏普比率 | 最大回撤 | 胜率 | 交易次数 | 买入持有CAGR |"
    sep = "|------|---------------|---------|---------|------|---------|------------|"
    rows = []
    for rd in result_dicts:
        m = rd["metrics"]
        rows.append(
            f"| {rd['name']}({rd['ts_code']}) "
            f"| {m['cagr']:.2%} "
            f"| {m['sharpe']:.2f} "
            f"| {m['max_drawdown']:.2%} "
            f"| {m['win_rate']:.1%} "
            f"| {m['trade_count']} "
            f"| {m['bh_cagr']:.2%} |"
        )
    table = "\n".join([header, sep] + rows)

    merged_params = single_results[0].params if single_results else (params or {})

    return {
        "strategy": strategy,
        "strategy_description": _STRATEGY_DESCRIPTIONS.get(strategy, ""),
        "params": merged_params,
        "ts_codes": [sr.ts_code for sr in single_results],
        "results": result_dicts,
        "comparison_table": table,
        "figures": figures or None,
        "errors": errors,
    }
