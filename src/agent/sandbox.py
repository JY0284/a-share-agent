"""Python execution sandbox for the A-Share agent.

Provides a safe(r) execution environment with:
- Pre-loaded data science libraries (pandas, numpy)
- Access to StockStore for loading A-share data
- Timeout and output capture
- Session-scoped namespace so variables persist across tool_execute_python calls in the same agent thread
- Automatic capture of matplotlib figures as base64-encoded images
- Optional seaborn (`sns`) support for higher-level statistical plotting
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import traceback
from contextlib import redirect_stdout, redirect_stderr
from contextvars import ContextVar
from typing import Any

# Pre-import common libraries for the sandbox
import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
from arch import arch_model

from stock_data.store import open_store

# Store directory from environment
STORE_DIR = os.environ.get("STOCK_DATA_STORE_DIR", "../stock_data/store")

# Singleton store instance
_store = None

# Session id for this request (set by middleware from thread_id / trace id)
_python_session_id_var: ContextVar[str | None] = ContextVar("python_session_id", default=None)

# Thread id for figure storage (the actual LangGraph thread_id, not trace_id)
_thread_id_var: ContextVar[str | None] = ContextVar("thread_id", default=None)

# One namespace per session: session_id -> dict (same as exec() globals)
_session_namespaces: dict[str, dict[str, Any]] = {}


def set_python_session_id(session_id: str) -> None:
    """Set the current Python execution session id (called by middleware from thread_id)."""
    _python_session_id_var.set(session_id)


def set_thread_id(thread_id: str) -> None:
    """Set the actual LangGraph thread_id for figure storage."""
    _thread_id_var.set(thread_id)


def get_thread_id() -> str | None:
    """Return the actual thread_id for figure storage, or None if not set."""
    return _thread_id_var.get()


def get_python_session_id() -> str | None:
    """Return the current session id, or None if not set."""
    return _python_session_id_var.get()


def clear_python_session(session_id: str | None = None) -> None:
    """Clear the namespace for a session so the next run starts fresh.

    Args:
        session_id: If given, clear that session only. If None, clear the current session.
    """
    global _session_namespaces
    if session_id is None:
        session_id = get_python_session_id()
    if session_id and session_id in _session_namespaces:
        del _session_namespaces[session_id]


def get_store():
    """Get the shared StockStore instance."""
    global _store
    if _store is None:
        _store = open_store(STORE_DIR)
    return _store


_DATE_COLS = ("trade_date", "end_date", "ann_date", "f_ann_date")


def _coerce_yyyymmdd_intlike(s: pd.Series) -> pd.Series:
    """Coerce YYYYMMDD-like columns to int-like values when safe.

    Many datasets store dates as strings/Arrow strings; code often compares them
    with ints (e.g. df[df['trade_date'] >= 20240101]), which raises dtype errors.
    """
    if s is None:
        return s

    if pd.api.types.is_integer_dtype(s.dtype):
        return s

    # Convert to string, remove '-', then numeric
    try:
        ss = s.astype(str).str.replace("-", "", regex=False)
        out = pd.to_numeric(ss, errors="ignore")
        return out
    except Exception:
        return s


def _coerce_df_date_cols(df: Any) -> Any:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    for c in _DATE_COLS:
        if c in df.columns:
            df[c] = _coerce_yyyymmdd_intlike(df[c])
    return df


class _StoreProxy:
    """Proxy StockStore to normalize common gotchas for analysis code."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs):
            out = attr(*args, **kwargs)
            return _coerce_df_date_cols(out)

        return _wrapped


def _enhance_error_message(err: str) -> str:
    """Append targeted hints for common failures seen in traces."""
    if not err:
        return err
    hints: list[str] = []

    # Tool misuse inside Python
    if "from tool_use" in err or "tool_get_" in err and "ModuleNotFoundError" in err:
        hints.append("不要在 Python 代码里 import/调用工具；请先在对话里调用 tool_* 获取数据，再用 Python 做计算。")

    # Store API misuse: offset/limit
    if "unexpected keyword argument 'offset'" in err or "unexpected keyword argument \"offset\"" in err:
        hints.append("store.read() 不支持 offset 分页；请改用 tool_get_universe(offset, limit) 或一次性读取后用 pandas 切片。")
    if "unexpected keyword argument 'limit'" in err or "unexpected keyword argument \"limit\"" in err:
        hints.append("很多 store.* 方法不支持 limit；请先加载 DataFrame，再用 .head(n)/.tail(n) 截取。")

    # DuckDB/Parquet binder error: missing ts_code
    if "Referenced column \"ts_code\" not found" in err or "Referenced column 'ts_code' not found" in err:
        hints.append("不要用 store.read(..., where={'ts_code': ...}) 过滤该表；优先用专用方法如 store.income/ store.fina_indicator(ts_code, ...)。")

    # Common dtype mismatch on trade_date
    if "Invalid comparison between dtype=str and int" in err or "not supported between instances of 'str' and 'int'" in err:
        hints.append("日期列请统一类型：推荐把 trade_date/end_date 转成 YYYYMMDD 的 int（例如 df['trade_date']=df['trade_date'].astype(str).str.replace('-', '').astype(int)）。")

    # Float vs str comparison (also a dtype mismatch)
    if "not supported between instances of 'float' and 'str'" in err:
        hints.append("数值与字符串比较错误：请确保比较的两边类型一致（都是数值或都是字符串）。")

    # ETF adj_factor misuse
    if "etf_daily" in err and ("adj" in err.lower() or "复权" in err):
        hints.append("ETF 没有复权因子 (adj_factor)；直接用 store.etf_daily(ts_code) 获取价格即可（ETF 不拆股，无需复权）。")
    if "'StockStore' object has no attribute 'etf_adj'" in err or "'StockStore' object has no attribute 'fund_daily_adj'" in err:
        hints.append("store 没有 etf_adj/fund_daily_adj 方法；ETF 不需要复权，直接用 store.etf_daily(ts_code) 获取原始价格。")
    if "'StockStore' object has no attribute 'etf_daily'" in err:
        hints.append("store.etf_daily(ts_code) 是正确的 ETF 日线方法；请检查 ts_code 格式（如 510300.SH）。")

    # IndexError on empty DataFrame
    if "IndexError" in err and "out-of-bounds" in err:
        hints.append("IndexError 通常表示 DataFrame 为空或筛选后无数据；请先检查 df.empty 或 len(df)，再使用 iloc 访问。")

    # Optional deps
    if "No module named 'matplotlib'" in err:
        hints.append("环境缺少 matplotlib；如不画图请删掉 import/绘图代码，或安装依赖后再运行。")
    if "No module named 'scipy'" in err:
        hints.append("环境缺少 scipy；如不需要统计检验请移除 scipy 依赖，或安装 scipy 后再运行。")
    if "No module named 'statsmodels'" in err:
        hints.append("环境缺少 statsmodels；已预装 sm (statsmodels.api)，直接使用 sm.OLS / sm.add_constant 等。")
    if "No module named 'arch'" in err:
        hints.append("环境缺少 arch；已预装 arch_model，直接使用 arch_model(returns, vol='Garch', p=1, q=1) 即可。")

    # KeyError for derived columns
    if "KeyError:" in err and ("ma" in err or "significant" in err or "公告" in err):
        hints.append("KeyError 通常表示列没生成/数据为空；先检查 df.empty / len(df) 是否足够，再创建列后再引用。")

    # NameError
    if "NameError:" in err:
        hints.append("NameError 表示变量未定义：确保变量在所有分支都被赋值，或把依赖变量的代码放到变量定义之后。")

    if not hints:
        return err

    return err + "\n\nHints:\n- " + "\n- ".join(hints)


def _create_base_namespace() -> dict[str, Any]:
    """Build the initial execution namespace (pd, np, store, plt, sm, arch_model if available)."""
    namespace: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "pandas": pd,
        "numpy": np,
        "scipy": scipy,
        "sm": sm,
        "statsmodels": sm,
        "arch_model": arch_model,
        "store": _StoreProxy(get_store()),
        "STORE_DIR": STORE_DIR,
        "__builtins__": __builtins__,
    }
    try:
        mpl_cfg = os.path.join(tempfile.gettempdir(), "a_share_agent_mplconfig")
        os.makedirs(mpl_cfg, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_cfg)
        cache_home = os.path.join(tempfile.gettempdir(), "a_share_agent_cache")
        os.makedirs(cache_home, exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", cache_home)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        namespace["plt"] = plt
        namespace["matplotlib"] = matplotlib

        # Seaborn is optional, but when present expose it as `sns`.
        # Import it only after matplotlib backend is configured.
        try:
            import seaborn as sns

            # Make plots look reasonable by default; users can override.
            sns.set_theme()
            namespace["sns"] = sns
            namespace["seaborn"] = sns
        except ImportError:
            pass
    except ImportError:
        pass
    return namespace


def _capture_matplotlib_figures(namespace: dict[str, Any], thread_id: str | None = None) -> list[dict[str, Any]]:
    """Capture all open matplotlib figures, save to disk, and return metadata.
    
    Figures are saved to the filesystem via the figures service for persistent storage.
    The returned metadata includes URLs for frontend display.
    
    Args:
        namespace: Execution namespace containing plt module
        thread_id: Thread ID for organizing figures (uses current session if not provided)
    
    Returns:
        List of dicts with keys:
        - "id": unique figure ID (e.g., "fig_abc12345")
        - "url": API URL to fetch the image (e.g., "/api/figures/{thread_id}/{fig_id}")
        - "title": figure title or default "Figure N"
        - "format": "png"
        - "reference": formatted reference for agent responses (e.g., "[[fig:fig_abc12345|Title]]")
    """
    figures: list[dict[str, Any]] = []
    
    plt = namespace.get("plt")
    if plt is None:
        return figures
    
    # Import figure storage service
    from agent.figures import save_figure, get_figure_url, format_figure_reference, get_thread_id as get_fig_thread_id
    
    # Determine thread ID - prefer the actual thread_id from set_thread_id() over the passed parameter
    # get_thread_id() returns the actual LangGraph thread_id set by middleware
    # The thread_id parameter may be the session_id/trace_id which is different
    sandbox_thread_id = get_thread_id()
    fig_thread_id = get_fig_thread_id()
    session_id = get_python_session_id()
    print(f"[DEBUG] _capture_matplotlib_figures: thread_id={thread_id}, sandbox_thread_id={sandbox_thread_id}, fig_thread_id={fig_thread_id}, session_id={session_id}")
    # Priority: actual thread_id from middleware > passed thread_id > fallbacks
    tid = sandbox_thread_id or thread_id or fig_thread_id or session_id or "default"
    print(f"[DEBUG] Using tid={tid} for figure storage")
    
    try:
        # Get all open figure numbers
        fig_nums = plt.get_fignums()
        for i, num in enumerate(fig_nums):
            fig = plt.figure(num)
            
            # Get title from figure or axes
            title = fig.get_suptitle() or ""
            if not title and fig.axes:
                title = fig.axes[0].get_title() or ""
            if not title:
                title = f"Figure {i + 1}"
            
            # Save figure to bytes buffer as PNG
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
            buf.seek(0)
            
            # Encode as base64 for storage
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            
            # Save to filesystem and get metadata
            fig_meta = save_figure(
                image_base64=img_base64,
                title=title,
                format="png",
                thread_id=tid,
            )
            
            # Build figure info with URL
            fig_id = fig_meta["id"]
            figures.append({
                "id": fig_id,
                "url": get_figure_url(tid, fig_id),
                "title": title,
                "format": "png",
                "reference": format_figure_reference(fig_id, title),
                # Keep base64 for backward compatibility (artifact display)
                "image": img_base64,
            })
        
        # Close all figures to free memory
        plt.close("all")
        
    except Exception:
        # If anything fails, just return empty list
        pass
    
    return figures


def execute_python(code: str, session_id: str | None = None, timeout_seconds: int = 60) -> dict[str, Any]:
    """Execute Python code in a sandboxed environment.
    
    Uses a session-scoped namespace so variables persist across calls in the same
    agent thread (session_id is set by middleware from thread_id).
    
    The execution environment includes:
    - `pd`: pandas
    - `np`: numpy
    - `scipy`: scipy (for stats)
    - `sm`: statsmodels.api (OLS, add_constant, time series, etc.)
    - `arch_model`: arch.arch_model (GARCH volatility models)
    - `store`: StockStore instance for data access
    - `plt`: matplotlib.pyplot
    - `sns`: seaborn
    
    Args:
        code: Python code to execute
        session_id: Agent thread/session id. If None, uses get_python_session_id() or "default".
        timeout_seconds: Max execution time (not strictly enforced in this simple impl)
    
    Returns:
        {
            "success": bool,
            "output": str,       # stdout/print output
            "error": str | None, # error message if failed
            "result": Any,       # last expression value (if any)
        }
    """
    if session_id is None:
        session_id = get_python_session_id() or "default"

    if session_id not in _session_namespaces:
        _session_namespaces[session_id] = _create_base_namespace()
    namespace = _session_namespaces[session_id]

    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    result = None
    error = None
    success = False
    figures: list[dict[str, Any]] = []  # List of figure metadata with URLs
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the code
            # Use exec for statements, but try to capture last expression
            exec(compile(code, "<agent_code>", "exec"), namespace)
            
            # Check if there's a 'result' variable set by the code
            if "result" in namespace and namespace["result"] is not None:
                result = namespace["result"]
            # Or check for '_' (last expression in interactive mode)
            elif "_" in namespace:
                result = namespace["_"]
        
        # Capture any matplotlib figures that were created
        figures = _capture_matplotlib_figures(namespace, session_id)
        
        success = True
        
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        error = _enhance_error_message(error)
        success = False
    
    # Get captured output
    output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    if stderr_output:
        output += f"\n[stderr]\n{stderr_output}"
    
    # Format result for JSON serialization
    result_str = None
    if result is not None:
        if isinstance(result, pd.DataFrame):
            if len(result) > 50:
                result_str = f"DataFrame ({len(result)} rows x {len(result.columns)} cols):\n{result.head(30).to_string()}\n... ({len(result) - 30} more rows)"
            else:
                result_str = f"DataFrame ({len(result)} rows x {len(result.columns)} cols):\n{result.to_string()}"
        elif isinstance(result, pd.Series):
            if len(result) > 50:
                result_str = f"Series ({len(result)} items):\n{result.head(30).to_string()}\n... ({len(result) - 30} more items)"
            else:
                result_str = f"Series ({len(result)} items):\n{result.to_string()}"
        elif isinstance(result, (dict, list)):
            import json
            try:
                result_str = json.dumps(result, indent=2, ensure_ascii=False, default=str)
            except:
                result_str = str(result)
        else:
            result_str = str(result)
    
    return {
        "success": success,
        "output": output.strip() if output else None,
        "error": error,
        "result": result_str,
        "figures": figures if figures else None,  # List of figure metadata with URLs and references
    }
