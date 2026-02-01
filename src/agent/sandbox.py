"""Python execution sandbox for the A-Share agent.

Provides a safe(r) execution environment with:
- Pre-loaded data science libraries (pandas, numpy)
- Access to StockStore for loading A-share data
- Timeout and output capture
- Session-scoped namespace so variables persist across tool_execute_python calls in the same agent thread
"""

from __future__ import annotations

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

from stock_data.store import open_store

# Store directory from environment
STORE_DIR = os.environ.get("STOCK_DATA_STORE_DIR", "../stock_data/store")

# Singleton store instance
_store = None

# Session id for this request (set by middleware from thread_id / trace id)
_python_session_id_var: ContextVar[str | None] = ContextVar("python_session_id", default=None)

# One namespace per session: session_id -> dict (same as exec() globals)
_session_namespaces: dict[str, dict[str, Any]] = {}


def set_python_session_id(session_id: str) -> None:
    """Set the current Python execution session id (called by middleware from thread_id)."""
    _python_session_id_var.set(session_id)


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


def _create_base_namespace() -> dict[str, Any]:
    """Build the initial execution namespace (pd, np, store, plt if available)."""
    namespace: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "pandas": pd,
        "numpy": np,
        "scipy": scipy,
        "store": get_store(),
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
    except ImportError:
        pass
    return namespace


def execute_python(code: str, session_id: str | None = None, timeout_seconds: int = 60) -> dict[str, Any]:
    """Execute Python code in a sandboxed environment.
    
    Uses a session-scoped namespace so variables persist across calls in the same
    agent thread (session_id is set by middleware from thread_id).
    
    The execution environment includes:
    - `pd`: pandas
    - `np`: numpy
    - `store`: StockStore instance for data access
    - `plt`: matplotlib.pyplot (if available)
    
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
        
        success = True
        
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
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
    }
