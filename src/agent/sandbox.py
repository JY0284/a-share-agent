"""Python execution sandbox for the A-Share agent.

Provides a safe(r) execution environment with:
- Pre-loaded data science libraries (pandas, numpy)
- Access to StockStore for loading A-share data
- Timeout and output capture
"""

from __future__ import annotations

import io
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
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


def get_store():
    """Get the shared StockStore instance."""
    global _store
    if _store is None:
        _store = open_store(STORE_DIR)
    return _store


def execute_python(code: str, timeout_seconds: int = 60) -> dict[str, Any]:
    """Execute Python code in a sandboxed environment.
    
    The execution environment includes:
    - `pd`: pandas
    - `np`: numpy
    - `store`: StockStore instance for data access
    - `plt`: matplotlib.pyplot (if available)
    
    Args:
        code: Python code to execute
        timeout_seconds: Max execution time (not strictly enforced in this simple impl)
    
    Returns:
        {
            "success": bool,
            "output": str,       # stdout/print output
            "error": str | None, # error message if failed
            "result": Any,       # last expression value (if any)
        }
    """
    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Build execution namespace with pre-loaded tools
    namespace = {
        # Data science
        "pd": pd,
        "np": np,
        "pandas": pd,
        "numpy": np,
        "scipy": scipy,
        # Stock data
        "store": get_store(),
        "STORE_DIR": STORE_DIR,
        # Utilities
        "__builtins__": __builtins__,
    }
    
    # Try to add matplotlib if available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        namespace["plt"] = plt
        namespace["matplotlib"] = matplotlib
    except ImportError:
        pass
    
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
