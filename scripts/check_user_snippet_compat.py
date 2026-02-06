from __future__ import annotations

import inspect

from agent.sandbox import _create_base_namespace


def safe_sig(obj, name: str) -> str:
    if not hasattr(obj, name):
        return "<missing>"
    try:
        return str(inspect.signature(getattr(obj, name)))
    except Exception as e:  # pragma: no cover
        return f"<signature_error: {type(e).__name__}: {e}>"


def main() -> None:
    ns = _create_base_namespace()
    store = ns["store"]
    inner = getattr(store, "_inner", store)

    print("store:", type(store))
    print("inner:", type(inner))

    for name in ["daily", "daily_adj", "daily_basic", "read", "trading_days"]:
        print(f"{name} sig:", safe_sig(inner, name))

    ts_code = "600519.SH"

    # Fast checks: avoid huge reads; still validate basic connectivity.
    print("\n-- quick call checks --")

    try:
        df = store.daily(ts_code)
        print("daily ok:", None if df is None else len(df), "cols:", None if df is None else list(df.columns)[:12])
    except Exception as e:
        print("daily error:", type(e).__name__, str(e))

    try:
        basic = store.daily_basic(ts_code)
        print(
            "daily_basic ok:",
            None if basic is None else len(basic),
            "cols:",
            None if basic is None else list(basic.columns)[:12],
        )
    except Exception as e:
        print("daily_basic error:", type(e).__name__, str(e))

    # Critical compatibility: does read() accept limit=?
    try:
        sample = store.read("index_basic", limit=3)
        print("read(index_basic, limit=3) ok:", None if sample is None else len(sample))
    except Exception as e:
        print("read(index_basic, limit=3) error:", type(e).__name__, str(e))

    try:
        days = store.trading_days("20240101", "20240110")
        print("trading_days ok:", None if days is None else len(days))
    except Exception as e:
        print("trading_days error:", type(e).__name__, str(e))


if __name__ == "__main__":
    main()
