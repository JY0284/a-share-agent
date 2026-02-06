"""Tests that simulate tool calls during an agent conversation.

These tests verify that the Python execution sandbox keeps runtime state
(session-scoped namespace) across multiple tool_execute_python calls in the
same thread, and that tool_clear_python_session resets state as expected.
"""

import pytest

from agent.sandbox import (
    clear_python_session,
    execute_python,
    get_python_session_id,
    set_python_session_id,
)
from agent.tools import tool_clear_python_session, tool_execute_python


# ---- Simulate conversation via execute_python + session_id ----


def test_session_persists_variables_across_calls():
    """Simulate: agent runs two Python steps in one thread; second step sees first step's variables."""
    session_id = "thread-conv-1"
    set_python_session_id(session_id)

    # First "turn": agent loads data into df (we use a simple DataFrame, no store)
    r1 = execute_python(
        "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}); result = len(df)",
        session_id=session_id,
    )
    assert r1["success"] is True
    assert r1["error"] is None
    assert "3" in (r1["result"] or "")

    # Second "turn": agent reuses df and computes something else
    r2 = execute_python(
        "result = df['a'].sum()",
        session_id=session_id,
    )
    assert r2["success"] is True
    assert r2["error"] is None
    assert "6" in (r2["result"] or "")

    clear_python_session(session_id)


def test_different_sessions_have_isolated_namespaces():
    """Simulate: two conversations (threads) run; each has its own namespace."""
    set_python_session_id("thread-A")
    execute_python("x = 1", session_id="thread-A")
    set_python_session_id("thread-B")
    execute_python("x = 2", session_id="thread-B")

    # Thread A should still see x == 1
    rA = execute_python("result = x", session_id="thread-A")
    assert rA["success"] is True
    assert "1" in (rA["result"] or "")

    # Thread B should see x == 2
    rB = execute_python("result = x", session_id="thread-B")
    assert rB["success"] is True
    assert "2" in (rB["result"] or "")

    clear_python_session("thread-A")
    clear_python_session("thread-B")


def test_clear_python_session_resets_namespace():
    """Simulate: agent clears session; next Python run in that thread has fresh namespace."""
    session_id = "thread-clear-test"
    set_python_session_id(session_id)
    execute_python("y = 10", session_id=session_id)

    clear_python_session(session_id)

    # Next run should not see y
    r = execute_python("result = y", session_id=session_id)
    assert r["success"] is False
    assert "NameError" in (r["error"] or "")
    assert r["result"] is None

    # New variable in same session works
    r2 = execute_python("y = 5; result = y", session_id=session_id)
    assert r2["success"] is True
    assert "5" in (r2["result"] or "")

    clear_python_session(session_id)


def test_execute_python_uses_context_session_id_when_not_passed():
    """When session_id is not passed, execute_python uses get_python_session_id() or 'default'."""
    set_python_session_id("ctx-thread")
    # Do not pass session_id; should use context
    r1 = execute_python("z = 100")
    assert r1["success"] is True
    r2 = execute_python("result = z")
    assert r2["success"] is True
    assert "100" in (r2["result"] or "")
    clear_python_session("ctx-thread")


# ---- Simulate conversation via tools (as the agent would call them) ----


def test_tool_execute_python_persists_state_via_context():
    """Simulate: middleware set session id; agent calls tool_execute_python twice; state persists."""
    set_python_session_id("conv-tool-1")

    out1 = tool_execute_python.invoke({"code": "data = [1, 2, 3]; result = sum(data)"})
    assert out1.get("success") is True
    assert "6" in (out1.get("result") or "")

    out2 = tool_execute_python.invoke({"code": "result = data + [4]"})
    assert out2.get("success") is True
    assert "4" in (out2.get("result") or "")

    clear_python_session("conv-tool-1")


def test_tool_clear_python_session_resets_state_then_tool_sees_fresh_namespace():
    """Simulate: agent runs Python, then calls tool_clear_python_session; next Python run is fresh."""
    set_python_session_id("conv-clear-tool")

    tool_execute_python.invoke({"code": "w = 99"})
    out_clear = tool_clear_python_session.invoke({})
    assert out_clear.get("cleared") is True

    # Next Python run should not see w
    out_fail = tool_execute_python.invoke({"code": "result = w"})
    assert out_fail.get("success") is False
    assert "NameError" in (out_fail.get("error") or "")

    clear_python_session("conv-clear-tool")


def test_three_turn_conversation_simulation():
    """Simulate a short conversation: load -> compute -> summarize across three tool calls."""
    set_python_session_id("conv-three-turn")

    # Turn 1: "load" data
    t1 = tool_execute_python.invoke({
        "code": "prices = pd.Series([100, 102, 101, 105]); result = 'loaded'",
    })
    assert t1.get("success") is True

    # Turn 2: compute returns
    t2 = tool_execute_python.invoke({
        "code": "returns = prices.pct_change().dropna(); result = returns.tolist()",
    })
    assert t2.get("success") is True

    # Turn 3: use both
    t3 = tool_execute_python.invoke({
        "code": "result = {'len_prices': len(prices), 'len_returns': len(returns)}",
    })
    assert t3.get("success") is True
    assert "len_prices" in (t3.get("result") or "")
    assert "len_returns" in (t3.get("result") or "")

    clear_python_session("conv-three-turn")


def test_tool_returns_skills_used():
    """tool_execute_python echoes skills_used in the response."""
    set_python_session_id("conv-skills")
    out = tool_execute_python.invoke({
        "code": "result = 1",
        "skills_used": ["backtest_ma_crossover", "rolling_indicators"],
    })
    assert out.get("success") is True
    assert out.get("skills_used") == ["backtest_ma_crossover", "rolling_indicators"]
    clear_python_session("conv-skills")


def test_execute_python_error_hints_offset() -> None:
    out = execute_python("raise TypeError(\"StockStore.read() got an unexpected keyword argument 'offset'\")")
    assert out["success"] is False
    assert "Hints:" in (out["error"] or "")
    assert "offset" in (out["error"] or "")


def test_execute_python_error_hints_missing_ts_code() -> None:
    out = execute_python("raise Exception(\"BinderException: Referenced column \\\"ts_code\\\" not found\")")
    assert out["success"] is False
    assert "Hints:" in (out["error"] or "")
    assert "ts_code" in (out["error"] or "")
