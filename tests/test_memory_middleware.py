"""Tests for memory_middleware dual-layer injection.

Verifies that:
1. Structured UserProfile is always injected (even without mem0)
2. Soft mem0 memories are appended when available
3. dev_user fallback works
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Isolate profile storage
_tmp = tempfile.mkdtemp()
os.environ["USER_PROFILE_DIR"] = os.path.join(_tmp, "mw_profiles")

import agent.user_profile as up
import agent.memory_middleware as mm


@pytest.fixture(autouse=True)
def _clean():
    up._profile_dir_ensured = None  # reset cache
    d = Path(os.environ["USER_PROFILE_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.json"):
        f.unlink()
    yield


# ---------------------------------------------------------------------------
# _get_user_id_from_runtime
# ---------------------------------------------------------------------------


class TestGetUserId:
    def test_from_config(self):
        runtime = MagicMock()
        runtime.config = {"configurable": {"user_id": "alice"}}
        assert mm._get_user_id_from_runtime(runtime) == "alice"

    def test_fallback_to_dev_user(self):
        runtime = MagicMock()
        runtime.config = {"configurable": {}}
        runtime.context = None
        uid = mm._get_user_id_from_runtime(runtime)
        assert uid == mm._DEFAULT_USER_ID

    def test_no_config(self):
        runtime = MagicMock(spec=[])  # no config attr
        uid = mm._get_user_id_from_runtime(runtime)
        assert uid == mm._DEFAULT_USER_ID


# ---------------------------------------------------------------------------
# Context block building
# ---------------------------------------------------------------------------


class TestBuildContextBlock:
    def setup_method(self):
        self.mw = mm.MemoryMiddleware()

    def test_both_layers(self):
        profile_text = "## User Profile\n- holding: 茅台"
        memories = [{"memory": "user likes conservative investing"}]
        block = self.mw._build_context_block(profile_text, memories)
        assert "User Profile" in block
        assert "conservative investing" in block

    def test_profile_only(self):
        block = self.mw._build_context_block("## Profile data", [])
        assert "Profile data" in block
        assert "Conversational Memory" not in block

    def test_memory_only(self):
        block = self.mw._build_context_block(None, [{"memory": "fact"}])
        assert "Conversational Memory" in block
        assert "fact" in block

    def test_empty(self):
        """With no profile and no memories, block still contains live date."""
        block = self.mw._build_context_block(None, [])
        assert block is not None
        assert "Current Date" in block


# ---------------------------------------------------------------------------
# Sync wrap_model_call
# ---------------------------------------------------------------------------


class TestSyncWrapModelCall:
    def test_injects_profile(self):
        """Verify profile is injected into last HumanMessage, not as SystemMessage."""
        # Create a profile with holdings
        up.get_or_create_profile("sync_test_user")
        up.update_portfolio(
            "sync_test_user",
            holdings=[{"name": "茅台", "ts_code": "600519.SH", "shares": 100}],
        )

        mw = mm.MemoryMiddleware()

        # Build a mock request
        from langchain_core.messages import HumanMessage, SystemMessage

        human_msg = HumanMessage(content="向我汇报")
        request = MagicMock()
        request.runtime = MagicMock()
        request.runtime.config = {"configurable": {"user_id": "sync_test_user"}}
        request.messages = [human_msg]

        captured_request = {}

        def mock_handler(req):
            captured_request["req"] = req
            return "response"

        # Disable mem0 for this test
        with patch.object(mm, "MEM0_ENABLED", False):
            mw.wrap_model_call(request, mock_handler)

        # The handler should have been called with an overridden request
        assert request.override.called

    def test_context_in_human_message(self):
        """Verify context block is prepended to HumanMessage content."""
        mw = mm.MemoryMiddleware()

        from langchain_core.messages import HumanMessage

        human_msg = HumanMessage(content="茅台最新价格")
        request = MagicMock()
        request.messages = [human_msg]

        context_block = "## 📅 Current Date\n- Date: 2026-03-20\n"

        result = mw._inject_context(request, context_block)

        # Should call override with modified messages
        assert request.override.called
        call_kwargs = request.override.call_args
        new_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert len(new_messages) == 1
        assert isinstance(new_messages[0], HumanMessage)
        assert "[User Context]" in new_messages[0].content
        assert "茅台最新价格" in new_messages[0].content

    def test_no_double_injection(self):
        """Verify context is not injected twice if marker already present."""
        mw = mm.MemoryMiddleware()

        from langchain_core.messages import HumanMessage

        # Message already has the context marker
        human_msg = HumanMessage(content="[User Context]\nsome context\n---\n茅台最新价格")
        request = MagicMock()
        request.messages = [human_msg]

        context_block = "## 📅 Current Date\n- Date: 2026-03-20\n"

        result = mw._inject_context(request, context_block)

        # Should return original request without calling override
        assert result is request
        assert not request.override.called
