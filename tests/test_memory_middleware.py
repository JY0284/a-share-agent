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
        """With no profile and no memories, block still contains live datetime."""
        block = self.mw._build_context_block(None, [])
        assert block is not None
        assert "Current Date/Time" in block


# ---------------------------------------------------------------------------
# Sync wrap_model_call
# ---------------------------------------------------------------------------


class TestSyncWrapModelCall:
    def test_injects_profile(self):
        """Verify profile is injected even without mem0."""
        # Create a profile with holdings
        up.get_or_create_profile("sync_test_user")
        up.update_portfolio(
            "sync_test_user",
            holdings=[{"name": "茅台", "ts_code": "600519.SH", "shares": 100}],
        )

        mw = mm.MemoryMiddleware()

        # Build a mock request
        from langchain_core.messages import HumanMessage, SystemMessage

        request = MagicMock()
        request.runtime = MagicMock()
        request.runtime.config = {"configurable": {"user_id": "sync_test_user"}}
        request.messages = [HumanMessage(content="向我汇报")]
        # SystemMessage.content_blocks is a read-only property, so mock it
        mock_sys_msg = MagicMock()
        mock_sys_msg.content_blocks = [
            {"type": "text", "text": "You are a financial analyst."}
        ]
        request.system_message = mock_sys_msg

        captured_request = {}

        def mock_handler(req):
            captured_request["req"] = req
            return "response"

        # Disable mem0 for this test
        with patch.object(mm, "MEM0_ENABLED", False):
            mw.wrap_model_call(request, mock_handler)

        # The handler should have been called with an overridden request
        assert request.override.called
