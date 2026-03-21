"""Vision middleware — pre-processes images using a vision-capable model.

DeepSeek-chat is text-only, so images uploaded by users (e.g. stock holding
screenshots from brokerage apps) cannot be processed directly.  This middleware:

1. Detects image content blocks in the last HumanMessage.
2. Sends them to a vision-capable model (configured via env vars).
3. Extracts structured text (holdings, P&L, prices, etc.).
4. Replaces image blocks with the extracted text, so DeepSeek can analyze it.

Configuration (environment variables):
  VISION_API_KEY   — API key for the vision model provider
  VISION_MODEL     — Model name (default: gpt-5.2)
  VISION_BASE_URL  — Base URL for OpenAI-compatible API (default: https://yunwu.ai/v1)
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from agent.usage_cost import estimate_cost

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read lazily so .env is loaded before first use
# ---------------------------------------------------------------------------


def _vision_api_key() -> str:
    return os.environ.get("VISION_API_KEY", "")


def _vision_model() -> str:
    return os.environ.get("VISION_MODEL", "gpt-5.2")


def _vision_base_url() -> str:
    return os.environ.get("VISION_BASE_URL", "https://yunwu.ai/v1")


def VISION_ENABLED() -> bool:  # noqa: N802  — kept as upper-case for back-compat
    return bool(_vision_api_key())

# ---------------------------------------------------------------------------
# Vision prompts for financial screenshots
# ---------------------------------------------------------------------------

_VISION_SYSTEM_PROMPT = (
    "You are a precise OCR assistant specialized in extracting financial data from "
    "Chinese stock/brokerage app screenshots (e.g. 东方财富、同花顺、雪球、通达信、富途、老虎). "
    "Extract ALL visible information into structured text. Be thorough and accurate. "
    "Output in Chinese, matching the original text exactly."
)

_VISION_USER_PROMPT = """\
请仔细分析这张图片，提取所有可见的金融数据。输出结构化文本：

1. **持仓明细**（如有）：每只股票/基金列出：
   - 名称、代码
   - 持仓数量/份额
   - 成本价、现价
   - 盈亏金额、盈亏比例
   - 市值

2. **账户概览**（如有）：
   - 总资产、总市值
   - 总盈亏（金额+比例）
   - 可用资金/余额

3. **行情数据**（如有）：
   - 股票名称、代码、当前价格、涨跌幅
   - 成交量、换手率等

4. **其他信息**：图中任何其他可见的金融数据或文字。

注意：准确提取数字，不要遗漏、不要编造。如果某项信息不可见，不要猜测。"""

# ---------------------------------------------------------------------------
# Module-level cache to avoid re-processing the same images across agent
# loop iterations (each tool-call cycle triggers a new model call, but
# the original HumanMessage with images stays the same).
# ---------------------------------------------------------------------------

_vision_cache: dict[str, str] = {}

# ---------------------------------------------------------------------------
# ContextVar bridge: vision costs flow to LocalTraceMiddleware
# ---------------------------------------------------------------------------

_pending_vision_costs: ContextVar[list[dict] | None] = ContextVar(
    "pending_vision_costs", default=None
)

# Original image metadata so TraceMiddleware can log what images arrived
_pending_image_info: ContextVar[dict | None] = ContextVar(
    "pending_image_info", default=None
)


def consume_pending_vision_costs() -> list[dict]:
    """Return and clear all pending vision costs for the current context.

    Called by LocalTraceMiddleware after each model call to pick up
    vision API costs incurred during request pre-processing.
    """
    costs = _pending_vision_costs.get(None) or []
    _pending_vision_costs.set(None)
    return costs


def consume_pending_image_info() -> dict | None:
    """Return and clear original image metadata for the current context.

    Called by LocalTraceMiddleware to log what images the user uploaded
    (before VisionMiddleware strips them from the messages).
    """
    info = _pending_image_info.get(None)
    _pending_image_info.set(None)
    return info


def _set_image_info(image_blocks: list[dict]) -> None:
    """Record original image metadata (count + mimeTypes) for trace logging."""
    _pending_image_info.set({
        "count": len(image_blocks),
        "mimeTypes": [b.get("mimeType", "image/*") for b in image_blocks],
    })


def _add_pending_vision_cost(entry: dict) -> None:
    """Accumulate a vision cost entry for the current execution context."""
    costs = _pending_vision_costs.get(None)
    if costs is None:
        costs = []
        _pending_vision_costs.set(costs)
    costs.append(entry)


def _cache_key(images: list[dict]) -> str:
    """Create a lightweight fingerprint from image blocks."""
    parts = []
    for img in images:
        data = img.get("data", "")
        # Use prefix + length as fingerprint (fast, collision-resistant enough)
        prefix = data[:64] if isinstance(data, str) else str(data)[:64]
        parts.append(f"{prefix}_{len(data) if isinstance(data, str) else 0}")
    return "|".join(sorted(parts))


# ---------------------------------------------------------------------------
# OpenAI-compatible client (lazy)
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """Lazy-init OpenAI-compatible client for vision calls."""
    global _client
    if _client is not None:
        return _client
    try:
        from openai import OpenAI
    except ImportError:
        _logger.warning("openai package not installed — vision middleware disabled")
        return None
    _client = OpenAI(api_key=_vision_api_key(), base_url=_vision_base_url())
    return _client


# ---------------------------------------------------------------------------
# Image extraction helpers
# ---------------------------------------------------------------------------


def _extract_images(content: list) -> tuple[list[dict], list[dict]]:
    """Separate image blocks from non-image blocks.

    Returns (image_blocks, other_blocks).
    Handles both LangChain-native (type=image) and OpenAI-style (type=image_url).
    """
    images: list[dict] = []
    others: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            others.append(block)
            continue
        btype = block.get("type", "")
        if btype in ("image", "image_url"):
            images.append(block)
        else:
            others.append(block)
    return images, others


def _to_openai_image_content(block: dict) -> dict:
    """Convert an image content block to OpenAI vision API format."""
    btype = block.get("type", "")

    if btype == "image_url":
        return block

    if btype == "image":
        mime = block.get("mimeType", "image/png")
        data = block.get("data", "")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}", "detail": "high"},
        }

    return block


# ---------------------------------------------------------------------------
# Vision model call
# ---------------------------------------------------------------------------


def _call_vision_model(image_blocks: list[dict], user_text: str = "") -> str:
    """Send images to the vision model and return extracted text."""
    client = _get_client()
    if client is None:
        return ""

    # Build content array for the vision API
    api_content: list[dict] = []

    prompt = _VISION_USER_PROMPT
    if user_text.strip():
        prompt = f"用户的问题/指令：{user_text.strip()}\n\n{prompt}"
    api_content.append({"type": "text", "text": prompt})

    for img in image_blocks:
        api_content.append(_to_openai_image_content(img))

    try:
        model = _vision_model()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                {"role": "user", "content": api_content},
            ],
            max_tokens=2000,
            temperature=0.1,
        )
        result = response.choices[0].message.content or ""

        # Capture usage and compute cost for billing
        if response.usage:
            usage_info = {
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
                "model": model,
            }
            cost_info = estimate_cost(usage_info, model_name=model)
            _add_pending_vision_cost({
                "model": model,
                "images": len(image_blocks),
                "usage": usage_info,
                "cost": cost_info,
            })

        _logger.info(
            "Vision analysis complete: model=%s, images=%d, output_chars=%d",
            model,
            len(image_blocks),
            len(result),
        )
        return result
    except Exception as exc:
        _logger.error("Vision model call failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class VisionMiddleware(AgentMiddleware[Any, Any]):
    """Pre-processes images in user messages via a vision-capable model.

    Replaces image content blocks with structured text so that text-only
    models (like DeepSeek) can analyze the extracted data.
    """

    tools: list = []

    _MARKER = "[Image Analysis]"

    def _process_request(self, request: ModelRequest) -> ModelRequest:
        """Strip images from the last HumanMessage; replace with vision text.

        Image blocks MUST always be removed — text-only models (DeepSeek)
        reject unknown content types with HTTP 400.  If vision extraction
        fails or is disabled, a fallback note is inserted instead.
        """
        messages = list(request.messages or [])

        # Find the last HumanMessage
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, HumanMessage):
                continue
            content = msg.content
            if not isinstance(content, list):
                break  # String content — no images possible

            # Already processed?
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and self._MARKER in block.get("text", "")
                ):
                    return request

            images, others = _extract_images(content)
            if not images:
                break

            # Record image metadata for trace logging before stripping
            _set_image_info(images)

            # Try vision extraction (if enabled)
            extracted = ""
            if VISION_ENABLED():
                key = _cache_key(images)
                cached = _vision_cache.get(key)
                if cached is not None:
                    extracted = cached
                else:
                    user_text = " ".join(
                        b.get("text", "")
                        for b in others
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    extracted = _call_vision_model(images, user_text)
                    if extracted:
                        _vision_cache[key] = extracted

            # Always strip images — build replacement content
            new_content = list(others)
            if extracted:
                new_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"\n{self._MARKER}\n"
                            f"以下是从用户上传的图片中提取的信息：\n\n{extracted}\n"
                        ),
                    }
                )
            else:
                _logger.warning(
                    "Could not extract image content (enabled=%s, images=%d); "
                    "stripping image blocks with fallback note",
                    VISION_ENABLED(), len(images),
                )
                new_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"\n{self._MARKER}\n"
                            "用户上传了图片，但图片解析服务暂时不可用，"
                            "无法提取图片内容。请告知用户稍后重试或手动输入信息。\n"
                        ),
                    }
                )

            new_msg = HumanMessage(content=new_content)
            new_messages = messages[:i] + [new_msg] + messages[i + 1:]
            return request.override(messages=new_messages)

        return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        return handler(self._process_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        modified = await asyncio.to_thread(self._process_request, request)
        return await handler(modified)
