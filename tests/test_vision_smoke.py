"""Quick smoke-test for the VisionMiddleware — sends a stock screenshot to GPT-5.2."""

import base64
import os
import sys
import textwrap


# Minimal test: call the vision model directly with a sample image
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["VISION_API_KEY"],
    base_url=os.environ["VISION_BASE_URL"],
)

# Create a simple test image (a colored rectangle with text-like content)
# For a real test we'd use a stock screenshot, but let's first verify the API works
print("=" * 60)
print("Vision Middleware Smoke Test — GPT-5.2 via yunwu.ai")
print("=" * 60)

# Test 1: Basic API connectivity with a text-only vision call
print("\n[Test 1] Basic API connectivity...")
try:
    resp = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "user", "content": "Say 'hello' in Chinese. One word only."},
        ],
        max_tokens=10,
    )
    result = resp.choices[0].message.content
    print(f"  Response: {result}")
    print(f"  Usage: {resp.usage}")
    print("  ✓ API connection works!")
except Exception as e:
    print(f"  ✗ API error: {e}")
    sys.exit(1)

# Test 2: Vision capability — generate a minimal PNG with stock-like data
print("\n[Test 2] Vision capability with generated image...")

# Create a minimal PNG image with stock data text using pure Python
# (We'll use matplotlib if available, otherwise create a minimal valid PNG)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")
    fig.set_facecolor("#1a1a2e")

    # Simulated stock holding data
    stock_text = textwrap.dedent("""\
        持仓明细
        ─────────────────────────
        贵州茅台 600519  1798.50  +2.35%
        持仓: 100股  成本: 1720.00  盈亏: +7,850.00

        宁德时代 300750  218.60  -1.02%
        持仓: 500股  成本: 230.00  盈亏: -5,700.00

        中国平安 601318  52.80  +0.57%
        持仓: 1000股  成本: 48.50  盈亏: +4,300.00
        ─────────────────────────
        总资产: 338,060.00
        总盈亏: +6,450.00 (+1.95%)
        可用资金: 50,000.00
    """)

    ax.text(0.05, 0.95, stock_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", color="white",
            fontfamily="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    print(f"  Generated test image: {len(img_b64)} bytes (base64)")

except ImportError:
    # Fallback: use a 1x1 red PNG
    # Minimal valid PNG
    import struct, zlib
    def make_minimal_png():
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data)
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        raw = zlib.compress(b'\x00\xff\x00\x00')
        idat_crc = zlib.crc32(b'IDAT' + raw)
        idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + struct.pack('>I', idat_crc)
        iend_crc = zlib.crc32(b'IEND')
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        return sig + ihdr + idat + iend
    img_b64 = base64.b64encode(make_minimal_png()).decode()
    print("  Using minimal test PNG (matplotlib not available)")

# Send to vision model
try:
    resp = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "You are a precise OCR assistant. Extract all visible text and data from images."},
            {"role": "user", "content": [
                {"type": "text", "text": "请提取这张图片中的所有金融数据，输出结构化文本。"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high",
                }},
            ]},
        ],
        max_tokens=2000,
        temperature=0.1,
    )
    result = resp.choices[0].message.content
    print(f"  Vision response ({len(result)} chars):")
    print(textwrap.indent(result, "    "))
    print(f"  Usage: {resp.usage}")
    print("  ✓ Vision works!")
except Exception as e:
    print(f"  ✗ Vision error: {e}")
    sys.exit(1)

# Test 3: Full middleware pipeline
print("\n[Test 3] Full VisionMiddleware pipeline...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent.vision_middleware import (
    _call_vision_model,
    _extract_images,
    _to_openai_image_content,
    VISION_ENABLED,
    VISION_MODEL,
    VISION_BASE_URL,
)

print(f"  VISION_ENABLED={VISION_ENABLED}")
print(f"  VISION_MODEL={VISION_MODEL}")
print(f"  VISION_BASE_URL={VISION_BASE_URL}")

# Simulate a multimodal message content array
content_blocks = [
    {"type": "text", "text": "帮我分析一下我的持仓"},
    {"type": "image", "mimeType": "image/png", "data": img_b64, "metadata": {"name": "holdings.png"}},
]

images, others = _extract_images(content_blocks)
print(f"  Extracted {len(images)} image(s), {len(others)} other block(s)")

user_text = " ".join(b.get("text", "") for b in others if isinstance(b, dict) and b.get("type") == "text")
extracted = _call_vision_model(images, user_text)
print(f"  Middleware extraction ({len(extracted)} chars):")
print(textwrap.indent(extracted[:500], "    "))
if len(extracted) > 500:
    print(f"    ... ({len(extracted) - 500} more chars)")
print("  ✓ Middleware pipeline works!")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
