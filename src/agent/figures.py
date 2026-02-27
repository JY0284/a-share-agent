"""Figure storage service for persisting matplotlib figures per thread.

Figures are saved to the filesystem and served via API. This keeps base64 data
out of LLM context while allowing the frontend to display figures.

Directory structure:
    assets/
        {thread_id}/
            {figure_id}.png
            metadata.json  # index of all figures in thread

Usage:
    from agent.figures import save_figure, get_figure_url, get_thread_figures

    # In sandbox.py after capturing matplotlib figure:
    fig_id = save_figure(thread_id, image_bytes, title="My Chart")
    url = get_figure_url(thread_id, fig_id)  # /api/figures/{thread_id}/{fig_id}

    # Agent can reference: [[fig:{fig_id}]]
"""
import base64
import json
import os
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import TypedDict

# Default assets directory (can be overridden via env)
ASSETS_DIR = os.environ.get("AGENT_ASSETS_DIR", "./assets")

# Current thread ID (set by middleware)
_current_thread_id: ContextVar[str | None] = ContextVar("figure_thread_id", default=None)


class FigureMetadata(TypedDict):
    """Metadata for a saved figure."""
    id: str
    title: str
    format: str
    created_at: str
    tool_call_id: str | None
    width: int | None
    height: int | None


class ThreadFigures(TypedDict):
    """Index of all figures in a thread."""
    thread_id: str
    figures: list[FigureMetadata]
    updated_at: str


def set_thread_id(thread_id: str) -> None:
    """Set the current thread ID for figure storage."""
    _current_thread_id.set(thread_id)


def get_thread_id() -> str | None:
    """Get the current thread ID."""
    return _current_thread_id.get()


def _get_thread_dir(thread_id: str) -> Path:
    """Get the directory for a thread's assets."""
    return Path(ASSETS_DIR) / thread_id


def _get_metadata_path(thread_id: str) -> Path:
    """Get the path to the thread's metadata file."""
    return _get_thread_dir(thread_id) / "metadata.json"


def _load_thread_metadata(thread_id: str) -> ThreadFigures:
    """Load or create the metadata for a thread."""
    meta_path = _get_metadata_path(thread_id)
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "thread_id": thread_id,
        "figures": [],
        "updated_at": datetime.utcnow().isoformat(),
    }


def _save_thread_metadata(thread_id: str, metadata: ThreadFigures) -> None:
    """Save the thread metadata."""
    meta_path = _get_metadata_path(thread_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    metadata["updated_at"] = datetime.utcnow().isoformat()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_figure(
    image_base64: str,
    title: str = "",
    format: str = "png",
    tool_call_id: str | None = None,
    thread_id: str | None = None,
) -> FigureMetadata:
    """Save a figure to the thread's assets directory.
    
    Args:
        image_base64: Base64-encoded image data
        title: Figure title (from plt.title)
        format: Image format (default: png)
        tool_call_id: ID of the tool call that generated this figure
        thread_id: Thread ID (uses current thread if not provided)
    
    Returns:
        FigureMetadata with id, title, and other info
    """
    # Get thread ID
    tid = thread_id or get_thread_id()
    if not tid:
        # Fallback to a default thread if not set
        tid = "default"
    
    # Generate figure ID
    fig_id = f"fig_{uuid.uuid4().hex[:8]}"
    
    # Decode and save image
    image_bytes = base64.b64decode(image_base64)
    thread_dir = _get_thread_dir(tid)
    thread_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = thread_dir / f"{fig_id}.{format}"
    with open(fig_path, "wb") as f:
        f.write(image_bytes)
    
    # Create metadata
    metadata: FigureMetadata = {
        "id": fig_id,
        "title": title,
        "format": format,
        "created_at": datetime.utcnow().isoformat(),
        "tool_call_id": tool_call_id,
        "width": None,
        "height": None,
    }
    
    # Try to get image dimensions
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        metadata["width"] = img.width
        metadata["height"] = img.height
    except ImportError:
        pass  # PIL not available
    except Exception:
        pass  # Failed to read image
    
    # Update thread metadata
    thread_meta = _load_thread_metadata(tid)
    thread_meta["figures"].append(metadata)
    _save_thread_metadata(tid, thread_meta)
    
    return metadata


def get_figure_path(thread_id: str, figure_id: str, format: str = "png") -> Path | None:
    """Get the filesystem path to a figure.
    
    Args:
        thread_id: Thread ID
        figure_id: Figure ID
        format: Image format
    
    Returns:
        Path to the figure file, or None if not found
    """
    fig_path = _get_thread_dir(thread_id) / f"{figure_id}.{format}"
    if fig_path.exists():
        return fig_path
    return None


def get_figure_url(thread_id: str, figure_id: str) -> str:
    """Get the API URL for a figure.
    
    Args:
        thread_id: Thread ID
        figure_id: Figure ID
    
    Returns:
        URL path like /api/figures/{thread_id}/{figure_id}
    """
    return f"/api/figures/{thread_id}/{figure_id}"


def get_thread_figures(thread_id: str) -> list[FigureMetadata]:
    """Get all figures for a thread.
    
    Args:
        thread_id: Thread ID
    
    Returns:
        List of figure metadata
    """
    meta = _load_thread_metadata(thread_id)
    return meta["figures"]


def get_figure_metadata(thread_id: str, figure_id: str) -> FigureMetadata | None:
    """Get metadata for a specific figure.
    
    Args:
        thread_id: Thread ID
        figure_id: Figure ID
    
    Returns:
        FigureMetadata or None if not found
    """
    figures = get_thread_figures(thread_id)
    for fig in figures:
        if fig["id"] == figure_id:
            return fig
    return None


def format_figure_reference(figure_id: str, title: str = "") -> str:
    """Format a figure reference for the agent to use in responses.
    
    The agent should include this in its response text, and the frontend
    will render it as an interactive thumbnail.
    
    Args:
        figure_id: Figure ID
        title: Optional title for display
    
    Returns:
        Reference string like [[fig:fig_abc123|Title]]
    """
    if title:
        return f"[[fig:{figure_id}|{title}]]"
    return f"[[fig:{figure_id}]]"
