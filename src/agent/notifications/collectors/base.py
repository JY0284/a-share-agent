"""Base class for A-share quant collectors.

Extends spiderman's BaseCollector with stock_data store access and
user profile helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from spiderman.collectors.base import BaseCollector
from spiderman.core.config import SpidermanConfig

logger = logging.getLogger(__name__)


class QuantBaseCollector(BaseCollector):
    """Shared base for all quant collectors — adds store + profile access."""

    def __init__(self, config: SpidermanConfig) -> None:
        self._store_dir = Path(config.stock_data_store_dir).resolve()
        self._profiles_dir = Path(config.user_profiles_dir).resolve()
        super().__init__(config)

    def _list_user_ids(self) -> list[str]:
        """List all user IDs that have profile files."""
        if not self._profiles_dir.is_dir():
            return []
        return [
            p.stem
            for p in self._profiles_dir.glob("*.json")
            if p.stem != "__template__"
        ]

    def _load_user_profile(self, user_id: str) -> dict | None:
        """Load a user profile JSON. Returns None if not found."""
        path = self._profiles_dir / f"{user_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self.logger.warning("Failed to load profile %s: %s", user_id, exc)
            return None

    def _open_store(self):
        """Open the stock_data store."""
        from stock_data.store import open_store

        return open_store(str(self._store_dir))
