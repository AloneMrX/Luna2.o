"""Persistent conversation memory backed by JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class MemoryStore:
    """Simple persistent conversation storage."""

    file_path: Path
    max_entries: int = 1000

    def __post_init__(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")

    def load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def append(self, user_text: str, assistant_text: str, intent: str) -> None:
        items = self.load()
        items.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "intent": intent,
                "user": user_text,
                "assistant": assistant_text,
            }
        )
        if len(items) > self.max_entries:
            items = items[-self.max_entries :]
        self.file_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
