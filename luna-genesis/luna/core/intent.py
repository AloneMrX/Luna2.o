"""Intent detection and routing helpers."""

from __future__ import annotations

import re

INTENTS = ("math", "code", "system", "search", "chat")


MATH_PATTERN = re.compile(r"^[\d\s\+\-\*\/\%\(\)\.\^]+$")
CODE_KEYWORDS = {"code", "python", "bug", "function", "class", "refactor", "algorithm"}
SYSTEM_KEYWORDS = {"shutdown", "restart", "status", "disk", "memory", "cpu", "system"}
SEARCH_KEYWORDS = {"search", "find", "look up", "latest", "news", "web"}


def detect_intent(user_input: str) -> str:
    """Detect high-level intent using lightweight rule-based routing."""
    text = (user_input or "").strip().lower()
    if not text:
        return "chat"

    if MATH_PATTERN.fullmatch(text):
        return "math"

    if any(keyword in text for keyword in SEARCH_KEYWORDS):
        return "search"

    if any(keyword in text for keyword in SYSTEM_KEYWORDS):
        return "system"

    if any(keyword in text for keyword in CODE_KEYWORDS):
        return "code"

    return "chat"
