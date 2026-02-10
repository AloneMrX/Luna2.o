"""Simple web search client with graceful offline fallback."""

from __future__ import annotations

from urllib.parse import urlencode

import requests


def search_web(query: str, endpoint: str, timeout_seconds: int = 10) -> str:
    if not query.strip():
        return "Search query is empty."

    params = {"q": query}
    url = f"{endpoint}?{urlencode(params)}"

    try:
        response = requests.get(url, timeout=timeout_seconds, headers={"User-Agent": "Luna-Genesis/0.1"})
        response.raise_for_status()
    except requests.RequestException:
        return "Web search is unavailable right now (offline or endpoint unreachable)."

    text = response.text
    snippet = text[:500].replace("\n", " ").strip()
    return f"Search results snapshot for '{query}': {snippet}"
