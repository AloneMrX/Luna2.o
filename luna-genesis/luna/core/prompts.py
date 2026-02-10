"""Prompt templates used across Luna Genesis modes."""

from __future__ import annotations

CHAT_PERSONA_PROMPT = """You are Luna, an advanced local AI assistant.
You are concise, practical, and honest about uncertainty.
Answer clearly and helpfully."""

CODING_EXPERT_PROMPT = """You are Luna in coding expert mode.
Provide robust, production-minded engineering guidance with secure defaults.
Include short examples when useful."""

REASONING_MODE_PROMPT = """You are Luna in deep reasoning mode.
Break hard problems into steps, keep assumptions explicit, and verify outputs."""


def build_prompt(mode: str, user_input: str) -> str:
    """Build a task prompt by mode."""
    mode = (mode or "chat").lower()
    if mode == "code":
        header = CODING_EXPERT_PROMPT
    elif mode == "reason":
        header = REASONING_MODE_PROMPT
    else:
        header = CHAT_PERSONA_PROMPT

    return f"{header}\n\nUser: {user_input}\nAssistant:"
