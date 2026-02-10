"""Tool registry and built-in tools for Luna Genesis."""

from __future__ import annotations

import ast
import json
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from luna.core.math_engine import evaluate_expression


@dataclass
class Tool:
    """Describes a callable tool function."""

    name: str
    description: str
    function: Callable[..., str]


def calculator_tool(expression: str) -> str:
    """Evaluate arithmetic expressions safely."""
    return evaluate_expression(expression)


def file_reader_tool(path: str) -> str:
    """Read a UTF-8 text file and return a bounded snippet."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        return "File reader error: file does not exist."

    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "File reader error: unable to read file as UTF-8 text."

    preview = content[:4000]
    suffix = "\n...[truncated]" if len(content) > 4000 else ""
    return preview + suffix


def system_info_tool() -> str:
    """Return basic host information for diagnostics."""
    total, used, free = shutil.disk_usage("/")
    return (
        f"System={platform.system()} {platform.release()} | "
        f"Machine={platform.machine()} | "
        f"Disk(total={total // (1024**3)}GB, used={used // (1024**3)}GB, free={free // (1024**3)}GB)"
    )


class ToolRegistry:
    """Registry for discoverable and callable tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            Tool(
                name="calculator_tool",
                description="Safely evaluate an arithmetic expression.",
                function=calculator_tool,
            )
        )
        self.register(
            Tool(
                name="file_reader_tool",
                description="Read the content of a local UTF-8 text file.",
                function=file_reader_tool,
            )
        )
        self.register(
            Tool(
                name="system_info_tool",
                description="Return system and disk information.",
                function=system_info_tool,
            )
        )

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]

    def _parse_tool_args(self, args: Any | None) -> tuple[list[Any], dict[str, Any]]:
        if args is None:
            return [], {}
        if isinstance(args, dict):
            return [], args
        if isinstance(args, (list, tuple)):
            return list(args), {}
        if not isinstance(args, str):
            return [args], {}

        normalized = args.strip()
        if not normalized:
            return [], {}

        # Allow quoted string arguments.
        if (normalized.startswith('"') and normalized.endswith('"')) or (
            normalized.startswith("'") and normalized.endswith("'")
        ):
            return [normalized[1:-1]], {}

        # Prefer JSON for structured arguments.
        try:
            parsed_json = json.loads(normalized)
            if isinstance(parsed_json, dict):
                return [], parsed_json
            if isinstance(parsed_json, list):
                return parsed_json, {}
            return [parsed_json], {}
        except json.JSONDecodeError:
            pass

        # Fallback to Python literals for convenience: {'path': '/tmp/a.txt'}
        try:
            parsed_literal = ast.literal_eval(normalized)
            if isinstance(parsed_literal, dict):
                return [], parsed_literal
            if isinstance(parsed_literal, (list, tuple)):
                return list(parsed_literal), {}
            return [parsed_literal], {}
        except (ValueError, SyntaxError):
            pass

        # Final fallback: raw string.
        return [normalized], {}

    def call_tool(self, name: str, args: Any | None = None) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Tool error: unknown tool '{name}'."

        try:
            positional_args, keyword_args = self._parse_tool_args(args)
            return tool.function(*positional_args, **keyword_args)
        except Exception as exc:  # noqa: BLE001
            return f"Tool error: {exc}"
