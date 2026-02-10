"""Multi-step agent loop with tool execution."""

from __future__ import annotations

import re
from dataclasses import dataclass

from luna.core.intent import detect_intent
from luna.core.model_interface import ModelInterface
from luna.core.prompts import build_prompt
from luna.tools.tool_registry import ToolRegistry


_ACTION_PATTERN = re.compile(r"^ACTION:(?P<tool>[a-zA-Z_][\w]*)\((?P<args>.*)\)$", re.DOTALL)
_FINAL_PREFIX = "FINAL:"


@dataclass
class AgentLoopConfig:
    max_steps: int = 5


class AgentLoop:
    """Reasoning loop that decides between tool calls and final answers."""

    def __init__(
        self,
        llm: ModelInterface,
        tools: ToolRegistry,
        config: AgentLoopConfig | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.config = config or AgentLoopConfig()

    def _build_reasoning_prompt(
        self,
        user_input: str,
        intent: str,
        relevant_knowledge: list[str],
        transcript: list[str],
    ) -> str:
        base_mode = "code" if intent == "code" else "chat"
        base_prompt = build_prompt(base_mode, user_input)
        tools_text = "\n".join(
            f"- {tool['name']}: {tool['description']}" for tool in self.tools.list_tools()
        )
        rag_text = "\n".join(f"- {item}" for item in relevant_knowledge) if relevant_knowledge else "- (none)"
        history = "\n".join(transcript) if transcript else "(empty)"

        return (
            f"{base_prompt}\n\n"
            "System instructions:\n"
            "Think step by step. Decide if a tool is needed.\n"
            "You may output exactly one of:\n"
            "1) ACTION:tool_name(args)\n"
            "2) FINAL:answer\n\n"
            "Rules:\n"
            "- Prefer FINAL when no tool is required.\n"
            "- For tools with named parameters, pass JSON object in args, e.g. ACTION:file_reader_tool({\"path\":\"./README.md\"}).\n"
            "- For simple single-string arguments, ACTION:calculator_tool(2 + 2 * 3) is valid.\n\n"
            f"Intent: {intent}\n"
            "Available tools:\n"
            f"{tools_text}\n\n"
            "Relevant knowledge:\n"
            f"{rag_text}\n\n"
            "Reasoning transcript:\n"
            f"{history}\n"
        )

    def _parse_action(self, response: str) -> tuple[str, str] | None:
        match = _ACTION_PATTERN.match(response.strip())
        if not match:
            return None
        return match.group("tool"), match.group("args").strip()

    def run_agent(self, user_input: str, relevant_knowledge: list[str] | None = None) -> str:
        intent = detect_intent(user_input)
        knowledge = relevant_knowledge or []
        transcript: list[str] = []

        for step in range(1, self.config.max_steps + 1):
            prompt = self._build_reasoning_prompt(user_input, intent, knowledge, transcript)
            llm_output = self.llm.generate_text(prompt).strip()

            if llm_output.startswith(_FINAL_PREFIX):
                final_answer = llm_output[len(_FINAL_PREFIX) :].strip()
                return final_answer or "I could not determine a final answer."

            action = self._parse_action(llm_output)
            if action:
                tool_name, raw_args = action
                tool_result = self.tools.call_tool(tool_name, raw_args if raw_args else None)
                transcript.append(
                    f"Step {step}: ACTION {tool_name}({raw_args}) -> RESULT {tool_result}"
                )
                continue

            # Fallback for models that do not follow strict format.
            if step == self.config.max_steps:
                return llm_output or "I could not complete the request."
            transcript.append(f"Step {step}: Unstructured output -> {llm_output}")

        return "I could not complete the request within max reasoning steps."
