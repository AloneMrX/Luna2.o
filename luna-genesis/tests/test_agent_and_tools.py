from __future__ import annotations

import unittest

from luna.agent.agent_loop import AgentLoop
from luna.core.model_interface import ModelInterface
from luna.tools.tool_registry import ToolRegistry


class ScriptedLLM(ModelInterface):
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self._cursor = 0

    def generate_text(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        if self._cursor >= len(self.outputs):
            return "FINAL:done"
        out = self.outputs[self._cursor]
        self._cursor += 1
        return out

    def model_info(self) -> dict:
        return {"name": "scripted"}


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()

    def test_calculator_plain_text_args(self) -> None:
        result = self.registry.call_tool("calculator_tool", "2 + 3 * 4")
        self.assertEqual(result, "14")

    def test_file_reader_dict_args_parsing(self) -> None:
        result = self.registry.call_tool("file_reader_tool", '{"path": "missing.txt"}')
        self.assertIn("file does not exist", result)


class AgentLoopTests(unittest.TestCase):
    def test_action_then_final(self) -> None:
        llm = ScriptedLLM([
            "ACTION:calculator_tool(2 + 2)",
            "FINAL:4",
        ])
        agent = AgentLoop(llm=llm, tools=ToolRegistry())
        answer = agent.run_agent("what is 2+2")
        self.assertEqual(answer, "4")


if __name__ == "__main__":
    unittest.main()
