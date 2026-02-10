"""Luna Genesis application entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import yaml

from luna.agent.agent_loop import AgentLoop, AgentLoopConfig
from luna.core.local_llm import LocalLLM, LocalLLMConfig
from luna.core.memory import MemoryStore
from luna.interface.cli import run_cli
from luna.memory.vector_store import VectorStore, VectorStoreConfig
from luna.tools.tool_registry import ToolRegistry
from luna.utils.logger import get_logger


logger = get_logger("luna.main")


class SupportsVectorMemory(Protocol):
    def search(self, query: str, k: int = 3) -> list[str]: ...

    def add_document(self, text: str) -> None: ...


class InMemoryVectorFallback:
    """Fallback store used when vector dependencies are unavailable."""

    def __init__(self) -> None:
        self._docs: list[str] = []

    def search(self, query: str, k: int = 3) -> list[str]:
        query = query.strip().lower()
        if not query:
            return []
        matches = [doc for doc in self._docs if query in doc.lower()]
        return matches[-k:][::-1]

    def add_document(self, text: str) -> None:
        clean = text.strip()
        if clean:
            self._docs.append(clean)


def load_settings(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_llm_from_settings(settings: dict) -> LocalLLM:
    model_cfg = settings.get("model", {})
    deploy_cfg = settings.get("deployment", {})
    quant_cfg = model_cfg.get("quantization", {})

    config = LocalLLMConfig(
        backend=model_cfg.get("backend", "transformers"),
        model_id=model_cfg.get("model_id", "mistralai/Mistral-7B-Instruct-v0.3"),
        context_window=model_cfg.get("context_window", 4096),
        max_new_tokens=model_cfg.get("max_new_tokens", 256),
        temperature=model_cfg.get("temperature", 0.7),
        top_p=model_cfg.get("top_p", 0.9),
        quant_enabled=quant_cfg.get("enabled", False),
        quant_mode=quant_cfg.get("mode", "4bit"),
        preferred_device=deploy_cfg.get("preferred_device", "auto"),
        dtype=model_cfg.get("dtype", "auto"),
    )
    return LocalLLM(config)


def create_vector_store(base_dir: Path, settings: dict) -> SupportsVectorMemory:
    vector_cfg = settings.get("vector_memory", {})
    try:
        return VectorStore(
            VectorStoreConfig(
                index_path=base_dir / vector_cfg.get("index_path", "data/vector.index"),
                docs_path=base_dir / vector_cfg.get("docs_path", "data/vector_docs.json"),
                embedding_model=vector_cfg.get(
                    "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector store unavailable; using fallback memory. Reason: %s", exc)
        return InMemoryVectorFallback()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    settings = load_settings(base_dir / "config" / "settings.yaml")

    logger.info("Loading local model...")
    llm = create_llm_from_settings(settings)
    logger.info("Model loaded: %s", llm.model_info())

    memory_cfg = settings.get("memory", {})
    memory_path = base_dir / memory_cfg.get("file_path", "data/memory.json")
    memory = MemoryStore(file_path=memory_path, max_entries=memory_cfg.get("max_entries", 1000))

    vector_store = create_vector_store(base_dir, settings)

    agent_cfg = settings.get("agent", {})
    tools = ToolRegistry()
    agent = AgentLoop(
        llm=llm,
        tools=tools,
        config=AgentLoopConfig(max_steps=agent_cfg.get("max_steps", 5)),
    )
    rag_top_k = int(agent_cfg.get("rag_top_k", 3))

    def process_input(user_input: str) -> str:
        relevant_knowledge = vector_store.search(user_input, k=rag_top_k)
        response = agent.run_agent(user_input, relevant_knowledge=relevant_knowledge)

        memory.append(user_input, response, intent="agent")
        vector_store.add_document(user_input)
        vector_store.add_document(response)
        return response

    run_cli(process_input)


if __name__ == "__main__":
    main()
