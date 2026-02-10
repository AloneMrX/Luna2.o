"""Abstract model interface for local language model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelInterface(ABC):
    """Defines the contract for text generation backends."""

    @abstractmethod
    def generate_text(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        """Generate text from a prompt."""

    @abstractmethod
    def model_info(self) -> dict:
        """Return metadata about the loaded model and backend."""
