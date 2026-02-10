"""Local LLM engine for transformers and llama.cpp backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from luna.core.model_interface import ModelInterface


@dataclass
class LocalLLMConfig:
    backend: str = "transformers"
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    context_window: int = 4096
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    quant_enabled: bool = False
    quant_mode: str = "4bit"
    preferred_device: str = "auto"
    dtype: str = "auto"


class LocalLLM(ModelInterface):
    """Single-load local LLM wrapper with hardware-aware initialization."""

    SUPPORTED_MODELS = {
        "llama": "LLaMA 3.3 70B",
        "gpt-oss-20b": "GPT-OSS 20B",
        "gpt-oss-120b": "GPT-OSS 120B",
        "mistral": "Mistral 7B",
        "deepseek": "DeepSeek V3",
    }

    def __init__(self, config: LocalLLMConfig) -> None:
        self.config = config
        self.device = self._detect_device(config.preferred_device)
        self.backend = config.backend.lower()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._llama_cpp: Any | None = None
        self._load_model_once()

    def _detect_device(self, preferred: str) -> str:
        preferred = (preferred or "auto").lower()
        if preferred == "cpu":
            return "cpu"
        if preferred == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self) -> Any:
        mode = (self.config.dtype or "auto").lower()
        if mode == "float16":
            return torch.float16
        if mode == "bfloat16":
            return torch.bfloat16
        if mode == "float32":
            return torch.float32
        if self.device == "cuda":
            return torch.float16
        return torch.float32

    def _load_model_once(self) -> None:
        if self.backend == "llama_cpp":
            self._load_llama_cpp()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_kwargs: dict[str, Any] = {}
        quant_mode = self.config.quant_mode.lower()

        if self.config.quant_enabled and quant_mode in {"4bit", "8bit"}:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quant_mode == "4bit",
                load_in_8bit=quant_mode == "8bit",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=self._resolve_dtype(),
            device_map="auto" if self.device == "cuda" else None,
            **model_kwargs,
        )

        if self.device == "cpu":
            self._model.to("cpu")

    def _load_llama_cpp(self) -> None:
        from llama_cpp import Llama

        model_path = Path(self.config.model_id)
        if not model_path.exists():
            raise FileNotFoundError(
                "For llama_cpp backend, model_id must be a local GGUF file path."
            )

        gpu_layers = -1 if self.device == "cuda" else 0
        self._llama_cpp = Llama(
            model_path=str(model_path),
            n_ctx=self.config.context_window,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )

    def generate_text(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        max_tokens = max_new_tokens or self.config.max_new_tokens
        if self.backend == "llama_cpp":
            output = self._llama_cpp(
                prompt,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=["User:"],
            )
            return output["choices"][0]["text"].strip()

        encoded = self._tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            encoded = {k: v.to("cuda") for k, v in encoded.items()}

        generated = self._model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        output = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        return output[len(prompt) :].strip() if output.startswith(prompt) else output.strip()

    def model_info(self) -> dict:
        return {
            "backend": self.backend,
            "model_id": self.config.model_id,
            "device": self.device,
            "quantization": {
                "enabled": self.config.quant_enabled,
                "mode": self.config.quant_mode,
            },
            "supported_families": self.SUPPORTED_MODELS,
        }
