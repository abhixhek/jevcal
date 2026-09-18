"""Decision providers. Every provider answers typed questions about a state.

Normalized response:
    {"model": str, "answers": {qid: {"type", "answer", "probabilities", "confidence"}},
     "usage": {"input_tokens": int, "output_tokens": int}, "latency_ms": float}
"""

from __future__ import annotations

from typing import Any

from ..spec import Question


class ProviderError(RuntimeError):
    pass


class Provider:
    name = "base"
    cacheable = True

    def ask(self, state: Any, questions: dict[str, Question], row: dict | None = None) -> dict:
        raise NotImplementedError

    def cache_key(self) -> dict:
        return {"provider": self.name}

    def row_key(self, row: dict):
        """Anything beyond state + questions that the answer depends on. None for real models."""
        return None


def build_provider(kind: str, *, model: str = "jev-latest", llm: str | None = None, seed: int = 7) -> Provider:
    if kind == "typesafe":
        from .typesafe import TypeSafeProvider

        return TypeSafeProvider(model=model)
    if kind == "sim":
        from .sim import SimProvider

        return SimProvider(seed=seed)
    if kind == "llm":
        from .llm import LLMProvider, build_llm

        return LLMProvider(build_llm(llm))
    raise ProviderError(f"unknown provider {kind!r} (expected typesafe, llm, or sim)")
