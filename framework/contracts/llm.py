"""LLM — model contract over OpenAI / Claude / DeepSeek / Qwen / Gemini.

Per Principle 7 (AI Assists, Never Fabricates): an LLM here only ASSISTS
research. Any factor or conclusion it produces MUST be recorded in the
Registry and independently re-evaluated by the Evaluator (DSR/BH-FDR) before
being marked trustworthy. The LLM never writes conclusions into a report
directly.

Reference Implementations register against a specific provider name.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    @property
    def provider(self) -> str:
        ...

    def complete(self, prompt: str, **kwargs) -> str:
        ...

    def embed(self, text: str, **kwargs) -> list[float]:
        ...
