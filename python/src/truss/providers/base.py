from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class LLMMessage:
    role: str   # "user" | "assistant" | "system"
    content: str


@dataclass
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: LLMUsage
    raw: Any = None


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        **opts: Any,
    ) -> LLMResponse: ...


COST_TABLE: dict[str, tuple[float, float]] = {
    # (input_$/1k, output_$/1k)
    "claude-haiku-4-5":  (0.001,   0.005),
    "claude-sonnet-4-6": (0.003,   0.015),
    "claude-opus-4-8":   (0.015,   0.075),
    "gpt-4o-mini":       (0.00015, 0.0006),
    "gpt-4o":            (0.005,   0.015),
    "gpt-4-turbo":       (0.010,   0.030),
    "o1":                (0.015,   0.060),
    "o1-mini":           (0.003,   0.012),
}

_DEFAULT_RATES = (0.001, 0.005)


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_TABLE.get(model, _DEFAULT_RATES)
    return (input_tokens / 1000) * rates[0] + (output_tokens / 1000) * rates[1]
