from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from truss.types import AgentEnvelope, ContextBlock, ContextWeight


@dataclass
class BudgetCarve:
    _kind: str
    _value: float

    @staticmethod
    def fixed_usd(amount: float) -> "BudgetCarve":
        return BudgetCarve(_kind="usd", _value=amount)

    @staticmethod
    def percent(fraction: float) -> "BudgetCarve":
        return BudgetCarve(_kind="pct", _value=fraction)

    @staticmethod
    def fixed_tokens(tokens: int) -> "BudgetCarve":
        return BudgetCarve(_kind="tokens", _value=float(tokens))

    def apply(self, parent_budget: Optional[float]) -> Optional[float]:
        if parent_budget is None:
            if self._kind == "usd":
                return self._value
            return None  # no limit → pass through None
        if self._kind == "usd":
            return min(self._value, parent_budget)
        if self._kind == "pct":
            return parent_budget * self._value
        return parent_budget * 0.5  # tokens→usd: safe 50% default


def pack(
    parent: AgentEnvelope,
    task: str,
    carry_weights: list[ContextWeight],
    budget_carve: BudgetCarve,
) -> AgentEnvelope:
    child = AgentEnvelope(
        task=task,
        budget_usd_remaining=budget_carve.apply(parent.budget_usd_remaining),
        parent_agent=str(parent.id),
        model_hint=parent.model_hint,
    )
    child.context = [b for b in parent.context if b.weight in carry_weights]
    return child


def unpack(envelope: AgentEnvelope) -> list[ContextBlock]:
    return list(envelope.context)
