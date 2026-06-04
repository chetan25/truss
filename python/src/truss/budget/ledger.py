from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from truss.budget.config import BudgetWindow


@dataclass
class LedgerEntry:
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    id: UUID = field(default_factory=uuid4)
    user_id: str | None = None
    agent_name: str | None = None
    timestamp: int = 0
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class UsageReport:
    key: str
    total_tokens: int
    total_cost_usd: float
    window: BudgetWindow
    remaining_tokens: int | None = None
    remaining_usd: float | None = None
    pct_used: float = 0.0


@runtime_checkable
class LedgerStore(Protocol):
    def record(self, entry: LedgerEntry) -> None: ...
    def usage(self, key: str, window: BudgetWindow) -> UsageReport: ...
    def flush(self) -> None: ...
