from __future__ import annotations

import threading
from truss.budget.config import BudgetWindow
from truss.budget.ledger import LedgerEntry, UsageReport


class InMemoryStore:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._lock = threading.RLock()

    def record(self, entry: LedgerEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def usage(self, key: str, window: BudgetWindow) -> UsageReport:
        with self._lock:
            matching = [
                e for e in self._entries
                if e.session_id == key or e.user_id == key or e.agent_name == key
            ]
            total_tokens = sum(e.total_tokens for e in matching)
            total_cost = sum(e.cost_usd for e in matching)
        return UsageReport(
            key=key,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            window=window,
        )

    def flush(self) -> None:
        pass
