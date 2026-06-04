from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from truss.budget.config import BudgetWindow
from truss.budget.ledger import LedgerEntry, LedgerStore
from truss.budget.memory_store import InMemoryStore
from truss.context.surgeon import SurgeonConfig, SurgeonResult, compress
from truss.coord.checkpoint import CheckpointStore, InMemoryCheckpointStore, Checkpoint
from truss.fence.memory_fence import FenceStore, InMemoryFence
from truss.types import AgentEnvelope, ContextBlock


@dataclass
class SessionReport:
    session_id: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    cost_saved_usd: float
    budget_used_usd: float
    budget_limit_usd: Optional[float]
    checkpoint_count: int

    def __str__(self) -> str:
        lines = [
            f"Session: {self.session_id[:8]}",
            f"Context: {self.tokens_before:,} -> {self.tokens_after:,} tokens (saved {self.tokens_saved:,})",
            f"Est. savings: ${self.cost_saved_usd:.4f}",
        ]
        if self.budget_limit_usd is not None:
            pct = (self.budget_used_usd / self.budget_limit_usd * 100) if self.budget_limit_usd else 0
            lines.append(f"Budget: ${self.budget_used_usd:.4f} of ${self.budget_limit_usd:.2f} used ({pct:.1f}%)")
        if self.checkpoint_count:
            lines.append(f"Checkpoints: {self.checkpoint_count}")
        return "\n".join(lines)


class Session:
    _COST_PER_1K_TOKENS = 0.001

    def __init__(
        self,
        envelope: Optional[AgentEnvelope] = None,
        budget_usd: Optional[float] = None,
        target_tokens: int = 8_000,
        preserve_recent: int = 5,
        ledger: Optional[LedgerStore] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        fence: Optional[FenceStore] = None,
    ) -> None:
        self._envelope = envelope or AgentEnvelope(task="session")
        self._budget_usd = budget_usd
        self._tokens_before = 0
        self._tokens_after = 0
        self._ledger: LedgerStore = ledger or InMemoryStore()
        self._checkpoint_store: CheckpointStore = checkpoint_store or InMemoryCheckpointStore()
        self._fence: FenceStore = fence or InMemoryFence()
        self._surgeon_config = SurgeonConfig(target_tokens=target_tokens, preserve_recent=preserve_recent)
        self._checkpoint_count = 0

    @property
    def session_id(self) -> str:
        return str(self._envelope.id)

    def compress(self, blocks: list[ContextBlock]) -> SurgeonResult:
        result = compress(blocks, self._surgeon_config)
        self._tokens_before += result.tokens_before
        self._tokens_after += result.tokens_after
        return result

    def record_usage(self, input_tokens: int, output_tokens: int, cost_usd: float = 0.0, model: str = "unknown") -> None:
        if cost_usd == 0.0:
            cost_usd = (input_tokens + output_tokens) / 1000 * self._COST_PER_1K_TOKENS
        self._ledger.record(LedgerEntry(
            session_id=self.session_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ))

    def checkpoint(self, description: str = "") -> UUID:
        cp = Checkpoint(
            session_id=self.session_id,
            agent_name="session",
            envelope_snapshot=self._envelope.model_copy(deep=True),
            description=description,
        )
        self._checkpoint_count += 1
        return self._checkpoint_store.save(cp)

    def rollback(self, checkpoint_id: UUID) -> None:
        self._envelope = self._checkpoint_store.rollback(checkpoint_id)

    def report(self) -> SessionReport:
        tokens_saved = max(0, self._tokens_before - self._tokens_after)
        cost_saved = tokens_saved / 1000 * self._COST_PER_1K_TOKENS
        usage = self._ledger.usage(self.session_id, BudgetWindow.PER_SESSION)
        return SessionReport(
            session_id=self.session_id,
            tokens_before=self._tokens_before,
            tokens_after=self._tokens_after,
            tokens_saved=tokens_saved,
            cost_saved_usd=cost_saved,
            budget_used_usd=usage.total_cost_usd,
            budget_limit_usd=self._budget_usd,
            checkpoint_count=self._checkpoint_count,
        )

    async def __aenter__(self) -> "Session":
        return self

    async def __aexit__(self, *_) -> None:
        pass
