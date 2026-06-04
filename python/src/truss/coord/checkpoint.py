from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from truss.types import AgentEnvelope
from truss.errors import CheckpointNotFound


@dataclass
class CheckpointMeta:
    id: UUID
    session_id: str
    agent_name: str
    description: str
    created_at: int


@dataclass
class Checkpoint:
    session_id: str
    agent_name: str
    envelope_snapshot: AgentEnvelope
    description: str = ""
    id: UUID = field(default_factory=uuid4)
    external_state: dict[str, bytes] = field(default_factory=dict)
    created_at: int = 0

    def meta(self) -> CheckpointMeta:
        return CheckpointMeta(
            id=self.id,
            session_id=self.session_id,
            agent_name=self.agent_name,
            description=self.description,
            created_at=self.created_at,
        )


@runtime_checkable
class CheckpointStore(Protocol):
    def save(self, cp: Checkpoint) -> UUID: ...
    def load(self, id: UUID) -> Checkpoint: ...
    def rollback(self, id: UUID) -> AgentEnvelope: ...
    def list(self, session_id: str) -> list[CheckpointMeta]: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._store: dict[UUID, Checkpoint] = {}
        self._lock = threading.RLock()

    def save(self, cp: Checkpoint) -> UUID:
        with self._lock:
            self._store[cp.id] = cp
        return cp.id

    def load(self, id: UUID) -> Checkpoint:
        with self._lock:
            cp = self._store.get(id)
        if cp is None:
            raise CheckpointNotFound(str(id))
        return cp

    def rollback(self, id: UUID) -> AgentEnvelope:
        return self.load(id).envelope_snapshot

    def list(self, session_id: str) -> list[CheckpointMeta]:
        with self._lock:
            metas = [cp.meta() for cp in self._store.values() if cp.session_id == session_id]
        return sorted(metas, key=lambda m: m.created_at)
