from __future__ import annotations

import sqlite3
import threading
from uuid import UUID, uuid4

from truss.types import AgentEnvelope
from truss.coord.checkpoint import Checkpoint, CheckpointMeta
from truss.errors import CheckpointNotFound


class SqliteCheckpointStore:
    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id            TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                agent_name    TEXT NOT NULL,
                description   TEXT NOT NULL,
                envelope_json TEXT NOT NULL,
                created_at    INTEGER NOT NULL
            )
        """)
        self._conn.commit()

    def save(self, cp: Checkpoint) -> UUID:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?)",
                (
                    str(cp.id), cp.session_id, cp.agent_name, cp.description,
                    cp.envelope_snapshot.model_dump_json(), cp.created_at,
                ),
            )
            self._conn.commit()
        return cp.id

    def load(self, id: UUID) -> Checkpoint:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, session_id, agent_name, description, envelope_json, created_at FROM checkpoints WHERE id = ?",
                (str(id),),
            ).fetchone()
        if row is None:
            raise CheckpointNotFound(str(id))
        cp_id, session_id, agent_name, description, env_json, created_at = row
        return Checkpoint(
            id=UUID(cp_id),
            session_id=session_id,
            agent_name=agent_name,
            description=description,
            envelope_snapshot=AgentEnvelope.model_validate_json(env_json),
            created_at=created_at,
        )

    def rollback(self, id: UUID) -> AgentEnvelope:
        return self.load(id).envelope_snapshot

    def list(self, session_id: str) -> list[CheckpointMeta]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, agent_name, description, created_at FROM checkpoints WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [
            CheckpointMeta(id=UUID(r[0]), session_id=r[1], agent_name=r[2], description=r[3], created_at=r[4])
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
