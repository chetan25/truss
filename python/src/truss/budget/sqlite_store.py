from __future__ import annotations

import json
import sqlite3
import threading
from truss.budget.config import BudgetWindow
from truss.budget.ledger import LedgerEntry, UsageReport


class SqliteLedgerStore:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ledger (
                id            TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                user_id       TEXT,
                agent_name    TEXT,
                model         TEXT NOT NULL,
                input_tokens  INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd      REAL NOT NULL,
                timestamp     INTEGER NOT NULL,
                tags          TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def record(self, entry: LedgerEntry) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    str(entry.id), entry.session_id, entry.user_id, entry.agent_name,
                    entry.model, entry.input_tokens, entry.output_tokens,
                    entry.cost_usd, entry.timestamp, json.dumps(entry.tags),
                ),
            )
            self._conn.commit()

    def usage(self, key: str, window: BudgetWindow) -> UsageReport:
        with self._lock:
            row = self._conn.execute(
                """SELECT COALESCE(SUM(input_tokens + output_tokens), 0),
                          COALESCE(SUM(cost_usd), 0.0)
                   FROM ledger
                   WHERE session_id = ? OR user_id = ? OR agent_name = ?""",
                (key, key, key),
            ).fetchone()
        total_tokens, total_cost = row
        return UsageReport(
            key=key,
            total_tokens=int(total_tokens),
            total_cost_usd=float(total_cost),
            window=window,
        )

    def flush(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
