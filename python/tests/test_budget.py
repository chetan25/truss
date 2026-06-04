import pytest
import threading
from uuid import uuid4
from truss.budget.config import BudgetConfig, BudgetWindow, AlertConfig, BudgetLimit, ExceededAction
from truss.budget.ledger import LedgerEntry, UsageReport
from truss.budget.memory_store import InMemoryStore


def make_entry(session_id: str, input_tokens: int = 100, cost_usd: float = 0.01) -> LedgerEntry:
    return LedgerEntry(
        session_id=session_id,
        model="test-model",
        input_tokens=input_tokens,
        output_tokens=0,
        cost_usd=cost_usd,
    )


def test_budget_config_defaults():
    cfg = BudgetConfig()
    assert cfg.alert_at_pct == 0.8
    assert cfg.alerts.log_to_stderr is True


def test_budget_limit_serialises():
    limit = BudgetLimit(tokens=10_000, usd=1.0, window=BudgetWindow.PER_SESSION)
    assert limit.window == BudgetWindow.PER_SESSION


def test_in_memory_store_records_and_queries():
    store = InMemoryStore()
    store.record(make_entry("sess-1", 1000, 0.50))
    store.record(make_entry("sess-1", 500, 0.25))
    store.record(make_entry("sess-2", 999, 9.99))
    report = store.usage("sess-1", BudgetWindow.PER_SESSION)
    assert report.total_tokens == 1500
    assert abs(report.total_cost_usd - 0.75) < 0.001


def test_in_memory_store_unknown_key_returns_zero():
    store = InMemoryStore()
    report = store.usage("nobody", BudgetWindow.PER_SESSION)
    assert report.total_tokens == 0
    assert report.total_cost_usd == 0.0


def test_in_memory_store_concurrent_writes():
    store = InMemoryStore()
    errors = []

    def writer(i: int):
        try:
            store.record(make_entry(f"sess-{i}", 100, 0.01))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    total = sum(store.usage(f"sess-{i}", BudgetWindow.PER_SESSION).total_tokens for i in range(20))
    assert total == 2000


def test_ledger_entry_total_tokens():
    entry = LedgerEntry(session_id="s", model="m", input_tokens=100, output_tokens=50, cost_usd=0.01)
    assert entry.total_tokens == 150
