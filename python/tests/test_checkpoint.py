import pytest
import os
import tempfile
from uuid import uuid4
from truss.types import AgentEnvelope
from truss.coord.checkpoint import Checkpoint, CheckpointMeta, InMemoryCheckpointStore
from truss.coord.sqlite_checkpoint import SqliteCheckpointStore
from truss.errors import CheckpointNotFound


def make_checkpoint(session_id: str, description: str) -> Checkpoint:
    return Checkpoint(
        session_id=session_id,
        agent_name="test-agent",
        envelope_snapshot=AgentEnvelope(task="test task"),
        description=description,
    )


def test_save_and_load_checkpoint():
    store = InMemoryCheckpointStore()
    cp = make_checkpoint("sess-1", "after planning")
    cp_id = store.save(cp)
    loaded = store.load(cp_id)
    assert loaded.description == "after planning"


def test_rollback_returns_envelope():
    store = InMemoryCheckpointStore()
    cp = make_checkpoint("sess-1", "step 1")
    cp.envelope_snapshot.task = "original task"
    cp_id = store.save(cp)
    env = store.rollback(cp_id)
    assert env.task == "original task"


def test_load_nonexistent_raises():
    store = InMemoryCheckpointStore()
    with pytest.raises(CheckpointNotFound):
        store.load(uuid4())


def test_list_filters_by_session():
    store = InMemoryCheckpointStore()
    store.save(make_checkpoint("sess-a", "cp1"))
    store.save(make_checkpoint("sess-b", "cp2"))
    metas = store.list("sess-a")
    assert len(metas) == 1
    assert metas[0].description == "cp1"


def test_sqlite_checkpoint_save_and_load():
    store = SqliteCheckpointStore(":memory:")
    cp = make_checkpoint("sess-1", "sqlite test")
    cp_id = store.save(cp)
    loaded = store.load(cp_id)
    assert loaded.description == "sqlite test"
    assert loaded.envelope_snapshot.task == "test task"


def test_sqlite_checkpoint_survives_reopen():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        store1 = SqliteCheckpointStore(path)
        cp = make_checkpoint("sess-p", "persistent")
        cp_id = store1.save(cp)
        store1.close()

        store2 = SqliteCheckpointStore(path)
        loaded = store2.load(cp_id)
        assert loaded.description == "persistent"
        store2.close()
    finally:
        os.unlink(path)
