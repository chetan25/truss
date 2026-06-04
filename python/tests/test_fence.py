import pytest
from truss.fence.memory_fence import InMemoryFence, LockHandle
from truss.errors import FenceLockConflict


def test_acquire_grants_lock_to_first_owner():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=0)


def test_acquire_blocks_second_owner():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=0)
    with pytest.raises(FenceLockConflict):
        fence.acquire("doc-1", "agent-b", ttl_ms=30_000, now_ms=1_000)


def test_expired_lock_can_be_reacquired():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=5_000, now_ms=0)
    fence.acquire("doc-1", "agent-b", ttl_ms=5_000, now_ms=30_000)


def test_release_frees_lock():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=0)
    fence.release("doc-1", "agent-a")
    fence.acquire("doc-1", "agent-b", ttl_ms=30_000, now_ms=1_000)


def test_release_by_wrong_owner_is_noop():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=0)
    fence.release("doc-1", "agent-b")
    assert fence.is_locked("doc-1", now_ms=1_000) is True


def test_is_locked_returns_false_after_expiry():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=5_000, now_ms=0)
    assert fence.is_locked("doc-1", now_ms=6_000) is False


def test_owner_can_refresh_own_lock():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=0)
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=1_000)
