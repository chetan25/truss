from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from truss.errors import FenceLockConflict


@dataclass
class LockHandle:
    key: str
    owner: str
    acquired_at_ms: int
    ttl_ms: int

    def is_expired(self, now_ms: int) -> bool:
        return now_ms > self.acquired_at_ms + self.ttl_ms


@runtime_checkable
class FenceStore(Protocol):
    def acquire(self, key: str, owner: str, ttl_ms: int, now_ms: int) -> None: ...
    def release(self, key: str, owner: str) -> None: ...
    def is_locked(self, key: str, now_ms: int) -> bool: ...


class InMemoryFence:
    def __init__(self) -> None:
        self._locks: dict[str, LockHandle] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str, owner: str, ttl_ms: int, now_ms: int) -> None:
        with self._lock:
            handle = self._locks.get(key)
            if handle is not None and not handle.is_expired(now_ms) and handle.owner != owner:
                raise FenceLockConflict(f"{key} held by {handle.owner}")
            self._locks[key] = LockHandle(key=key, owner=owner, acquired_at_ms=now_ms, ttl_ms=ttl_ms)

    def release(self, key: str, owner: str) -> None:
        with self._lock:
            handle = self._locks.get(key)
            if handle is not None and handle.owner == owner:
                del self._locks[key]

    def is_locked(self, key: str, now_ms: int) -> bool:
        with self._lock:
            handle = self._locks.get(key)
            return handle is not None and not handle.is_expired(now_ms)
