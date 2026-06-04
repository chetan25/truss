from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CircuitTrip(Enum):
    RATE_LIMIT = "rate_limit"
    COST_VELOCITY = "cost_velocity"
    MAX_RETRY_DEPTH = "max_retry_depth"
    REPEATED_PROMPT = "repeated_prompt"


@dataclass
class CircuitBreakerConfig:
    max_requests_per_minute: int = 60
    max_cost_velocity_usd: float = 1.0
    max_retry_depth: int = 3
    trip_on_repeated_prompt: bool = True


@dataclass
class _Record:
    timestamp_ms: int
    cost_usd: float
    prompt_hash: int


def _fnv1a(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode():
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._window: deque[_Record] = deque()
        self._retry_depth = 0
        self._lock = threading.Lock()

    def check_and_record(self, prompt: str, cost_usd: float, now_ms: int) -> Optional[CircuitTrip]:
        prompt_hash = _fnv1a(prompt)
        with self._lock:
            cutoff = now_ms - 60_000
            while self._window and self._window[0].timestamp_ms < cutoff:
                self._window.popleft()

            if len(self._window) >= self._config.max_requests_per_minute:
                return CircuitTrip.RATE_LIMIT

            running_cost = sum(r.cost_usd for r in self._window)
            if running_cost + cost_usd > self._config.max_cost_velocity_usd:
                return CircuitTrip.COST_VELOCITY

            if self._config.trip_on_repeated_prompt:
                recent = list(self._window)[-3:]
                if any(r.prompt_hash == prompt_hash for r in recent):
                    return CircuitTrip.REPEATED_PROMPT

            self._window.append(_Record(now_ms, cost_usd, prompt_hash))
            return None

    def increment_retry(self) -> Optional[CircuitTrip]:
        with self._lock:
            self._retry_depth += 1
            if self._retry_depth > self._config.max_retry_depth:
                return CircuitTrip.MAX_RETRY_DEPTH
            return None

    def reset_retry(self) -> None:
        with self._lock:
            self._retry_depth = 0
