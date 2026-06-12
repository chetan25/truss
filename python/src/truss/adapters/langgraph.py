from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable


class TrussNode:
    """Wraps a LangGraph node callable with Truss session checkpointing.

    Checkpoints the session before each node call so any node boundary can be
    rolled back on failure.  If the returned state contains a
    ``__truss_usage__`` key (a dict with ``input_tokens``, ``output_tokens``,
    and optional ``model``/``cost_usd``), that usage is recorded to the session
    ledger automatically.

    Works with both sync and async node functions; LangGraph itself is not
    imported — any callable that matches the ``state -> state`` contract works.

    Usage::

        @TrussNode(session)
        def plan_node(state: dict) -> dict:
            ...

        graph.add_node("plan", TrussNode(session, name="plan")(plan_fn))
    """

    def __init__(self, session: Any, name: str = "") -> None:
        self._session = session
        self._name = name

    def __call__(self, fn: Callable) -> Callable:
        node_name = self._name or fn.__name__
        session = self._session

        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(state: Any) -> Any:
                try:
                    session.checkpoint(f"before-{node_name}")
                except Exception:
                    pass
                result = await fn(state)
                _record_usage(session, result)
                return result
            return async_wrapper

        @functools.wraps(fn)
        def wrapper(state: Any) -> Any:
            try:
                session.checkpoint(f"before-{node_name}")
            except Exception:
                pass
            result = fn(state)
            _record_usage(session, result)
            return result

        return wrapper


def _record_usage(session: Any, result: Any) -> None:
    if not isinstance(result, dict):
        return
    usage = result.get("__truss_usage__")
    if not usage:
        return
    try:
        session.record_usage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cost_usd=usage.get("cost_usd", 0.0),
            model=usage.get("model", "unknown"),
        )
    except Exception:
        pass
