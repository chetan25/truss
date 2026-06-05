from __future__ import annotations

import time
from typing import Any, Optional

from truss.providers.base import LLMMessage, LLMResponse, LLMUsage, compute_cost


class OllamaProvider:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        session: Any = None,
        circuit_breaker: Any = None,
        default_model: str = "llama3",
    ) -> None:
        try:
            import httpx as _httpx
        except ImportError:
            raise ImportError(
                "httpx required: pip install truss-ai[ollama]"
            ) from None

        self._client = _httpx.Client(base_url=base_url, timeout=120.0)
        self._session = session
        self._circuit_breaker = circuit_breaker
        self._default_model = default_model

    def complete(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        **opts: Any,
    ) -> LLMResponse:
        from truss.errors import BudgetExceeded

        model_id = model or self._default_model

        if self._circuit_breaker:
            prompt = messages[0].content if messages else ""
            trip = self._circuit_breaker.check_and_record(prompt, 0.0, int(time.time() * 1000))
            if trip is not None:
                raise BudgetExceeded(f"Circuit breaker tripped: {trip.value}")

        response = self._client.post(
            "/api/chat",
            json={
                "model": model_id,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

        text = data.get("message", {}).get("content", "")
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        cost = compute_cost(model_id, input_tokens, output_tokens)  # 0.0 for local models

        if self._session is not None:
            self._session.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                model=model_id,
            )

        return LLMResponse(
            text=text,
            model=model_id,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost),
            raw=data,
        )
