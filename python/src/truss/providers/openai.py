from __future__ import annotations

import os
import time
from typing import Any, Iterator, Optional

from truss.providers.base import LLMMessage, LLMResponse, LLMUsage, StreamChunk, compute_cost


class OpenAIProvider:
    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Any = None,
        circuit_breaker: Any = None,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "openai package required: pip install truss-ai[openai]"
            ) from None

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set and api_key not provided")

        self._client = _openai.OpenAI(api_key=key)
        self._async_client = _openai.AsyncOpenAI(api_key=key)
        self._session = session
        self._circuit_breaker = circuit_breaker
        self._default_model = default_model

    def complete(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        **opts: Any,
    ) -> LLMResponse:
        from truss.errors import BudgetExceeded

        model_id = model or self._default_model

        if self._circuit_breaker:
            trip = self._circuit_breaker.check_and_record("", 0.0, int(time.time() * 1000))
            if trip is not None:
                raise BudgetExceeded(f"Circuit breaker tripped: {trip.value}")

        response = self._client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = compute_cost(model_id, input_tokens, output_tokens)

        if self._session is not None:
            self._session.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                model=model_id,
            )

        text = response.choices[0].message.content if response.choices else ""
        return LLMResponse(
            text=text,
            model=model_id,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost),
            raw=response,
        )

    def stream(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        **opts: Any,
    ) -> Iterator[StreamChunk]:
        from truss.errors import BudgetExceeded

        model_id = model or self._default_model

        if self._circuit_breaker:
            trip = self._circuit_breaker.check_and_record("", 0.0, int(time.time() * 1000))
            if trip is not None:
                raise BudgetExceeded(f"Circuit breaker tripped: {trip.value}")

        response_iter = self._client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            stream_options={"include_usage": True},
        )

        input_tokens = 0
        output_tokens = 0

        for chunk in response_iter:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(text=chunk.choices[0].delta.content, is_final=False)
            if chunk.usage is not None:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        cost = compute_cost(model_id, input_tokens, output_tokens)
        usage = LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)

        if self._session is not None and (input_tokens or output_tokens):
            self._session.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                model=model_id,
            )
        yield StreamChunk(text="", is_final=True, usage=usage)

    async def async_complete(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        **opts: Any,
    ) -> LLMResponse:
        from truss.errors import BudgetExceeded

        model_id = model or self._default_model

        if self._circuit_breaker:
            trip = self._circuit_breaker.check_and_record("", 0.0, int(time.time() * 1000))
            if trip is not None:
                raise BudgetExceeded(f"Circuit breaker tripped: {trip.value}")

        response = await self._async_client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = compute_cost(model_id, input_tokens, output_tokens)

        if self._session is not None:
            self._session.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                model=model_id,
            )

        text = response.choices[0].message.content if response.choices else ""
        return LLMResponse(
            text=text,
            model=model_id,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost),
            raw=response,
        )
