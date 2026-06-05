from __future__ import annotations

import os
import time
from typing import Any, Optional

from truss.providers.base import LLMMessage, LLMResponse, LLMUsage, compute_cost


class GoogleProvider:
    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Any = None,
        circuit_breaker: Any = None,
        default_model: str = "gemini-1.5-flash",
    ) -> None:
        try:
            import google.generativeai as _genai
        except ImportError:
            raise ImportError(
                "google-generativeai required: pip install truss-ai[google]"
            ) from None

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set and api_key not provided")

        _genai.configure(api_key=key)
        self._genai = _genai
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

        # Separate system message — Gemini prepends it to the first user turn
        system_content = ""
        conversation: list[LLMMessage] = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                conversation.append(msg)

        if not conversation:
            raise ValueError("At least one user or assistant message is required")

        # Build Gemini history (all turns except the last user message)
        history = []
        for msg in conversation[:-1]:
            gemini_role = "model" if msg.role == "assistant" else "user"
            history.append({"role": gemini_role, "parts": [msg.content]})

        last = conversation[-1]
        user_text = f"{system_content}\n\n{last.content}".strip() if system_content else last.content

        genai_model = self._genai.GenerativeModel(model_id)
        chat = genai_model.start_chat(history=history)
        result = chat.send_message(user_text)
        response = result.response

        input_tokens = getattr(getattr(response, "usage_metadata", None), "prompt_token_count", 0) or 0
        output_tokens = getattr(getattr(response, "usage_metadata", None), "candidates_token_count", 0) or 0
        cost = compute_cost(model_id, input_tokens, output_tokens)

        if self._session is not None:
            self._session.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                model=model_id,
            )

        return LLMResponse(
            text=response.text,
            model=model_id,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost),
            raw=response,
        )
