from __future__ import annotations

from typing import Any

from truss.context.surgeon import CompressionStrategy, SurgeonConfig, compress
from truss.types import ContextBlock, ContextRole, ContextWeight


class TrussMemory:
    """Drop-in replacement for LangChain ConversationBufferMemory.

    Compresses conversation history using Truss context surgeon.
    No dependency on langchain-core at import time.
    """

    def __init__(
        self,
        target_tokens: int = 8_000,
        preserve_recent: int = 5,
        strategy: str = "hybrid",
        memory_key: str = "history",
    ) -> None:
        self.memory_key = memory_key
        self.blocks: list[ContextBlock] = []
        self._config = SurgeonConfig(
            target_tokens=target_tokens,
            preserve_recent=preserve_recent,
            strategy=CompressionStrategy(strategy),
        )

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        human = inputs.get("input", "")
        ai = outputs.get("output", "")
        if human:
            self.blocks.append(ContextBlock(
                role=ContextRole.TASK, weight=ContextWeight.NORMAL,
                content=f"Human: {human}", source="user",
            ))
        if ai:
            self.blocks.append(ContextBlock(
                role=ContextRole.FINDING, weight=ContextWeight.NORMAL,
                content=f"AI: {ai}", source="assistant",
            ))

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        if not self.blocks:
            return {self.memory_key: ""}
        result = compress(self.blocks, self._config)
        self.blocks = result.blocks
        return {self.memory_key: "\n".join(b.content for b in self.blocks)}

    def clear(self) -> None:
        self.blocks = []


try:
    from langchain_core.callbacks.base import BaseCallbackHandler  # noqa: F401
    from langchain_core.outputs import LLMResult  # noqa: F401
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


def _require_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError("langchain-core required: pip install truss-ai[langchain]")


class TrussCallbackHandler:
    """LangChain callback that auto-records token usage into a Truss Session."""

    def __init__(self, session: Any) -> None:
        _require_langchain()
        self._session = session

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        llm_output = (response.llm_output or {}) if hasattr(response, "llm_output") else {}

        usage = llm_output.get("usage") or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        if not input_tokens and not output_tokens:
            token_usage = llm_output.get("token_usage") or {}
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)

        model = (
            llm_output.get("model")
            or llm_output.get("model_name")
            or "unknown"
        )

        if input_tokens or output_tokens:
            self._session.record_usage(
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                model=model,
            )

    def on_llm_error(self, error: Any, **kwargs: Any) -> None:
        pass


class TrussLLM:
    """LangChain-compatible LLM that routes calls through a Truss LLMProvider."""

    def __init__(self, provider: Any, default_model: str = "claude-haiku-4-5") -> None:
        _require_langchain()
        self._provider = provider
        self.default_model = default_model

    @property
    def _llm_type(self) -> str:
        return "truss"

    def _call(self, prompt: str, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> str:
        from truss.providers.base import LLMMessage
        messages = [LLMMessage(role="user", content=prompt)]
        response = self._provider.complete(messages=messages, model=self.default_model)
        return response.text

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        return self._call(prompt, **kwargs)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self._call(prompt, **kwargs)
