# Truss Phase 3 Python — Streaming, Async, Full Google + Ollama Providers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sync streaming, async completion, and full Google/Ollama provider implementations to the existing `truss-ai` Python package.

**Architecture:** Streaming uses a sync generator returning `Iterator[StreamChunk]` — no new event loop required. Async uses `AsyncAnthropic`/`AsyncOpenAI` clients created eagerly in `__init__`. `GoogleProvider` wraps `google-generativeai`, normalising Gemini's "model" role to the standard "assistant" role. `OllamaProvider` wraps `httpx` and calls the local Ollama REST API; cost is always `0.0` for local models.

**Tech Stack:** Python 3.10+ · Pydantic v2 · anthropic>=0.40 · openai>=1.0 · google-generativeai>=0.8 · httpx>=0.27 · pytest 8 · pytest-asyncio

---

## File Structure

```
python/src/truss/
├── providers/
│   ├── base.py        # add StreamChunk dataclass + LLMStreamProvider Protocol + Gemini/Ollama COST_TABLE entries
│   ├── anthropic.py   # add stream() generator + async_complete() + _async_client
│   └── openai.py      # add stream() generator + async_complete() + _async_client
│   ├── google.py      # REPLACE stub with full implementation
│   └── ollama.py      # REPLACE stub with full implementation

python/tests/
└── test_providers_p3.py   # all Phase 3 tests (streaming, async, Google, Ollama)
```

---

## Task 1: StreamChunk type + COST_TABLE additions

**Files:**
- Modify: `python/src/truss/providers/base.py`
- Create: `python/tests/test_providers_p3.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_providers_p3.py`:

```python
import pytest
from truss.providers.base import (
    LLMMessage, LLMUsage, StreamChunk, LLMStreamProvider,
    COST_TABLE, compute_cost,
)


def test_stream_chunk_non_final_has_no_usage():
    chunk = StreamChunk(text="hello", is_final=False)
    assert chunk.usage is None


def test_stream_chunk_final_carries_usage():
    usage = LLMUsage(input_tokens=10, output_tokens=5, cost_usd=0.001)
    chunk = StreamChunk(text="", is_final=True, usage=usage)
    assert chunk.usage.input_tokens == 10


def test_cost_table_has_gemini_models():
    assert "gemini-1.5-flash" in COST_TABLE
    assert "gemini-1.5-pro" in COST_TABLE


def test_ollama_cost_is_zero():
    cost = compute_cost("llama3", 1000, 1000)
    assert cost == 0.0


def test_llm_stream_provider_protocol_satisfied():
    from typing import Iterator

    class FakeStreamProvider:
        def complete(self, messages, model, **opts): ...
        def stream(self, messages, model, **opts) -> Iterator[StreamChunk]: ...

    assert isinstance(FakeStreamProvider(), LLMStreamProvider)
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd python && python -m pytest tests/test_providers_p3.py -v
```

Expected: `ImportError: cannot import name 'StreamChunk'`

- [ ] **Step 3: Update base.py**

In `python/src/truss/providers/base.py`, add after the `LLMResponse` dataclass and update `COST_TABLE`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol, runtime_checkable


@dataclass
class LLMMessage:
    role: str   # "user" | "assistant" | "system"
    content: str


@dataclass
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: LLMUsage
    raw: Any = None


@dataclass
class StreamChunk:
    text: str
    is_final: bool
    usage: Optional[LLMUsage] = None


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        **opts: Any,
    ) -> LLMResponse: ...


@runtime_checkable
class LLMStreamProvider(LLMProvider, Protocol):
    def stream(
        self,
        messages: list[LLMMessage],
        model: str,
        **opts: Any,
    ) -> Iterator[StreamChunk]: ...


COST_TABLE: dict[str, tuple[float, float]] = {
    # (input_$/1k, output_$/1k)
    "claude-haiku-4-5":   (0.001,     0.005),
    "claude-sonnet-4-6":  (0.003,     0.015),
    "claude-opus-4-8":    (0.015,     0.075),
    "gpt-4o-mini":        (0.00015,   0.0006),
    "gpt-4o":             (0.005,     0.015),
    "gpt-4-turbo":        (0.010,     0.030),
    "o1":                 (0.015,     0.060),
    "o1-mini":            (0.003,     0.012),
    # Gemini models
    "gemini-1.5-flash":   (0.000075,  0.0003),
    "gemini-1.5-pro":     (0.00125,   0.005),
    "gemini-2.0-flash":   (0.0001,    0.0004),
    # Ollama (local — always free)
    "llama3":             (0.0,       0.0),
    "llama3.1":           (0.0,       0.0),
    "mistral":            (0.0,       0.0),
    "qwen2":              (0.0,       0.0),
}

_DEFAULT_RATES = (0.001, 0.005)


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_TABLE.get(model, _DEFAULT_RATES)
    return (input_tokens / 1000) * rates[0] + (output_tokens / 1000) * rates[1]
```

Also update `python/src/truss/providers/__init__.py` to export the new symbols:

```python
from truss.providers.base import (
    LLMMessage, LLMUsage, LLMResponse, LLMProvider,
    StreamChunk, LLMStreamProvider,
    COST_TABLE, compute_cost,
)
from truss.providers.anthropic import AnthropicProvider
from truss.providers.openai import OpenAIProvider
from truss.providers.google import GoogleProvider
from truss.providers.ollama import OllamaProvider

__all__ = [
    "LLMMessage", "LLMUsage", "LLMResponse", "LLMProvider",
    "StreamChunk", "LLMStreamProvider",
    "COST_TABLE", "compute_cost",
    "AnthropicProvider", "OpenAIProvider", "GoogleProvider", "OllamaProvider",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```
cd python && python -m pytest tests/test_providers_p3.py::test_stream_chunk_non_final_has_no_usage tests/test_providers_p3.py::test_stream_chunk_final_carries_usage tests/test_providers_p3.py::test_cost_table_has_gemini_models tests/test_providers_p3.py::test_ollama_cost_is_zero tests/test_providers_p3.py::test_llm_stream_provider_protocol_satisfied -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/providers/base.py python/src/truss/providers/__init__.py python/tests/test_providers_p3.py
git commit -m "feat: add StreamChunk, LLMStreamProvider, Gemini/Ollama cost table entries"
```

---

## Task 2: AnthropicProvider — stream()

**Files:**
- Modify: `python/src/truss/providers/anthropic.py`
- Modify: `python/tests/test_providers_p3.py`

- [ ] **Step 1: Append streaming tests**

Append to `python/tests/test_providers_p3.py`:

```python
from unittest.mock import MagicMock


def _make_anthropic_stream_mock(chunks=("Hello", " world"), input_tokens=10, output_tokens=5):
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cm)
    mock_cm.__exit__ = MagicMock(return_value=False)
    mock_cm.text_stream = iter(chunks)
    final_msg = MagicMock()
    final_msg.usage.input_tokens = input_tokens
    final_msg.usage.output_tokens = output_tokens
    mock_cm.get_final_message.return_value = final_msg
    return mock_cm


def test_anthropic_stream_yields_text_chunks():
    from truss.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.messages.stream.return_value = _make_anthropic_stream_mock(("Hi", "!"))

    chunks = list(provider.stream(
        [LLMMessage(role="user", content="hello")],
        model="claude-haiku-4-5",
    ))
    non_final = [c for c in chunks if not c.is_final]
    assert [c.text for c in non_final] == ["Hi", "!"]


def test_anthropic_stream_final_chunk_has_usage():
    from truss.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.messages.stream.return_value = _make_anthropic_stream_mock(
        input_tokens=100, output_tokens=50
    )

    chunks = list(provider.stream(
        [LLMMessage(role="user", content="hi")],
        model="claude-haiku-4-5",
    ))
    final = next(c for c in chunks if c.is_final)
    assert final.usage.input_tokens == 100
    assert final.usage.output_tokens == 50
    assert final.usage.cost_usd > 0


def test_anthropic_stream_records_usage_to_session():
    from truss.providers.anthropic import AnthropicProvider
    from truss.session import Session

    session = Session()
    provider = AnthropicProvider(api_key="test-key", session=session)
    provider._client = MagicMock()
    provider._client.messages.stream.return_value = _make_anthropic_stream_mock(
        input_tokens=200, output_tokens=100
    )

    list(provider.stream([LLMMessage(role="user", content="hi")], model="claude-haiku-4-5"))
    assert session.report().budget_used_usd > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd python && python -m pytest tests/test_providers_p3.py -k "anthropic_stream" -v
```

Expected: `AttributeError: 'AnthropicProvider' object has no attribute 'stream'`

- [ ] **Step 3: Add stream() to anthropic.py**

Add the following method to `AnthropicProvider` in `python/src/truss/providers/anthropic.py`, after `complete()`:

```python
    def stream(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        **opts: Any,
    ) -> "Iterator[StreamChunk]":
        from typing import Iterator
        from truss.errors import BudgetExceeded
        from truss.providers.base import StreamChunk

        model_id = model or self._default_model

        if self._circuit_breaker:
            prompt = messages[0].content if messages else ""
            trip = self._circuit_breaker.check_and_record(prompt, 0.0, int(time.time() * 1000))
            if trip is not None:
                raise BudgetExceeded(f"Circuit breaker tripped: {trip.value}")

        with self._client.messages.stream(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        ) as stream:
            for text in stream.text_stream:
                yield StreamChunk(text=text, is_final=False)

            final = stream.get_final_message()
            input_tokens = final.usage.input_tokens
            output_tokens = final.usage.output_tokens
            cost = compute_cost(model_id, input_tokens, output_tokens)

        usage = LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)
        if self._session is not None:
            self._session.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                model=model_id,
            )
        yield StreamChunk(text="", is_final=True, usage=usage)
```

Also add `from typing import Iterator` to the top-level imports of `anthropic.py` (it already imports `Optional` and `Any`):

```python
from typing import Any, Iterator, Optional
```

- [ ] **Step 4: Run tests**

```
cd python && python -m pytest tests/test_providers_p3.py -k "anthropic_stream" -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/providers/anthropic.py python/tests/test_providers_p3.py
git commit -m "feat: add AnthropicProvider.stream() sync generator"
```

---

## Task 3: OpenAIProvider — stream()

**Files:**
- Modify: `python/src/truss/providers/openai.py`
- Modify: `python/tests/test_providers_p3.py`

- [ ] **Step 1: Append OpenAI streaming tests**

Append to `python/tests/test_providers_p3.py`:

```python
def _make_openai_chunk(content=None, prompt_tokens=None, completion_tokens=None):
    chunk = MagicMock()
    if content is not None:
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
    else:
        chunk.choices = []
    if prompt_tokens is not None:
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = prompt_tokens
        chunk.usage.completion_tokens = completion_tokens
    else:
        chunk.usage = None
    return chunk


def test_openai_stream_yields_text_chunks():
    from truss.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = iter([
        _make_openai_chunk(content="Hello"),
        _make_openai_chunk(content=" world"),
        _make_openai_chunk(prompt_tokens=10, completion_tokens=5),
    ])

    chunks = list(provider.stream(
        [LLMMessage(role="user", content="hello")],
        model="gpt-4o-mini",
    ))
    non_final = [c for c in chunks if not c.is_final]
    assert [c.text for c in non_final] == ["Hello", " world"]


def test_openai_stream_final_chunk_has_usage():
    from truss.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = iter([
        _make_openai_chunk(content="hi"),
        _make_openai_chunk(prompt_tokens=50, completion_tokens=25),
    ])

    chunks = list(provider.stream(
        [LLMMessage(role="user", content="hi")],
        model="gpt-4o-mini",
    ))
    final = next(c for c in chunks if c.is_final)
    assert final.usage.input_tokens == 50
    assert final.usage.output_tokens == 25
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd python && python -m pytest tests/test_providers_p3.py -k "openai_stream" -v
```

Expected: `AttributeError: 'OpenAIProvider' object has no attribute 'stream'`

- [ ] **Step 3: Add stream() to openai.py**

Add the following method to `OpenAIProvider` in `python/src/truss/providers/openai.py`, after `complete()`. Also add `Iterator` to the imports at the top.

```python
from typing import Any, Iterator, Optional
```

```python
    def stream(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        **opts: Any,
    ) -> "Iterator[StreamChunk]":
        from truss.errors import BudgetExceeded
        from truss.providers.base import StreamChunk

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
```

- [ ] **Step 4: Run tests**

```
cd python && python -m pytest tests/test_providers_p3.py -k "openai_stream" -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/providers/openai.py python/tests/test_providers_p3.py
git commit -m "feat: add OpenAIProvider.stream() sync generator"
```

---

## Task 4: AnthropicProvider + OpenAIProvider — async_complete()

**Files:**
- Modify: `python/src/truss/providers/anthropic.py`
- Modify: `python/src/truss/providers/openai.py`
- Modify: `python/tests/test_providers_p3.py`

- [ ] **Step 1: Append async tests**

Append to `python/tests/test_providers_p3.py`:

```python
from unittest.mock import AsyncMock


async def test_anthropic_async_complete_returns_response():
    from truss.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    provider._async_client = MagicMock()
    provider._async_client.messages.create = AsyncMock(
        return_value=_make_anthropic_response_mock("Async reply", 100, 50)
    )

    result = await provider.async_complete(
        [LLMMessage(role="user", content="hello")],
        model="claude-haiku-4-5",
    )
    assert result.text == "Async reply"
    assert result.usage.input_tokens == 100


async def test_openai_async_complete_returns_response():
    from truss.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")
    provider._async_client = MagicMock()
    provider._async_client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response_mock("Async OpenAI", 80, 40)
    )

    result = await provider.async_complete(
        [LLMMessage(role="user", content="hello")],
        model="gpt-4o-mini",
    )
    assert result.text == "Async OpenAI"
    assert result.usage.input_tokens == 80


async def test_anthropic_async_complete_records_to_session():
    from truss.providers.anthropic import AnthropicProvider
    from truss.session import Session

    session = Session()
    provider = AnthropicProvider(api_key="test-key", session=session)
    provider._async_client = MagicMock()
    provider._async_client.messages.create = AsyncMock(
        return_value=_make_anthropic_response_mock(input_tokens=200, output_tokens=100)
    )

    await provider.async_complete([LLMMessage(role="user", content="hi")], model="claude-haiku-4-5")
    assert session.report().budget_used_usd > 0
```

These tests use two helper functions. Add them just before the async tests (above `test_anthropic_async_complete_returns_response`):

```python
def _make_anthropic_response_mock(text="Hello!", input_tokens=10, output_tokens=5):
    mock_content = MagicMock()
    mock_content.text = text
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    return mock_response


def _make_openai_response_mock(text="Hi!", prompt_tokens=10, completion_tokens=5):
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    return mock_response
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd python && python -m pytest tests/test_providers_p3.py -k "async_complete" -v
```

Expected: `AttributeError: 'AnthropicProvider' object has no attribute 'async_complete'`

- [ ] **Step 3: Add _async_client + async_complete() to anthropic.py**

In `python/src/truss/providers/anthropic.py`, modify `__init__` to also create `_async_client`:

```python
    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Any = None,
        circuit_breaker: Any = None,
        default_model: str = "claude-haiku-4-5",
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required: pip install truss-ai[anthropic]"
            ) from None

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set and api_key not provided")

        self._client = _anthropic.Anthropic(api_key=key)
        self._async_client = _anthropic.AsyncAnthropic(api_key=key)
        self._session = session
        self._circuit_breaker = circuit_breaker
        self._default_model = default_model
```

Add `async_complete()` method after `stream()`:

```python
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
            prompt = messages[0].content if messages else ""
            trip = self._circuit_breaker.check_and_record(prompt, 0.0, int(time.time() * 1000))
            if trip is not None:
                raise BudgetExceeded(f"Circuit breaker tripped: {trip.value}")

        response = await self._async_client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = compute_cost(model_id, input_tokens, output_tokens)

        if self._session is not None:
            self._session.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                model=model_id,
            )

        text = response.content[0].text if response.content else ""
        return LLMResponse(
            text=text,
            model=model_id,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost),
            raw=response,
        )
```

- [ ] **Step 4: Add _async_client + async_complete() to openai.py**

In `python/src/truss/providers/openai.py`, modify `__init__` to also create `_async_client`:

```python
        self._client = _openai.OpenAI(api_key=key)
        self._async_client = _openai.AsyncOpenAI(api_key=key)
        self._session = session
        self._circuit_breaker = circuit_breaker
        self._default_model = default_model
```

Add `async_complete()` method after `stream()`:

```python
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
```

- [ ] **Step 5: Run tests**

```
cd python && python -m pytest tests/test_providers_p3.py -k "async_complete" -v
```

Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add python/src/truss/providers/anthropic.py python/src/truss/providers/openai.py python/tests/test_providers_p3.py
git commit -m "feat: add async_complete() to AnthropicProvider and OpenAIProvider"
```

---

## Task 5: Full GoogleProvider

**Files:**
- Modify: `python/src/truss/providers/google.py` (replace stub)
- Modify: `python/tests/test_providers_p3.py`
- Modify: `python/pyproject.toml`

- [ ] **Step 1: Add google-generativeai to dev deps**

In `python/pyproject.toml`, update optional-dependencies:

```toml
[project.optional-dependencies]
langchain  = ["langchain-core>=0.2"]
anthropic  = ["anthropic>=0.40"]
openai     = ["openai>=1.0"]
crewai     = ["crewai>=0.50"]
google     = ["google-generativeai>=0.8"]
ollama     = ["httpx>=0.27"]
all        = ["anthropic>=0.40", "openai>=1.0", "langchain-core>=0.2", "crewai>=0.50", "google-generativeai>=0.8", "httpx>=0.27"]
dev        = ["pytest>=8.0", "pytest-asyncio>=0.23", "anthropic>=0.40", "openai>=1.0", "langchain-core>=0.2", "google-generativeai>=0.8", "httpx>=0.27"]
```

Install:

```
cd python && pip install -e ".[dev]"
```

- [ ] **Step 2: Append Google provider tests**

Append to `python/tests/test_providers_p3.py`:

```python
def _make_google_mock(text="Gemini reply", input_tokens=20, output_tokens=10):
    mock_response = MagicMock()
    mock_response.text = text
    mock_response.usage_metadata.prompt_token_count = input_tokens
    mock_response.usage_metadata.candidates_token_count = output_tokens
    mock_result = MagicMock()
    mock_result.response = mock_response
    mock_chat = MagicMock()
    mock_chat.send_message.return_value = mock_result
    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_chat
    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    return mock_genai


def test_google_provider_complete_returns_response():
    from truss.providers.google import GoogleProvider

    provider = GoogleProvider(api_key="test-key")
    provider._genai = _make_google_mock("Gemini reply", 20, 10)

    result = provider.complete(
        [LLMMessage(role="user", content="hello")],
        model="gemini-1.5-flash",
    )
    assert result.text == "Gemini reply"
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 10
    assert result.usage.cost_usd > 0


def test_google_provider_system_message_prepended():
    from truss.providers.google import GoogleProvider

    provider = GoogleProvider(api_key="test-key")
    mock_genai = _make_google_mock()
    provider._genai = mock_genai

    provider.complete(
        [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hi"),
        ],
        model="gemini-1.5-flash",
    )

    chat = mock_genai.GenerativeModel.return_value.start_chat.return_value
    call_args = chat.send_message.call_args[0][0]
    assert "You are helpful." in call_args
    assert "Hi" in call_args


def test_google_provider_records_usage_to_session():
    from truss.providers.google import GoogleProvider
    from truss.session import Session

    session = Session()
    provider = GoogleProvider(api_key="test-key", session=session)
    provider._genai = _make_google_mock(input_tokens=100, output_tokens=50)

    provider.complete([LLMMessage(role="user", content="hi")], model="gemini-1.5-flash")
    assert session.report().budget_used_usd > 0


def test_google_provider_missing_api_key_raises():
    import os
    from truss.providers.google import GoogleProvider

    original = os.environ.pop("GOOGLE_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GoogleProvider()
    finally:
        if original:
            os.environ["GOOGLE_API_KEY"] = original
```

- [ ] **Step 3: Run tests to verify they fail**

```
cd python && python -m pytest tests/test_providers_p3.py -k "google" -v
```

Expected: tests fail because current GoogleProvider is a stub that raises `NotImplementedError`.

- [ ] **Step 4: Replace google.py stub with full implementation**

Replace `python/src/truss/providers/google.py` entirely:

```python
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
```

- [ ] **Step 5: Run tests**

```
cd python && python -m pytest tests/test_providers_p3.py -k "google" -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add python/src/truss/providers/google.py python/tests/test_providers_p3.py python/pyproject.toml
git commit -m "feat: implement full GoogleProvider with Gemini API"
```

---

## Task 6: Full OllamaProvider

**Files:**
- Modify: `python/src/truss/providers/ollama.py` (replace stub)
- Modify: `python/tests/test_providers_p3.py`

- [ ] **Step 1: Append Ollama tests**

Append to `python/tests/test_providers_p3.py`:

```python
def test_ollama_provider_complete_returns_response():
    from truss.providers.ollama import OllamaProvider

    provider = OllamaProvider()
    provider._client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "Ollama reply"},
        "prompt_eval_count": 15,
        "eval_count": 8,
    }
    provider._client.post.return_value = mock_response

    result = provider.complete(
        [LLMMessage(role="user", content="hello")],
        model="llama3",
    )
    assert result.text == "Ollama reply"
    assert result.usage.input_tokens == 15
    assert result.usage.output_tokens == 8
    assert result.usage.cost_usd == 0.0


def test_ollama_provider_records_usage_to_session():
    from truss.providers.ollama import OllamaProvider
    from truss.session import Session

    session = Session()
    provider = OllamaProvider(session=session)
    provider._client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "hi"},
        "prompt_eval_count": 100,
        "eval_count": 50,
    }
    provider._client.post.return_value = mock_response

    provider.complete([LLMMessage(role="user", content="hi")], model="llama3")
    report = session.report()
    assert report.budget_used_usd == 0.0  # local model is free
    # tokens are still recorded
    assert report.budget_used_usd == 0.0


def test_ollama_provider_passes_messages_correctly():
    from truss.providers.ollama import OllamaProvider

    provider = OllamaProvider(base_url="http://localhost:11434")
    provider._client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"message": {"content": "ok"}, "prompt_eval_count": 0, "eval_count": 0}
    provider._client.post.return_value = mock_response

    provider.complete(
        [LLMMessage(role="user", content="hello")],
        model="mistral",
    )

    call_kwargs = provider._client.post.call_args
    payload = call_kwargs[1]["json"]
    assert payload["model"] == "mistral"
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "hello"
    assert payload["stream"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd python && python -m pytest tests/test_providers_p3.py -k "ollama" -v
```

Expected: tests fail because stub raises `NotImplementedError`.

- [ ] **Step 3: Replace ollama.py stub with full implementation**

Replace `python/src/truss/providers/ollama.py` entirely:

```python
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
```

- [ ] **Step 4: Run tests**

```
cd python && python -m pytest tests/test_providers_p3.py -k "ollama" -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/providers/ollama.py python/tests/test_providers_p3.py
git commit -m "feat: implement full OllamaProvider via httpx"
```

---

## Task 7: Run full suite + update public API

**Files:**
- Modify: `python/src/truss/__init__.py`

- [ ] **Step 1: Run full test suite**

```
cd python && python -m pytest -v 2>&1 | tail -10
```

Expected: all previous tests still pass + new Phase 3 tests. Total should be ~130+.

- [ ] **Step 2: Add new exports to __init__.py**

In `python/src/truss/__init__.py`, update the providers import line and `__all__`:

Replace:
```python
from truss.providers.base import LLMMessage, LLMUsage, LLMResponse, LLMProvider, COST_TABLE, compute_cost
```

With:
```python
from truss.providers.base import (
    LLMMessage, LLMUsage, LLMResponse, LLMProvider,
    StreamChunk, LLMStreamProvider,
    COST_TABLE, compute_cost,
)
```

Add to `__all__`:
```python
    "StreamChunk", "LLMStreamProvider",
```

- [ ] **Step 3: Verify import**

```
cd python && python -c "from truss import StreamChunk, LLMStreamProvider; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Run full suite one final time**

```
cd python && python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/__init__.py
git commit -m "feat: export StreamChunk and LLMStreamProvider from public API"
```

---

## Self-Review Against Spec

| Requirement | Task |
|---|---|
| `StreamChunk(text, is_final, usage?)` dataclass | Task 1 |
| `LLMStreamProvider` Protocol | Task 1 |
| Gemini + Ollama `COST_TABLE` entries | Task 1 |
| `AnthropicProvider.stream()` sync generator | Task 2 |
| `OpenAIProvider.stream()` sync generator with `include_usage` | Task 3 |
| `AnthropicProvider.async_complete()` | Task 4 |
| `OpenAIProvider.async_complete()` | Task 4 |
| `GoogleProvider` full impl — system msg handling, history, usage | Task 5 |
| `OllamaProvider` full impl — httpx, `/api/chat`, cost=0 | Task 6 |
| `pyproject.toml` optional dep groups for `google` and `ollama` | Task 5 |
| All new symbols exported from `truss.__init__` | Task 7 |

**Out of scope (Phase 4):**
- `async_stream()` (async generator streaming)
- LangGraph adapter
- Multi-provider router integration
- Observability / tracing hooks
