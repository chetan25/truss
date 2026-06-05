# Truss Phase 2 Python — Provider + Framework Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provider adapters (Anthropic full, OpenAI full, Google/Ollama stubs) and framework adapters (LangChain callback+LLM wrapper, CrewAI handoff+callback) to the existing `truss-ai` Python package.

**Architecture:** Option C (opt-in full stack) — callback/hook path for existing projects, wrapper path for new projects. All providers and framework adapters are optional-dep gated. Provider adapters auto-record to `Session` ledger if a session is passed. Framework adapters work with any LLM client via callbacks, or use Truss providers directly for full routing.

**Tech Stack:** Python 3.10+ · Pydantic v2 · anthropic>=0.40 · openai>=1.0 · langchain-core>=0.2 · crewai>=0.50 · pytest 8

---

## File Structure

```
python/src/truss/
├── providers/
│   ├── __init__.py       # re-exports
│   ├── base.py           # LLMMessage, LLMUsage, LLMResponse, LLMProvider, COST_TABLE, compute_cost
│   ├── anthropic.py      # AnthropicProvider (full)
│   ├── openai.py         # OpenAIProvider (full)
│   ├── google.py         # GoogleProvider (stub)
│   └── ollama.py         # OllamaProvider (stub)
├── adapters/
│   ├── langchain.py      # TrussMemory (existing) + TrussCallbackHandler + TrussLLM (new)
│   └── crewai.py         # PackHandoffTool, UnpackHandoffTool, TrussCrewCallback (new)
└── session.py            # add .envelope property (minor change)

python/tests/
├── test_providers.py     # all provider tests
├── test_adapters_langchain_p2.py   # new langchain adapter tests
└── test_adapters_crewai.py         # crewai adapter tests
```

---

## Task 1: Provider Base Types

**Files:**
- Create: `python/src/truss/providers/__init__.py`
- Create: `python/src/truss/providers/base.py`
- Create: `python/tests/test_providers.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_providers.py`:

```python
import pytest
from truss.providers.base import (
    LLMMessage, LLMUsage, LLMResponse, LLMProvider,
    COST_TABLE, compute_cost,
)


def test_cost_table_has_anthropic_models():
    assert "claude-haiku-4-5" in COST_TABLE
    assert "claude-sonnet-4-6" in COST_TABLE
    assert "claude-opus-4-8" in COST_TABLE


def test_cost_table_has_openai_models():
    assert "gpt-4o" in COST_TABLE
    assert "gpt-4o-mini" in COST_TABLE


def test_compute_cost_for_known_model():
    # claude-haiku-4-5: $0.001/1k input, $0.005/1k output
    cost = compute_cost("claude-haiku-4-5", input_tokens=1000, output_tokens=1000)
    assert abs(cost - 0.006) < 0.0001


def test_compute_cost_for_unknown_model_uses_default():
    cost = compute_cost("unknown-model-xyz", input_tokens=1000, output_tokens=0)
    assert cost == 0.001  # default input rate $0.001/1k


def test_llm_message_dataclass():
    msg = LLMMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_llm_usage_fields():
    usage = LLMUsage(input_tokens=100, output_tokens=50, cost_usd=0.01)
    assert usage.input_tokens + usage.output_tokens == 150


def test_llm_response_fields():
    usage = LLMUsage(input_tokens=10, output_tokens=5, cost_usd=0.001)
    resp = LLMResponse(text="hello", model="test-model", usage=usage)
    assert resp.text == "hello"
    assert resp.raw is None


def test_llm_provider_protocol_satisfied_by_duck_type():
    class FakeProvider:
        def complete(self, messages, model, **opts): ...

    assert isinstance(FakeProvider(), LLMProvider)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && python -m pytest tests/test_providers.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.providers'`

- [ ] **Step 3: Create providers package and base.py**

Create `python/src/truss/providers/__init__.py`:

```python
from truss.providers.base import LLMMessage, LLMUsage, LLMResponse, LLMProvider, COST_TABLE, compute_cost

__all__ = ["LLMMessage", "LLMUsage", "LLMResponse", "LLMProvider", "COST_TABLE", "compute_cost"]
```

Create `python/src/truss/providers/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


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


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        **opts: Any,
    ) -> LLMResponse: ...


COST_TABLE: dict[str, tuple[float, float]] = {
    # (input_$/1k, output_$/1k)
    "claude-haiku-4-5":  (0.001,   0.005),
    "claude-sonnet-4-6": (0.003,   0.015),
    "claude-opus-4-8":   (0.015,   0.075),
    "gpt-4o-mini":       (0.00015, 0.0006),
    "gpt-4o":            (0.005,   0.015),
    "gpt-4-turbo":       (0.010,   0.030),
    "o1":                (0.015,   0.060),
    "o1-mini":           (0.003,   0.012),
}

_DEFAULT_RATES = (0.001, 0.005)


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_TABLE.get(model, _DEFAULT_RATES)
    return (input_tokens / 1000) * rates[0] + (output_tokens / 1000) * rates[1]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && python -m pytest tests/test_providers.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/providers/ python/tests/test_providers.py
git commit -m "feat: add provider base types — LLMMessage, LLMUsage, LLMResponse, COST_TABLE"
```

---

## Task 2: AnthropicProvider

**Files:**
- Create: `python/src/truss/providers/anthropic.py`
- Modify: `python/tests/test_providers.py`
- Modify: `python/pyproject.toml`

- [ ] **Step 1: Add anthropic to dev dependencies**

Modify `python/pyproject.toml` — replace the `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
langchain = ["langchain-core>=0.2"]
anthropic = ["anthropic>=0.40"]
openai    = ["openai>=1.0"]
crewai    = ["crewai>=0.50"]
all       = ["anthropic>=0.40", "openai>=1.0", "langchain-core>=0.2", "crewai>=0.50"]
dev       = ["pytest>=8.0", "pytest-asyncio>=0.23", "anthropic>=0.40", "openai>=1.0", "langchain-core>=0.2"]
```

Install:

```bash
cd python && pip install -e ".[dev]"
```

- [ ] **Step 2: Append AnthropicProvider tests to test_providers.py**

Append to `python/tests/test_providers.py`:

```python
from unittest.mock import MagicMock


def _make_anthropic_response(text="Hello!", input_tokens=10, output_tokens=5):
    mock_content = MagicMock()
    mock_content.text = text
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    return mock_response


def test_anthropic_provider_complete_returns_response():
    from truss.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.messages.create.return_value = _make_anthropic_response("Test response", 100, 50)

    result = provider.complete(
        [LLMMessage(role="user", content="hello")],
        model="claude-haiku-4-5",
    )
    assert result.text == "Test response"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.usage.cost_usd > 0


def test_anthropic_provider_records_usage_to_session():
    from truss.providers.anthropic import AnthropicProvider
    from truss.session import Session

    session = Session()
    provider = AnthropicProvider(api_key="test-key", session=session)
    provider._client = MagicMock()
    provider._client.messages.create.return_value = _make_anthropic_response(input_tokens=200, output_tokens=100)

    provider.complete([LLMMessage(role="user", content="hi")], model="claude-haiku-4-5")

    report = session.report()
    assert report.budget_used_usd > 0


def test_anthropic_provider_missing_api_key_raises():
    import os
    from truss.providers.anthropic import AnthropicProvider

    original = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider()
    finally:
        if original:
            os.environ["ANTHROPIC_API_KEY"] = original


def test_anthropic_provider_circuit_breaker_trips():
    from truss.providers.anthropic import AnthropicProvider
    from truss.budget.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
    from truss.errors import BudgetExceeded

    cb = CircuitBreaker(CircuitBreakerConfig(max_requests_per_minute=0))
    provider = AnthropicProvider(api_key="test-key", circuit_breaker=cb)
    provider._client = MagicMock()

    with pytest.raises(BudgetExceeded):
        provider.complete([LLMMessage(role="user", content="hi")], model="claude-haiku-4-5")
```

- [ ] **Step 3: Run new tests to verify they fail**

```bash
cd python && python -m pytest tests/test_providers.py -k "anthropic" -v
```

Expected: `ModuleNotFoundError: No module named 'truss.providers.anthropic'`

- [ ] **Step 4: Implement anthropic.py**

Create `python/src/truss/providers/anthropic.py`:

```python
from __future__ import annotations

import os
import time
from typing import Any, Optional

from truss.providers.base import LLMMessage, LLMResponse, LLMUsage, compute_cost


class AnthropicProvider:
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

        response = self._client.messages.create(
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

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd python && python -m pytest tests/test_providers.py -v
```

Expected: `12 passed`

- [ ] **Step 6: Commit**

```bash
git add python/src/truss/providers/anthropic.py python/tests/test_providers.py python/pyproject.toml
git commit -m "feat: add AnthropicProvider with session recording and circuit breaker"
```

---

## Task 3: OpenAIProvider

**Files:**
- Create: `python/src/truss/providers/openai.py`
- Modify: `python/tests/test_providers.py`

- [ ] **Step 1: Append OpenAI tests to test_providers.py**

Append to `python/tests/test_providers.py`:

```python
def _make_openai_response(text="Hi!", prompt_tokens=10, completion_tokens=5):
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.model = "gpt-4o-mini"
    return mock_response


def test_openai_provider_complete_returns_response():
    from truss.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _make_openai_response("OpenAI reply", 50, 20)

    result = provider.complete(
        [LLMMessage(role="user", content="hello")],
        model="gpt-4o-mini",
    )
    assert result.text == "OpenAI reply"
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 20
    assert result.usage.cost_usd > 0


def test_openai_provider_records_usage_to_session():
    from truss.providers.openai import OpenAIProvider
    from truss.session import Session

    session = Session()
    provider = OpenAIProvider(api_key="test-key", session=session)
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _make_openai_response(prompt_tokens=100, completion_tokens=50)

    provider.complete([LLMMessage(role="user", content="hi")], model="gpt-4o-mini")

    report = session.report()
    assert report.budget_used_usd > 0


def test_openai_provider_missing_api_key_raises():
    import os
    from truss.providers.openai import OpenAIProvider

    original = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIProvider()
    finally:
        if original:
            os.environ["OPENAI_API_KEY"] = original
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd python && python -m pytest tests/test_providers.py -k "openai" -v
```

Expected: `ModuleNotFoundError: No module named 'truss.providers.openai'`

- [ ] **Step 3: Implement openai.py**

Create `python/src/truss/providers/openai.py`:

```python
from __future__ import annotations

import os
import time
from typing import Any, Optional

from truss.providers.base import LLMMessage, LLMResponse, LLMUsage, compute_cost


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
```

- [ ] **Step 4: Run all provider tests**

```bash
cd python && python -m pytest tests/test_providers.py -v
```

Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/providers/openai.py python/tests/test_providers.py
git commit -m "feat: add OpenAIProvider with session recording and circuit breaker"
```

---

## Task 4: Provider Stubs (Google + Ollama)

**Files:**
- Create: `python/src/truss/providers/google.py`
- Create: `python/src/truss/providers/ollama.py`
- Modify: `python/tests/test_providers.py`

- [ ] **Step 1: Append stub tests**

Append to `python/tests/test_providers.py`:

```python
def test_google_provider_raises_not_implemented():
    from truss.providers.google import GoogleProvider

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        provider = GoogleProvider()
        provider.complete([LLMMessage(role="user", content="hi")], model="gemini-pro")


def test_ollama_provider_raises_not_implemented():
    from truss.providers.ollama import OllamaProvider

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        provider = OllamaProvider()
        provider.complete([LLMMessage(role="user", content="hi")], model="llama3")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && python -m pytest tests/test_providers.py -k "google or ollama" -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement stubs**

Create `python/src/truss/providers/google.py`:

```python
from truss.providers.base import LLMMessage, LLMResponse


class GoogleProvider:
    def complete(self, messages: list[LLMMessage], model: str = "gemini-pro", **opts) -> LLMResponse:
        raise NotImplementedError(
            "GoogleProvider is not yet implemented. "
            "Track progress at github.com/your-org/truss — planned for Phase 3."
        )
```

Create `python/src/truss/providers/ollama.py`:

```python
from truss.providers.base import LLMMessage, LLMResponse


class OllamaProvider:
    def complete(self, messages: list[LLMMessage], model: str = "llama3", **opts) -> LLMResponse:
        raise NotImplementedError(
            "OllamaProvider is not yet implemented. "
            "Track progress at github.com/your-org/truss — planned for Phase 3."
        )
```

Update `python/src/truss/providers/__init__.py`:

```python
from truss.providers.base import LLMMessage, LLMUsage, LLMResponse, LLMProvider, COST_TABLE, compute_cost
from truss.providers.anthropic import AnthropicProvider
from truss.providers.openai import OpenAIProvider
from truss.providers.google import GoogleProvider
from truss.providers.ollama import OllamaProvider

__all__ = [
    "LLMMessage", "LLMUsage", "LLMResponse", "LLMProvider", "COST_TABLE", "compute_cost",
    "AnthropicProvider", "OpenAIProvider", "GoogleProvider", "OllamaProvider",
]
```

- [ ] **Step 4: Run all provider tests**

```bash
cd python && python -m pytest tests/test_providers.py -v
```

Expected: `17 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/providers/
git commit -m "feat: add GoogleProvider and OllamaProvider stubs"
```

---

## Task 5: Session.envelope Property

**Files:**
- Modify: `python/src/truss/session.py`
- Modify: `python/tests/test_session.py`

- [ ] **Step 1: Add test for envelope property**

Append to `python/tests/test_session.py`:

```python
def test_session_envelope_property_readable():
    from truss.session import Session
    s = Session()
    env = s.envelope
    assert env is s._envelope


def test_session_envelope_property_returns_agent_envelope():
    from truss.session import Session
    from truss.types import AgentEnvelope
    s = Session()
    assert isinstance(s.envelope, AgentEnvelope)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && python -m pytest tests/test_session.py -k "envelope_property" -v
```

Expected: `AttributeError: 'Session' object has no attribute 'envelope'`

- [ ] **Step 3: Add envelope property to Session**

In `python/src/truss/session.py`, add after the `session_id` property:

```python
@property
def envelope(self) -> AgentEnvelope:
    return self._envelope
```

- [ ] **Step 4: Run tests**

```bash
cd python && python -m pytest tests/test_session.py -v
```

Expected: all session tests pass (was 6, now 8).

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/session.py python/tests/test_session.py
git commit -m "feat: expose Session.envelope as read-only property"
```

---

## Task 6: TrussCallbackHandler (LangChain)

**Files:**
- Modify: `python/src/truss/adapters/langchain.py`
- Create: `python/tests/test_adapters_langchain_p2.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_adapters_langchain_p2.py`:

```python
import pytest
from unittest.mock import MagicMock


def make_llm_result(input_tokens=100, output_tokens=50, model="claude-haiku-4-5", format="anthropic"):
    """Build a LangChain LLMResult with token usage."""
    from langchain_core.outputs import LLMResult, ChatGeneration, ChatGenerationChunk
    from langchain_core.messages import AIMessage

    if format == "anthropic":
        llm_output = {"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}, "model": model}
    else:  # openai format
        llm_output = {"token_usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}, "model_name": model}

    return LLMResult(generations=[[]], llm_output=llm_output)


def test_callback_handler_records_usage_anthropic_format():
    from truss.adapters.langchain import TrussCallbackHandler
    from truss.session import Session

    session = Session()
    handler = TrussCallbackHandler(session)
    handler.on_llm_end(make_llm_result(100, 50, format="anthropic"))

    report = session.report()
    assert report.budget_used_usd > 0


def test_callback_handler_records_usage_openai_format():
    from truss.adapters.langchain import TrussCallbackHandler
    from truss.session import Session

    session = Session()
    handler = TrussCallbackHandler(session)
    handler.on_llm_end(make_llm_result(200, 100, format="openai"))

    report = session.report()
    assert report.budget_used_usd > 0


def test_callback_handler_silent_on_missing_usage():
    """If llm_output has no token counts, no exception and no usage recorded."""
    from truss.adapters.langchain import TrussCallbackHandler
    from truss.session import Session
    from langchain_core.outputs import LLMResult

    session = Session()
    handler = TrussCallbackHandler(session)
    handler.on_llm_end(LLMResult(generations=[[]], llm_output={}))

    report = session.report()
    assert report.budget_used_usd == 0.0


def test_callback_handler_on_llm_error_does_not_raise():
    from truss.adapters.langchain import TrussCallbackHandler
    from truss.session import Session

    session = Session()
    handler = TrussCallbackHandler(session)
    handler.on_llm_error(Exception("test error"))  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && python -m pytest tests/test_adapters_langchain_p2.py -v
```

Expected: `ImportError: cannot import name 'TrussCallbackHandler' from 'truss.adapters.langchain'`

- [ ] **Step 3: Add TrussCallbackHandler to langchain.py**

Append to `python/src/truss/adapters/langchain.py`:

```python
try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


def _require_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError("langchain-core required: pip install truss-ai[langchain]")


class TrussCallbackHandler:
    """LangChain callback that auto-records token usage into a Truss Session.

    Usage:
        session = Session(budget_usd=1.0)
        llm = ChatAnthropic(callbacks=[TrussCallbackHandler(session)])
    """

    def __init__(self, session: Any) -> None:
        _require_langchain()
        self._session = session

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        llm_output = (response.llm_output or {}) if hasattr(response, "llm_output") else {}

        # Anthropic format: {"usage": {"input_tokens": N, "output_tokens": N}}
        usage = llm_output.get("usage") or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # OpenAI format: {"token_usage": {"prompt_tokens": N, "completion_tokens": N}}
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
        pass  # no-op — never interrupt LangChain execution
```

Also add `from typing import Any` at the top of `langchain.py` if not already present.

- [ ] **Step 4: Run tests**

```bash
cd python && python -m pytest tests/test_adapters_langchain_p2.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/adapters/langchain.py python/tests/test_adapters_langchain_p2.py
git commit -m "feat: add TrussCallbackHandler for LangChain token usage recording"
```

---

## Task 7: TrussLLM (LangChain Wrapper)

**Files:**
- Modify: `python/src/truss/adapters/langchain.py`
- Modify: `python/tests/test_adapters_langchain_p2.py`

- [ ] **Step 1: Append TrussLLM tests**

Append to `python/tests/test_adapters_langchain_p2.py`:

```python
def test_truss_llm_call_returns_text():
    from truss.adapters.langchain import TrussLLM
    from truss.providers.base import LLMMessage, LLMResponse, LLMUsage

    mock_provider = MagicMock()
    mock_provider.complete.return_value = LLMResponse(
        text="LLM response text",
        model="claude-haiku-4-5",
        usage=LLMUsage(input_tokens=10, output_tokens=5, cost_usd=0.0001),
    )

    llm = TrussLLM(provider=mock_provider, default_model="claude-haiku-4-5")
    result = llm._call("What is 2+2?")
    assert result == "LLM response text"


def test_truss_llm_passes_prompt_as_user_message():
    from truss.adapters.langchain import TrussLLM
    from truss.providers.base import LLMMessage, LLMResponse, LLMUsage

    mock_provider = MagicMock()
    mock_provider.complete.return_value = LLMResponse(
        text="answer",
        model="claude-haiku-4-5",
        usage=LLMUsage(input_tokens=5, output_tokens=3, cost_usd=0.00005),
    )

    llm = TrussLLM(provider=mock_provider, default_model="claude-haiku-4-5")
    llm._call("my prompt")

    called_messages = mock_provider.complete.call_args[1]["messages"]
    assert called_messages[0].role == "user"
    assert called_messages[0].content == "my prompt"


def test_truss_llm_type_property():
    from truss.adapters.langchain import TrussLLM
    from unittest.mock import MagicMock

    llm = TrussLLM(provider=MagicMock(), default_model="claude-haiku-4-5")
    assert llm._llm_type == "truss"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && python -m pytest tests/test_adapters_langchain_p2.py -k "truss_llm" -v
```

Expected: `ImportError: cannot import name 'TrussLLM'`

- [ ] **Step 3: Add TrussLLM to langchain.py**

Append to `python/src/truss/adapters/langchain.py`:

```python
class TrussLLM:
    """LangChain-compatible LLM that routes calls through a Truss LLMProvider.

    Usage:
        from truss.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider(api_key="...", session=session)
        llm = TrussLLM(provider=provider, default_model="claude-haiku-4-5")
        result = llm.invoke("Hello")
    """

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
```

- [ ] **Step 4: Run all LangChain p2 tests**

```bash
cd python && python -m pytest tests/test_adapters_langchain_p2.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/adapters/langchain.py python/tests/test_adapters_langchain_p2.py
git commit -m "feat: add TrussLLM LangChain wrapper for Truss provider routing"
```

---

## Task 8: CrewAI Adapters

**Files:**
- Create: `python/src/truss/adapters/crewai.py`
- Create: `python/tests/test_adapters_crewai.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_adapters_crewai.py`:

```python
import pytest
import json
from unittest.mock import MagicMock


def test_pack_handoff_tool_returns_json_string():
    from truss.adapters.crewai import PackHandoffTool
    from truss.session import Session
    from truss.types import ContextBlock, ContextRole, ContextWeight, AgentEnvelope

    session = Session()
    session._envelope.context.append(
        ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL, content="important info", source="user")
    )
    session._envelope.budget_usd_remaining = 1.0

    tool = PackHandoffTool(session=session)
    result = tool._run(json.dumps({
        "task": "analyse pricing",
        "carry_weights": ["critical"],
        "budget_fraction": 0.3,
    }))

    envelope_data = json.loads(result)
    assert envelope_data["task"] == "analyse pricing"


def test_pack_handoff_carries_only_requested_weights():
    from truss.adapters.crewai import PackHandoffTool
    from truss.session import Session
    from truss.types import ContextBlock, ContextRole, ContextWeight

    session = Session()
    session._envelope.context = [
        ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL, content="critical", source="u"),
        ContextBlock(role=ContextRole.BACKGROUND, weight=ContextWeight.BACKGROUND, content="noise", source="u"),
    ]
    session._envelope.budget_usd_remaining = 1.0

    tool = PackHandoffTool(session=session)
    result = tool._run(json.dumps({"task": "sub-task", "carry_weights": ["critical"], "budget_fraction": 0.5}))
    data = json.loads(result)
    assert len(data["context"]) == 1
    assert data["context"][0]["weight"] == 3  # CRITICAL = 3


def test_unpack_handoff_tool_returns_context_blocks():
    from truss.adapters.crewai import PackHandoffTool, UnpackHandoffTool
    from truss.session import Session
    from truss.types import ContextBlock, ContextRole, ContextWeight

    session = Session()
    session._envelope.context = [
        ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.HIGH, content="finding A", source="agent"),
    ]
    session._envelope.budget_usd_remaining = 1.0

    pack_tool = PackHandoffTool(session=session)
    envelope_json = pack_tool._run(json.dumps({"task": "next", "carry_weights": ["high"], "budget_fraction": 0.5}))

    unpack_tool = UnpackHandoffTool()
    blocks_json = unpack_tool._run(envelope_json)
    blocks = json.loads(blocks_json)
    assert len(blocks) == 1
    assert blocks[0]["content"] == "finding A"


def test_truss_crew_callback_creates_checkpoint():
    from truss.adapters.crewai import TrussCrewCallback
    from truss.session import Session

    session = Session()
    callback = TrussCrewCallback(session=session)
    callback(MagicMock())  # simulate step output
    callback(MagicMock())

    report = session.report()
    assert report.checkpoint_count == 2


def test_truss_crew_callback_does_not_raise_on_error():
    from truss.adapters.crewai import TrussCrewCallback
    from truss.session import Session
    from unittest.mock import patch

    session = Session()
    callback = TrussCrewCallback(session=session)

    with patch.object(session, "checkpoint", side_effect=RuntimeError("checkpoint failed")):
        callback(MagicMock())  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && python -m pytest tests/test_adapters_crewai.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.adapters.crewai'`

- [ ] **Step 3: Implement crewai.py**

Create `python/src/truss/adapters/crewai.py`:

```python
from __future__ import annotations

import json
from typing import Any, Optional

from truss.types import ContextBlock, ContextRole, ContextWeight, AgentEnvelope
from truss.handoff.envelope import pack, unpack, BudgetCarve


_WEIGHT_MAP: dict[str, ContextWeight] = {
    "critical":   ContextWeight.CRITICAL,
    "high":       ContextWeight.HIGH,
    "normal":     ContextWeight.NORMAL,
    "background": ContextWeight.BACKGROUND,
}


class PackHandoffTool:
    """CrewAI-compatible tool that packs the current session context into a JSON envelope.

    Agents call this tool to hand off structured context to the next agent.
    Input JSON: {"task": str, "carry_weights": list[str], "budget_fraction": float}
    Returns: AgentEnvelope as a JSON string.
    """

    name: str = "pack_handoff"
    description: str = (
        "Pack the current session context into a structured JSON envelope to pass to another agent. "
        "Input must be JSON with keys: 'task' (str), "
        "'carry_weights' (list of 'critical'/'high'/'normal'/'background'), "
        "'budget_fraction' (float 0.0-1.0)."
    )

    def __init__(self, session: Any) -> None:
        self._session = session

    def _run(self, tool_input: str) -> str:
        data = json.loads(tool_input)
        carry_weights = [_WEIGHT_MAP[w] for w in data.get("carry_weights", ["critical", "high"])]
        fraction = float(data.get("budget_fraction", 0.3))

        child = pack(
            self._session.envelope,
            task=data["task"],
            carry_weights=carry_weights,
            budget_carve=BudgetCarve.percent(fraction),
        )
        return child.model_dump_json()


class UnpackHandoffTool:
    """CrewAI-compatible tool that unpacks a received JSON envelope into context blocks.

    Input: the JSON envelope string produced by PackHandoffTool.
    Returns: JSON array of context blocks with fields: content, role, weight.
    """

    name: str = "unpack_handoff"
    description: str = (
        "Unpack a JSON envelope received from another agent into a list of context blocks. "
        "Input: the JSON envelope string."
    )

    def _run(self, envelope_json: str) -> str:
        envelope = AgentEnvelope.model_validate_json(envelope_json)
        blocks = unpack(envelope)
        return json.dumps([
            {"content": b.content, "role": b.role.value, "weight": int(b.weight), "source": b.source}
            for b in blocks
        ])


class TrussCrewCallback:
    """CrewAI step_callback that auto-checkpoints the Truss session after every agent step.

    Usage:
        crew = Crew(agents=[...], tasks=[...], step_callback=TrussCrewCallback(session))
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self._step_count = 0

    def __call__(self, step_output: Any) -> None:
        self._step_count += 1
        try:
            self._session.checkpoint(f"step-{self._step_count}")
        except Exception:
            pass  # never interrupt crew execution
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && python -m pytest tests/test_adapters_crewai.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/adapters/crewai.py python/tests/test_adapters_crewai.py
git commit -m "feat: add CrewAI PackHandoffTool, UnpackHandoffTool, TrussCrewCallback"
```

---

## Task 9: Python Public API + pyproject.toml

**Files:**
- Modify: `python/src/truss/__init__.py`
- Modify: `python/pyproject.toml`

- [ ] **Step 1: Update __init__.py to export new symbols**

Replace `python/src/truss/__init__.py` — add to the existing exports:

```python
from truss.errors import TrussError, BudgetExceeded, ToolOutOfScope, CheckpointNotFound, FenceLockConflict
from truss.types import (
    ContextBlock, ContextRole, ContextWeight, estimate_tokens,
    AgentEnvelope, ModelTier, EvidenceRef, DecisionRecord,
)
from truss.context.surgeon import compress, SurgeonConfig, SurgeonResult, CompressionStrategy, score_relevance, detect_contradiction
from truss.handoff.envelope import pack, unpack, BudgetCarve
from truss.budget.config import BudgetConfig, BudgetLimit, BudgetWindow, AlertConfig, ExceededAction
from truss.budget.ledger import LedgerEntry, LedgerStore, UsageReport
from truss.budget.memory_store import InMemoryStore
from truss.budget.sqlite_store import SqliteLedgerStore
from truss.budget.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitTrip
from truss.coord.checkpoint import Checkpoint, CheckpointMeta, CheckpointStore, InMemoryCheckpointStore
from truss.coord.sqlite_checkpoint import SqliteCheckpointStore
from truss.fence.memory_fence import FenceStore, InMemoryFence, LockHandle
from truss.router.router import ModelSpec, RouterConfig, RouterRule, route
from truss.mcp.interceptor import McpManifest, McpInterceptor, McpCall
from truss.session import Session, SessionReport
from truss.adapters.langchain import TrussMemory
# Phase 2 additions
from truss.providers.base import LLMMessage, LLMUsage, LLMResponse, LLMProvider, COST_TABLE, compute_cost
from truss.providers.anthropic import AnthropicProvider
from truss.providers.openai import OpenAIProvider
from truss.providers.google import GoogleProvider
from truss.providers.ollama import OllamaProvider
from truss.adapters.langchain import TrussCallbackHandler, TrussLLM
from truss.adapters.crewai import PackHandoffTool, UnpackHandoffTool, TrussCrewCallback

__all__ = [
    # Errors
    "TrussError", "BudgetExceeded", "ToolOutOfScope", "CheckpointNotFound", "FenceLockConflict",
    # Types
    "ContextBlock", "ContextRole", "ContextWeight", "estimate_tokens",
    "AgentEnvelope", "ModelTier", "EvidenceRef", "DecisionRecord",
    # Context
    "compress", "SurgeonConfig", "SurgeonResult", "CompressionStrategy", "score_relevance", "detect_contradiction",
    # Handoff
    "pack", "unpack", "BudgetCarve",
    # Budget
    "BudgetConfig", "BudgetLimit", "BudgetWindow", "AlertConfig", "ExceededAction",
    "LedgerEntry", "LedgerStore", "UsageReport",
    "InMemoryStore", "SqliteLedgerStore",
    "CircuitBreaker", "CircuitBreakerConfig", "CircuitTrip",
    # Coord
    "Checkpoint", "CheckpointMeta", "CheckpointStore", "InMemoryCheckpointStore", "SqliteCheckpointStore",
    # Fence
    "FenceStore", "InMemoryFence", "LockHandle",
    # Router
    "ModelSpec", "RouterConfig", "RouterRule", "route",
    # MCP
    "McpManifest", "McpInterceptor", "McpCall",
    # Session
    "Session", "SessionReport",
    # Adapters (Phase 1)
    "TrussMemory",
    # Providers (Phase 2)
    "LLMMessage", "LLMUsage", "LLMResponse", "LLMProvider", "COST_TABLE", "compute_cost",
    "AnthropicProvider", "OpenAIProvider", "GoogleProvider", "OllamaProvider",
    # Adapters (Phase 2)
    "TrussCallbackHandler", "TrussLLM",
    "PackHandoffTool", "UnpackHandoffTool", "TrussCrewCallback",
]
```

- [ ] **Step 2: Verify import works**

```bash
cd python && python -c "import truss; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run the full test suite**

```bash
cd python && python -m pytest -v 2>&1 | tail -10
```

Expected: all tests pass (around 92+ total).

- [ ] **Step 4: Commit**

```bash
git add python/src/truss/__init__.py python/pyproject.toml
git commit -m "feat: export Phase 2 providers and adapters from public API"
```

---

## Self-Review Against Spec

| Spec requirement | Task |
|---|---|
| `LLMMessage`, `LLMUsage`, `LLMResponse`, `LLMProvider` Protocol | Task 1 |
| `COST_TABLE` with Anthropic + OpenAI models, `compute_cost()` | Task 1 |
| `AnthropicProvider` — full, records to session, checks circuit breaker | Task 2 |
| `OpenAIProvider` — full, records to session, checks circuit breaker | Task 3 |
| `GoogleProvider`, `OllamaProvider` — stubs with `NotImplementedError` | Task 4 |
| Optional dep gating with helpful `ImportError` messages | Tasks 2, 3 |
| `Session.envelope` read-only property | Task 5 |
| `TrussCallbackHandler` — normalizes Anthropic + OpenAI LLMResult formats | Task 6 |
| `TrussLLM` — wraps provider, `_llm_type = "truss"`, `_call()` | Task 7 |
| `PackHandoffTool`, `UnpackHandoffTool` — JSON round-trip | Task 8 |
| `TrussCrewCallback` — `step_callback`, auto-checkpoint, silent on error | Task 8 |
| All symbols exported from `truss.__init__` | Task 9 |
| `pyproject.toml` optional dep groups for each provider/framework | Task 2 (dev group) |

**Gaps not in this plan (out of scope per spec):**
- Streaming support for providers (Phase 3)
- Async `complete()` (Phase 3)
- `crewai` not installed in dev deps — tests mock the tool interface directly, no CrewAI runtime needed
