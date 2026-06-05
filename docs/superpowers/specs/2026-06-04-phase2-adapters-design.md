# Truss Phase 2 — Provider Adapters + Framework Adapters Design

**Date:** 2026-06-04
**Status:** Approved
**Scope:** Provider adapters (Anthropic, OpenAI, stubs for Google/Ollama) and framework adapters (LangChain callback+wrapper, CrewAI handoff, Vercel AI SDK middleware) for Python and TypeScript.

---

## Context

Phase 1 delivered all 7 Truss modules (Context Surgeon, Handoff, Budget/Ledger, Router, MCP, Checkpoints, Fence) as pure Python and TypeScript libraries with 75 + 65 tests. Phase 2 adds the integration layer: normalized provider clients that auto-record usage and enforce budgets, and framework adapters that drop into existing LangChain, CrewAI, and Vercel AI SDK projects with minimal friction.

**Design decision: Option C (opt-in full stack)**
- Callback/hook path for existing projects — add one line, get budget tracking
- Wrapper/provider path for new projects — full routing, circuit breaker, auto-recording
- Provider adapters work standalone or as backends for framework adapters

---

## Module A: Provider Adapters

### Shared Base Types

Same shape in Python and TypeScript:

| Type | Fields |
|------|--------|
| `LLMMessage` | `role: "user" \| "assistant" \| "system"`, `content: str` |
| `LLMUsage` | `input_tokens: int`, `output_tokens: int`, `cost_usd: float` |
| `LLMResponse` | `text: str`, `model: str`, `usage: LLMUsage`, `raw: Any` |
| `LLMProvider` | Protocol/interface: `complete(messages, model, **opts) → LLMResponse` |

`LLMProvider` is a `Protocol` (Python) / `interface` (TypeScript) — any object with a `complete()` method satisfies it. Providers optionally accept a `Session` at construction; if provided, every call auto-records usage to the ledger and checks the circuit breaker.

### Built-in Cost Table

Static dict mapping model ID → `(input_$/1k, output_$/1k)`. Lives in `base.py`/`base.ts`. Providers use it to compute `cost_usd` on every response. Updated manually when prices change.

```python
COST_TABLE = {
    # Anthropic
    "claude-haiku-4-5":   (0.001,  0.005),
    "claude-sonnet-4-6":  (0.003,  0.015),
    "claude-opus-4-8":    (0.015,  0.075),
    # OpenAI
    "gpt-4o-mini":        (0.00015, 0.0006),
    "gpt-4o":             (0.005,   0.015),
    "gpt-4-turbo":        (0.010,   0.030),
    "o1":                 (0.015,   0.060),
    "o1-mini":            (0.003,   0.012),
}
```

Unknown models default to `(0.001, 0.005)` and emit a warning.

### AnthropicProvider (Python + TypeScript — full)

Uses the official `anthropic` SDK. Wraps `client.messages.create()`.

**Construction:**
```python
provider = AnthropicProvider(
    api_key=os.environ["ANTHROPIC_API_KEY"],  # or reads env automatically
    session=session,          # optional — enables auto-recording to ledger
    circuit_breaker=breaker,  # optional — checked before every call
    default_model="claude-haiku-4-5",
)
```

`session` and `circuit_breaker` are independent optional parameters. Either, both, or neither can be passed. The `Session` class itself does not hold a `CircuitBreaker` — callers wire them together explicitly.

**complete():**
1. Calls `anthropic.messages.create(messages, model, max_tokens)`
2. Extracts `usage.input_tokens`, `usage.output_tokens` from response
3. Looks up cost in cost table
4. If session provided: checks circuit breaker → raises `BudgetExceeded` if tripped, else records `LedgerEntry`
5. Returns `LLMResponse`

**Error mapping:**
- `anthropic.APIStatusError` (4xx/5xx) → `TrussError` with original message preserved
- `anthropic.AuthenticationError` → `ValueError("ANTHROPIC_API_KEY invalid or missing")`

### OpenAIProvider (Python + TypeScript — full)

Uses the official `openai` SDK. Wraps `client.chat.completions.create()`.

Same contract as `AnthropicProvider`. Usage extracted from `response.usage.prompt_tokens` / `completion_tokens`.

### GoogleProvider + OllamaProvider (stubs)

Raise `NotImplementedError` with message directing to the roadmap:
```python
raise NotImplementedError(
    "GoogleProvider is not yet implemented. "
    "Track progress at github.com/your-org/truss/issues/XX"
)
```

### Python File Structure

```
python/src/truss/providers/
├── __init__.py
├── base.py          # LLMMessage, LLMUsage, LLMResponse, LLMProvider protocol, COST_TABLE
├── anthropic.py     # AnthropicProvider
├── openai.py        # OpenAIProvider
├── google.py        # GoogleProvider (stub)
└── ollama.py        # OllamaProvider (stub)
```

### TypeScript File Structure

```
typescript/src/providers/
├── index.ts
├── base.ts          # interfaces, COST_TABLE
├── anthropic.ts     # AnthropicProvider
├── openai.ts        # OpenAIProvider
├── google.ts        # stub
└── ollama.ts        # stub
```

### Optional Dependency Gating

All SDK imports are inside the class `__init__` / constructor — not at module level. If the dependency is missing:

```python
# Python
try:
    import anthropic as _anthropic
except ImportError:
    raise ImportError(
        "anthropic package required: pip install truss-ai[anthropic]"
    ) from None
```

```typescript
// TypeScript — checked at runtime in constructor
```

**New pyproject.toml optional groups:**
```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
openai    = ["openai>=1.0"]
langchain = ["langchain-core>=0.2"]
crewai    = ["crewai>=0.50"]
all       = ["anthropic>=0.40", "openai>=1.0", "langchain-core>=0.2", "crewai>=0.50"]
```

---

## Module B: Framework Adapters

### B1 — LangChain (Python)

Two additions to the existing `python/src/truss/adapters/langchain.py`. `TrussMemory` is unchanged.

#### `TrussCallbackHandler`

Extends LangChain's `BaseCallbackHandler`. Drop into any existing setup:

```python
from truss.adapters.langchain import TrussCallbackHandler

session = Session(budget_usd=1.0)
llm = ChatAnthropic(callbacks=[TrussCallbackHandler(session)])
# Every LLM call now auto-records usage + checks circuit breaker
```

**Implements:**
- `on_llm_end(response: LLMResult, **kwargs)` — extracts `token_usage` from `response.llm_output`, calls `session.record_usage(input_tokens, output_tokens, cost_usd, model)`
- `on_llm_error(error, **kwargs)` — records the error in session metadata (no-op if session has no error store)

LangChain's `LLMResult.llm_output` dict varies by provider. Handler normalizes:
- `"token_usage"` key (OpenAI format)
- `"usage"` key (Anthropic format via `langchain_anthropic`)
- Falls back to zero if neither key present (silent degradation, no crash)

#### `TrussLLM`

Extends `BaseLLM`. For new projects that want routing through a Truss provider:

```python
from truss.adapters.langchain import TrussLLM
from truss.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"], session=session)
llm = TrussLLM(provider=provider, default_model="claude-haiku-4-5")
```

**Implements:**
- `_call(prompt, stop, run_manager, **kwargs)` — converts prompt string to `[LLMMessage(role="user", content=prompt)]`, calls `provider.complete()`, returns `response.text`
- `_llm_type` property returns `"truss"`

Usage is recorded by the provider (not the wrapper) — no double-counting.

### B2 — CrewAI (Python)

New file: `python/src/truss/adapters/crewai.py`

#### `TrussHandoff`

A CrewAI-compatible `Tool` that serializes `AgentEnvelope` objects between agents. Agents that need to pass structured context to the next agent use this tool.

```python
from truss.adapters.crewai import TrussHandoff
from truss import Session, ContextWeight

session = Session()
handoff_tool = TrussHandoff(session=session)

# Agent A packs context for Agent B
envelope_json = handoff_tool.pack(
    task="Analyse pricing data found so far",
    carry_weights=[ContextWeight.CRITICAL, ContextWeight.HIGH],
    budget_fraction=0.3,
)

# Agent B receives and unpacks
blocks = handoff_tool.unpack(envelope_json)
```

**Implements:**
- `pack(task, carry_weights, budget_fraction) → str` — calls `truss.pack()` on `session.envelope` (exposed as a read-only property on `Session`, see below), serializes result as JSON string
- `unpack(envelope_json) → list[ContextBlock]` — deserializes JSON, calls `truss.unpack()`
- `name = "truss_handoff"`, `description` explains to the CrewAI LLM what the tool does

CrewAI requires tools to be `BaseTool` subclasses with a `_run(input)` method. `TrussHandoff` implements this with `input` parsed as JSON `{"action": "pack"|"unpack", ...}`.

**Required Session change:** `Session._envelope` is exposed as a public read-only property:
```python
@property
def envelope(self) -> AgentEnvelope:
    return self._envelope
```
This is a non-breaking addition — `_envelope` remains accessible for tests.

#### `TrussCrewCallback`

A callable passed to `Crew(step_callback=...)`. CrewAI calls `step_callback(step_output)` after every agent step. Auto-checkpoints the session:

```python
from truss.adapters.crewai import TrussCrewCallback

callback = TrussCrewCallback(session)
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    step_callback=callback,
)
```

**Implements `__call__(step_output)`:**
- Calls `session.checkpoint(f"step-{callback._step_count}")` where `_step_count` increments each call
- No-op on errors (crew execution must not be interrupted by a checkpoint failure)

### B3 — Vercel AI SDK (TypeScript)

New file: `typescript/src/adapters/vercel.ts`

#### `wrapModel(model, session, opts?)`

Wraps any Vercel AI SDK language model object. Returns a wrapped model with the same interface — drop-in replacement:

```typescript
import { anthropic } from '@ai-sdk/anthropic';
import { generateText } from 'ai';
import { wrapModel } from 'truss-ai';

const session = new Session({ budgetUsd: 1.0 });
const model = wrapModel(anthropic('claude-haiku-4-5'), session);

const { text, usage } = await generateText({ model, prompt: 'Hello' });
// session.recordUsage() called automatically with usage.promptTokens, completionTokens
```

**Implementation:**
Vercel AI SDK models implement a `doGenerate()` method. `wrapModel` creates a proxy that:
1. Checks the circuit breaker before the call
2. Calls the original `doGenerate()`
3. Extracts `usage.promptTokens` + `completionTokens` from the result
4. Calls `session.recordUsage(inputTokens, outputTokens, 0, modelId)`
5. Returns the original result unchanged

Streaming: wraps `doStream()` similarly, recording usage from the final chunk's `usage` field.

#### `createTrussMiddleware(session)`

Alternative for projects already using Vercel AI middleware chains:

```typescript
import { experimental_wrapLanguageModel as wrapLanguageModel } from 'ai';
import { createTrussMiddleware } from 'truss-ai';

const model = wrapLanguageModel({
  model: anthropic('claude-haiku-4-5'),
  middleware: createTrussMiddleware(session),
});
```

Implements Vercel AI's `LanguageModelV1Middleware` interface:
- `wrapGenerate({ doGenerate, params })` — circuit breaker + usage recording
- `wrapStream({ doStream, params })` — same for streaming

Both `wrapModel` and `createTrussMiddleware` are provided because different Vercel AI SDK versions and use cases favour one over the other; they have identical behaviour.

### B3 TypeScript File Structure

```
typescript/src/adapters/
├── vercel.ts        # wrapModel, createTrussMiddleware
```

---

## Testing Strategy

All tests run without live API calls.

| Component | Approach |
|-----------|----------|
| `AnthropicProvider` | Mock `anthropic.Anthropic` client; assert `LLMResponse` fields and `session.record_usage` called |
| `OpenAIProvider` | Mock `openai.OpenAI` client; same assertions |
| Stubs | Assert `NotImplementedError` raised with message containing "not yet implemented" |
| `TrussCallbackHandler` | Construct real `LLMResult` with token_usage dict; assert `session.record_usage` called |
| `TrussLLM` | Mock provider's `complete()`; assert `_call()` returns `response.text` |
| `TrussHandoff` | Full pack/unpack round-trip in-process; assert context blocks preserved |
| `TrussCrewCallback` | Call `on_task_end` directly; assert session checkpoint created |
| `wrapModel` (TS) | Mock `doGenerate` return value; assert usage recorded in session |
| `createTrussMiddleware` (TS) | Same mock pattern via middleware interface |

---

## Dependency Installation

```bash
# Python — install only what you need
pip install truss-ai[anthropic]           # Anthropic provider
pip install truss-ai[openai]              # OpenAI provider
pip install truss-ai[langchain]           # LangChain adapters
pip install truss-ai[crewai]              # CrewAI adapter
pip install truss-ai[all]                 # Everything

# TypeScript
npm install truss-ai @anthropic-ai/sdk    # Anthropic provider
npm install truss-ai openai               # OpenAI provider
npm install truss-ai ai                   # Vercel AI SDK adapter
```

---

## What Is Explicitly Out of Scope

- Streaming support in `TrussLLM` / `TrussCallbackHandler` (Phase 3)
- Async `complete()` on providers (Phase 3 — sync is enough for Phase 2 use cases)
- CrewAI memory integration (separate from handoff — Phase 3)
- Google / Ollama full implementation (stubs only, Phase 3)
- Redis store (Phase 3)
- Semantic dedup / Summarise compression strategies (Phase 3)
- `TokenCounter` trait / provider-specific token counting (Phase 3)

---

## File Change Summary

**Python — new files:**
- `python/src/truss/providers/__init__.py`
- `python/src/truss/providers/base.py`
- `python/src/truss/providers/anthropic.py`
- `python/src/truss/providers/openai.py`
- `python/src/truss/providers/google.py`
- `python/src/truss/providers/ollama.py`
- `python/src/truss/adapters/crewai.py`
- `python/tests/test_providers.py`
- `python/tests/test_adapters_langchain_p2.py`
- `python/tests/test_adapters_crewai.py`

**Python — modified files:**
- `python/src/truss/adapters/langchain.py` (add `TrussCallbackHandler`, `TrussLLM`)
- `python/src/truss/__init__.py` (export new providers + adapters)
- `python/pyproject.toml` (new optional dep groups)

**TypeScript — new files:**
- `typescript/src/providers/index.ts`
- `typescript/src/providers/base.ts`
- `typescript/src/providers/anthropic.ts`
- `typescript/src/providers/openai.ts`
- `typescript/src/providers/google.ts`
- `typescript/src/providers/ollama.ts`
- `typescript/src/adapters/vercel.ts`
- `typescript/tests/providers.test.ts`
- `typescript/tests/vercel.test.ts`

**TypeScript — modified files:**
- `typescript/src/index.ts` (export new providers + adapter)
