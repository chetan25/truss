# Truss

> The structural layer for agentic AI workflows.

LLM-agnostic · MCP-native · Python + TypeScript

---

## What it is

Truss is a composable infrastructure library that solves the six problems every team building production agentic systems hits from scratch:

| Problem | How teams solve it today |
|---|---|
| Context bloat | Manual truncation, ad-hoc summarisation |
| Agent handoff loss | Raw string dumps between agents |
| Faulty logic from missing context | Prompt engineering + prayer |
| MCP trust gap | Nothing — no standard exists |
| Token cost explosion | Per-call `max_tokens`, monthly billing surprises |
| Multi-LLM routing | Custom if/else per team |

Truss solves all six with one install.

---

## Architecture

Truss is structured as a Python core with a TypeScript surface. All logic (context compression, token counting, envelope packing, MCP interception, ledger accounting) lives in the Python package, with a TypeScript SDK wrapping the same primitives.

```
truss/
├── python/truss/       # Python core — all logic + framework adapters
├── typescript/         # TypeScript SDK → npm install truss-ai (Phase 2)
└── examples/           # Reference implementations
```

---

## Modules

### Module 1 — Context Surgeon
Compresses agent context before each LLM call using weighted pruning and sliding window strategies.

Strategies: `SlidingWindow`, `WeightedPrune`, `Hybrid` (Phase 1) · `SemanticDedup`, `Summarise` (Phase 2)

### Module 2 — Agent Handoff
Typed `AgentEnvelope` format for agent-to-agent handoffs. Carries compressed context, evidence, decisions, scope, and budget — framework-agnostic. Eliminates raw string dumps between agents.

### Module 3 — Token Budget + Session Ledger
Per-session, per-user, per-agent, and global budget enforcement with configurable actions (block, fallback, queue, graceful response). Built-in circuit breaker catches runaway agents via rate, cost velocity, retry depth, and repeated-prompt detection.

Storage: in-memory (`InMemoryStore`) or file-backed (`SqliteLedgerStore`). Redis support in Phase 2.

### Module 4 — Multi-LLM Router *(Phase 2)*
Routes tasks to the right model tier (cheap/standard/premium/auto) based on complexity, token budget, latency requirements, or keyword rules. Provider adapters: Anthropic, OpenAI, Google, Ollama.

### Module 5 — MCP Interceptor *(Phase 2)*
Intercepts MCP tool calls against a JSON scope manifest before they execute. Enforces path allow/deny lists, per-tool config, and trust levels. Every call is audit-logged.

### Module 6 — Checkpoints + Rollback
Snapshot an agent's full envelope state at any point. Roll back on failure and resume from a known-good state. Storage: in-memory (Phase 1), SQLite/Redis (Phase 2).

### Module 7 — Fence
TTL-based distributed lock store for multi-agent coordination. Prevents race conditions when two agents write to shared state. In-process `InMemoryFence` in Phase 1; `RedisFence` in Phase 2.

---

## Quick Start

```bash
pip install truss-ai
```

```python
from truss import Session, ContextBlock, ContextRole, ContextWeight

async with Session(budget_usd=1.00) as s:
    blocks = [
        ContextBlock(ContextRole.Task, ContextWeight.Critical, "your task here", "user"),
        # ... more blocks
    ]

    result = s.compress(blocks)

    response = await your_llm.complete(result.blocks)
    s.record_usage(response.usage.input_tokens, response.usage.output_tokens)

    print(s.report())
    # Context: 12,400 → 4,200 tokens (saved 8,200)
    # Est. savings: $0.0082
    # Budget: $0.18 of $1.00 used
```

---

## Framework Adapters

| Framework | Language | Adapter | Phase |
|---|---|---|---|
| LangChain | Python | `TrussMemory` | 1 |
| LangGraph | Python | `TrussNode` | 2 |
| CrewAI | Python | `TrussHandoff` | 2 |
| AutoGen | Python | `TrussMiddleware` | 2 |
| Pydantic AI | Python | `TrussRunHooks` | 2 |
| Vercel AI SDK | TypeScript | `trussMiddleware()` | 2 |
| Mastra | TypeScript | `TrussStep` | 2 |
| Claude Code | TypeScript | MCP server | 3 |
| Raw API | Both | `Session` | 1 |

---

## Packages

| Surface | Package | Import |
|---|---|---|
| PyPI | `truss-ai` | `import truss` |
| npm | `truss-ai` | `import { Session } from 'truss-ai'` |

---

## Build Phases

- **Phase 1** (Weeks 1–8): Python core. All 7 modules, LangChain adapter, `pip install truss-ai`.
- **Phase 2** (Weeks 9–16): Multi-LLM router, MCP interceptor, TypeScript surface, `npm install truss-ai`, additional framework adapters.
- **Phase 3** (Weeks 17–24): Claude Code MCP server, trust registry, embedding classifier, docs site, `0.1.0` public release.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Core engine | Python 3.11+ |
| Async runtime | asyncio |
| TypeScript SDK | Node.js (Phase 2) |
| Token counting | `chars / 4` provider-agnostic estimate |
| Storage | SQLite (bundled); Redis optional (Phase 2) |
| CI targets | x86_64/aarch64 Linux, macOS, Windows |
