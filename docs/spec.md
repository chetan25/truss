# Truss — Technical Specification

> The structural layer for agentic AI workflows.
> LLM-agnostic · MCP-native · Rust core · Python + TypeScript surfaces
> **Version:** 0.2 | **Status:** Pre-build (gaps resolved)

---

## Changelog from v0.1

| Gap | Fix |
|-----|-----|
| `BudgetWindow` enum undefined | Defined in Module 3 |
| `BudgetCarve` type undefined | Defined in Module 2 |
| `AlertConfig` struct undefined | Defined in Module 3 |
| `TrussError` hierarchy missing | New section added after Repo Structure |
| `AgentEnvelope.checkpoint_id` was `Uuid` (crashes first use) | Changed to `Option<Uuid>` |
| `AgentEnvelope.scope` was `ScopeManifest` (Phase 2 type) | Changed to `Vec<String>` for Phase 1 |
| `AgentEnvelope.budget` was `BudgetState` (undefined) | Changed to `budget_usd_remaining: f64` |
| Token counting used tiktoken-rs (OpenAI-only) | `estimate_tokens(text)` = `ceil(chars/4)`, provider-agnostic; `TokenCounter` trait for Phase 2 |
| `InMemoryStore` thread safety unaddressed | Spec now requires `Arc<RwLock<>>` |
| `s.report()` never defined | `SessionReport` struct + `report()` defined in Session section |
| `Fence` module had no Rust types | `FenceStore` trait + `InMemoryFence` fully defined |
| `score_relevance`/`detect_contradiction` needed NLP | Clarified as keyword/hash heuristics (no LLM call) |
| `Hybrid` strategy depended on `Summarise` (circular) | Phase 1 `Hybrid` = WeightedPrune → SlidingWindow only; `Summarise` deferred to Phase 2 |
| `CircuitBreaker` was a bare struct with public fields | Wrapped as `CircuitBreakerConfig`; `CircuitBreaker` is the runtime engine |

---

## The One-Line Pitch

Truss is the missing infrastructure layer between your LLM calls and your agentic
workflow — handling context bloat, agent handoffs, token budgets, multi-LLM routing,
MCP safety, and session accounting in one composable library, across every framework
and every model.

---

## Why This Exists

Every team building production agentic systems in 2026 is solving the same six
problems from scratch:

| Problem                           | How teams solve it today                         |
| --------------------------------- | ------------------------------------------------ |
| Context bloat                     | Manual truncation, ad-hoc summarisation          |
| Agent handoff loss                | Raw string dumps between agents                  |
| Faulty logic from missing context | Prompt engineering + prayer                      |
| MCP trust gap                     | Nothing — no standard exists                     |
| Token cost explosion              | Per-call `max_tokens`, monthly billing surprises |
| Multi-LLM routing                 | Custom if/else per team                          |

Truss solves all six with one install.

---

## Architecture Decision: Why Rust Core

The performance-critical modules (context compression, token counting, envelope
packing, MCP stream interception, ledger accounting) run as a Rust crate. Python
and TypeScript surfaces are thin bindings via PyO3 and napi-rs respectively —
the same pattern used by uv, Ruff, and Pydantic v2.

**What this gives you:**

- 7–10× faster context compression than Python equivalents
- True parallelism for concurrent MCP tool call interception (no GIL)
- Compile-time correctness on the AgentEnvelope type
- `pip install truss-ai` and `npm install truss-ai` both just work — no Rust
  toolchain required for users

**Build tool:** Maturin (Python wheels) + napi-rs (Node native addons)
**CI targets:** x86_64-linux, aarch64-linux, x86_64-macos, aarch64-macos, x86_64-windows

---

## Repo Structure

```
truss/
├── crates/
│   ├── truss-core/          # Pure Rust — all performance-critical logic
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── error.rs     # TrussError hierarchy
│   │   │   ├── types.rs     # AgentEnvelope + shared types
│   │   │   ├── context/     # block, surgeon, token_counter
│   │   │   ├── handoff/     # envelope (pack, unpack)
│   │   │   ├── budget/      # config, ledger, memory_store, sqlite_store, circuit_breaker
│   │   │   ├── coord/       # checkpoint, sqlite_checkpoint
│   │   │   └── fence/       # memory_fence
│   │   └── Cargo.toml
│   │
│   ├── truss-py/            # PyO3 bindings → pip install truss-ai
│   │   ├── src/lib.rs       # #[pymodule] exposing truss-core
│   │   ├── pyproject.toml   # maturin config
│   │   └── Cargo.toml
│   │
│   └── truss-node/          # napi-rs bindings → npm install truss-ai (Phase 2)
│       ├── src/lib.rs
│       ├── index.ts
│       ├── adapters/
│       └── package.json
│
├── python/truss/
│   ├── __init__.py
│   ├── session.py           # Session + SessionReport
│   ├── types.py             # Pydantic mirrors of Rust types
│   └── adapters/
│       ├── langchain.py
│       ├── crewai.py        # Phase 2
│       ├── autogen.py       # Phase 2
│       └── pydantic_ai.py   # Phase 2
│
├── examples/
│   ├── hermes-agent/        # Phase 1 reference impl
│   ├── crewai-research/     # Phase 2
│   ├── langchain-rag/       # Phase 2
│   └── raw-openai/          # Phase 2
│
├── benches/
│   └── compression.rs       # Criterion benchmarks
├── tests/
│   ├── integration/
│   └── fixtures/
├── Cargo.toml               # workspace
└── pytest.ini
```

---

## Error Hierarchy — `crates/truss-core/src/error.rs`

All Truss errors flow through one enum. Never return raw `String` errors across
module boundaries.

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrussError {
    #[error("budget exceeded: {0}")]
    BudgetExceeded(String),

    #[error("tool out of scope: {tool} denied by manifest")]
    ToolOutOfScope { tool: String },

    #[error("checkpoint not found: {id}")]
    CheckpointNotFound { id: String },

    #[error("fence lock conflict: {key} held by {owner}")]
    FenceLockConflict { key: String, owner: String },

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("storage error: {0}")]
    Storage(String),
}

pub type Result<T> = std::result::Result<T, TrussError>;
```

---

## Module 1: Context Surgeon — `crates/truss-core/src/context/`

### The problem it solves

Context grows with every agent turn. By message 20 in a long session, the model
is spending most of its attention budget on stale, low-signal content — and you're
paying for every token re-sent. The surgeon removes the noise without losing the
signal, and does it fast enough to run on every message without adding latency.

### Token counting

Phase 1 uses a provider-agnostic approximation: `ceil(char_count / 4)`. This is
accurate to within ~10% for English text across all major LLMs. A pluggable
`TokenCounter` trait is reserved for Phase 2 when provider-specific accuracy matters.

```rust
/// Provider-agnostic token estimator. Accurate to ~10% for English.
pub fn estimate_tokens(text: &str) -> u32 {
    ((text.len() as f32) / 4.0).ceil() as u32
}
```

### Core types (Rust)

```rust
pub enum ContextRole {
    Task,        // what the agent is being asked to do
    Constraint,  // hard limits (scope, budget, safety)
    Finding,     // discovered facts from tool calls
    Decision,    // what was decided and why
    Warning,     // flags raised during execution
    Background,  // low-priority supporting info
}

pub enum ContextWeight {
    Critical = 3,   // always preserved verbatim
    High = 2,       // preserved unless extreme compression needed
    Normal = 1,     // summarised when over budget
    Background = 0, // first to drop
}

pub struct ContextBlock {
    pub id: Uuid,
    pub role: ContextRole,
    pub weight: ContextWeight,
    pub content: String,
    pub source: String,             // which agent/tool produced this
    pub token_count: u32,           // set via estimate_tokens() at insertion
    pub created_at: u64,            // unix ms (set by caller)
    pub referenced_by: Vec<Uuid>,   // which blocks cite this one
}

/// Phase 1: SlidingWindow and WeightedPrune only.
/// SemanticDedup and Summarise are Phase 2.
/// Hybrid in Phase 1 = WeightedPrune first, then SlidingWindow if still over budget.
pub enum CompressionStrategy {
    SlidingWindow { keep_recent: usize },
    WeightedPrune { target_tokens: u32 },
    Hybrid,  // Phase 1: WeightedPrune → SlidingWindow
             // Phase 2 adds: → SemanticDedup → Summarise
}

pub struct SurgeonConfig {
    pub strategy: CompressionStrategy,
    pub target_tokens: u32,
    pub preserve_recent: usize,   // always keep last N blocks regardless of weight
}

impl Default for SurgeonConfig {
    fn default() -> Self {
        Self {
            strategy: CompressionStrategy::Hybrid,
            target_tokens: 8_000,
            preserve_recent: 5,
        }
    }
}

pub struct SurgeonResult {
    pub blocks: Vec<ContextBlock>,
    pub tokens_before: u32,
    pub tokens_after: u32,
    pub tokens_saved: u32,
    pub strategy_applied: String,
}
```

### Public API

```rust
// Core compression
pub fn compress(blocks: &[ContextBlock], config: &SurgeonConfig) -> SurgeonResult;

// Heuristic helpers (keyword/hash-based, no LLM call)
// score_relevance: word-overlap ratio between block content and task string (0.0–1.0)
pub fn score_relevance(block: &ContextBlock, task: &str) -> f32;
// detect_contradiction: returns true when one block contains "not X" while the other has "X"
pub fn detect_contradiction(a: &ContextBlock, b: &ContextBlock) -> bool;
```

```python
# Python (via PyO3)
from truss import compress_context, SurgeonConfig

result = compress_context(
    blocks=my_blocks,
    config=SurgeonConfig(
        strategy="hybrid",
        target_tokens=8_000,
        preserve_recent=5,
    )
)
print(f"Saved {result.tokens_saved} tokens")
```

### Framework adapter: LangChain

```python
from truss.adapters.langchain import TrussMemory

memory = TrussMemory(
    strategy="hybrid",
    target_tokens=8_000,
    preserve_recent=5,
)
# Drop-in for ConversationBufferMemory — same interface, Rust performance
chain = ConversationalRetrievalChain.from_llm(llm, memory=memory)
```

---

## Module 2: Agent Handoff — `crates/truss-core/src/handoff/`

### The problem it solves

When Agent A spawns Agent B, it currently dumps a raw string or full chat history.
The receiving agent either drowns in noise or loses the reasoning chain. Truss
introduces a typed `AgentEnvelope` — a structured handoff format that travels
between agents with provenance intact, framework-agnostic.

### Core types (Rust)

```rust
pub struct AgentEnvelope {
    pub id: Uuid,
    pub task: String,
    pub context: Vec<ContextBlock>,         // compressed, weighted
    pub evidence: Vec<EvidenceRef>,         // sources decisions cite
    pub decisions: Vec<DecisionRecord>,     // what was decided + why
    pub scope: Vec<String>,                 // allowed tool names (Phase 1)
                                            // Phase 2: replaced by ScopeManifest
    pub budget_usd_remaining: f64,          // Phase 1: plain f64
                                            // Phase 2: replaced by BudgetState
    pub checkpoint_id: Option<Uuid>,        // None until first checkpoint is saved
    pub model_hint: ModelTier,
    pub parent_agent: Option<String>,       // for audit trail
    pub created_at: u64,
}

pub struct EvidenceRef {
    pub id: Uuid,
    pub content: String,
    pub source_url: Option<String>,
    pub tool_name: Option<String>,
    pub confidence: f32,  // 0.0–1.0
}

pub struct DecisionRecord {
    pub id: Uuid,
    pub decision: String,
    pub reasoning: String,
    pub evidence_ids: Vec<Uuid>,
    pub confidence: f32,
    pub decided_by: String,
    pub timestamp: u64,
}

pub enum ModelTier {
    Cheap,    // Haiku, Flash-Lite, GPT-4o-mini
    Standard, // Sonnet, Flash, GPT-4o
    Premium,  // Opus, Pro, o1
    Auto,     // Truss router decides based on task complexity
}

/// How much of a parent budget to carve for a child agent.
pub enum BudgetCarve {
    FixedUsd(f64),       // exact dollar amount
    FixedTokens(u64),    // exact token count (converted at caller's rate)
    Percent(f32),        // 0.0–1.0 fraction of parent's remaining budget
}
```

### Packing and unpacking

```rust
// Pack: create a child envelope filtered to what the sub-agent needs.
// carved budget is clamped to parent's remaining budget.
pub fn pack(
    parent: &AgentEnvelope,
    task: impl Into<String>,
    carry_weights: &[ContextWeight],  // e.g. [Critical, High]
    budget_carve: BudgetCarve,
) -> AgentEnvelope;

// Unpack: expand envelope into a flat list of context blocks for direct use.
pub fn unpack(envelope: &AgentEnvelope) -> Vec<ContextBlock>;
```

```python
# Python
from truss import pack_handoff, unpack_handoff, ContextWeight, BudgetCarve

child_envelope = pack_handoff(
    parent=current_envelope,
    task="Find pricing for Stripe vs Paddle",
    carry=[ContextWeight.Critical, ContextWeight.High],
    budget_carve=BudgetCarve.fixed_usd(0.20),
)

context_blocks = unpack_handoff(child_envelope)
```

### Framework adapter: CrewAI (Phase 2)

```python
from truss.adapters.crewai import TrussHandoff

researcher = Agent(
    role="Researcher",
    goal="Find vendor pricing",
    backstory="...",
    tools=[search_tool],
    callbacks=[TrussHandoff(carry=["critical", "high"])]
)
```

---

## Module 3: Token Budget + Session Ledger — `crates/truss-core/src/budget/`

### The problem it solves

Two related problems: runaway agent costs (a looping agent burning $200 overnight),
and per-user cost attribution for SaaS products. One module solves both.

### Core types (Rust)

```rust
/// Time window for budget enforcement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BudgetWindow {
    PerSession,
    PerHour,
    PerDay,
    PerMonth,
}

pub struct BudgetLimit {
    pub tokens: Option<u64>,
    pub usd: Option<f64>,
    pub window: BudgetWindow,
}

/// Where to send budget alerts. log_to_stderr is always-available fallback.
pub struct AlertConfig {
    pub slack_webhook: Option<String>,
    pub log_to_stderr: bool,  // default true
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self { slack_webhook: None, log_to_stderr: true }
    }
}

pub struct BudgetConfig {
    pub per_session: Option<BudgetLimit>,
    pub per_user: Option<BudgetLimit>,
    pub per_agent: Option<BudgetLimit>,
    pub global: Option<BudgetLimit>,
    pub on_exceeded: ExceededAction,
    pub alert_at_pct: f32,   // default 0.8 (80%)
    pub alerts: AlertConfig,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            per_session: None,
            per_user: None,
            per_agent: None,
            global: None,
            on_exceeded: ExceededAction::Block,
            alert_at_pct: 0.8,
            alerts: AlertConfig::default(),
        }
    }
}

pub enum ExceededAction {
    Block,
    Fallback(ModelTier),
    Queue,
    GracefulResponse(String),
}

pub struct LedgerEntry {
    pub id: Uuid,
    pub session_id: String,
    pub user_id: Option<String>,
    pub agent_name: Option<String>,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_usd: f64,
    pub timestamp: u64,
    pub tags: HashMap<String, String>,
}

pub struct UsageReport {
    pub key: String,
    pub total_tokens: u64,
    pub total_cost_usd: f64,
    pub remaining_tokens: Option<u64>,
    pub remaining_usd: Option<f64>,
    pub pct_used: f32,
    pub window: BudgetWindow,
}

/// Normalised usage record — adapters translate provider-specific responses into this.
pub struct UsageRecord {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub model: String,
    pub cost_usd: Option<f64>,  // None = caller computes from tokens
}
```

### Storage backends

`LedgerStore` methods are **synchronous** in Phase 1 (SQLite + in-memory are fast
enough). Async wrappers are added in Phase 2 when Redis is introduced.

```rust
pub trait LedgerStore: Send + Sync {
    fn record(&self, entry: LedgerEntry) -> Result<()>;
    fn usage(&self, key: &str, window: &BudgetWindow) -> Result<UsageReport>;
    fn flush(&self) -> Result<()>;
}

/// Thread-safe via Arc<RwLock<Vec<LedgerEntry>>>.
pub struct InMemoryStore { /* ... */ }

/// File-backed. Thread-safe via Mutex<Connection>.
pub struct SqliteLedgerStore { /* ... */ }
```

### Circuit breaker — `crates/truss-core/src/budget/circuit_breaker.rs`

`CircuitBreakerConfig` holds parameters; `CircuitBreaker` is the runtime engine
that maintains the rolling window and retry counter.

Repeated-prompt detection uses FNV-1a 64-bit hash comparison — no LLM call, <1µs.

```rust
pub struct CircuitBreakerConfig {
    pub max_requests_per_minute: u32,  // default 60
    pub max_cost_velocity_usd: f64,    // default $1.00/min
    pub max_retry_depth: u8,           // default 3
    pub trip_on_repeated_prompt: bool, // FNV hash of last 3 prompts
}

impl Default for CircuitBreakerConfig { /* ... */ }

pub enum CircuitTrip {
    RateLimit,
    CostVelocity { usd_per_minute: f64 },
    MaxRetryDepth,
    RepeatedPrompt,
}

pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    // rolling window: Arc<Mutex<VecDeque<RequestRecord>>>
    // retry counter: Arc<Mutex<u8>>
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self;
    /// Returns Some(trip) if breaker should fire, None if request is allowed.
    pub fn check_and_record(&self, prompt: &str, cost_usd: f64, now_ms: u64) -> Option<CircuitTrip>;
    pub fn increment_retry(&self) -> Option<CircuitTrip>;
    pub fn reset_retry(&self);
}
```

### Public API (Python)

```python
from truss import Budget, BudgetConfig, ExceededAction
import os

budget = Budget(BudgetConfig(
    per_user=BudgetLimit(usd=5.00, window=BudgetWindow.PerMonth),
    per_session=BudgetLimit(tokens=100_000, window=BudgetWindow.PerSession),
    on_exceeded=ExceededAction.Fallback(ModelTier.Cheap),
    alert_at_pct=0.8,
    alerts=AlertConfig(slack_webhook=os.environ.get("SLACK_WEBHOOK")),
))

async def call_llm(prompt: str, user_id: str) -> str:
    async with budget.session(user_id=user_id) as session:
        response = await your_llm.complete(prompt)
        session.record(UsageRecord(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        ))
        return response.text

report = await budget.usage(user_id="user_123", window=BudgetWindow.PerMonth)
print(f"Used ${report.total_cost_usd:.2f} of $5.00 this month")
```

---

## Module 4: Multi-LLM Router — `crates/truss-core/src/router/` (Phase 2)

### Core types (Rust)

```rust
pub struct RouterConfig {
    pub rules: Vec<RoutingRule>,
    pub fallback_chain: Vec<ModelSpec>,
    pub classify: ClassifyMode,
    pub latency_budget_ms: Option<u32>,
}

pub struct RoutingRule {
    pub condition: RoutingCondition,
    pub target: ModelSpec,
    pub max_cost_usd: Option<f64>,
}

pub enum RoutingCondition {
    ComplexityBelow(f32),
    ComplexityAbove(f32),
    TokensBudgetAbove(u32),
    TaskContains(Vec<String>),
    LatencyRequired,
    Always,
}

pub struct ModelSpec {
    pub provider: Provider,   // Anthropic | OpenAI | Google | Ollama | Custom
    pub model_id: String,
    pub tier: ModelTier,
    pub api_key_env: String,
}

pub enum ClassifyMode {
    Disabled,
    Keyword,
    LightweightModel(ModelSpec),
    Embedding(EmbedConfig),  // Phase 3
}

pub trait ComplexityClassifier: Send + Sync {
    async fn classify(&self, task: &str, context: &[ContextBlock]) -> f32;
}
```

---

## Module 5: MCP Interceptor — `crates/truss-core/src/mcp/` (Phase 2)

### Scope manifest format

```json
{
  "$schema": "https://truss.dev/schema/scope/v1.json",
  "tools": ["readFile", "searchWeb", "sendSlack"],
  "toolConfig": {
    "readFile": {
      "allowPaths": ["/home/project/**"],
      "denyPaths": ["/etc/**", "**/.env"]
    }
  },
  "mcpServers": {
    "filesystem": { "trust": "verified" },
    "github": { "trust": "community" }
  },
  "agentInheritance": "subset"
}
```

### Public API (Phase 2)

```python
from truss import MCPInterceptor, ScopeManifest

interceptor = MCPInterceptor(
    scope=ScopeManifest.from_file("scope.json"),
    audit_path="./truss-audit",
    # trust_registry is optional; defaults to offline-only validation
    trust_registry=None,
)

safe_mcp = interceptor.wrap(mcp_client)
result = await safe_mcp.call_tool("readFile", {"path": "/etc/passwd"})
# → TrussError: TOOL_OUT_OF_SCOPE — path denied by manifest
```

---

## Module 6: Checkpoints + Rollback — `crates/truss-core/src/coord/`

### Core types (Rust)

```rust
pub struct CheckpointMeta {
    pub id: Uuid,
    pub session_id: String,
    pub agent_name: String,
    pub description: String,
    pub created_at: u64,
}

pub struct Checkpoint {
    pub id: Uuid,
    pub session_id: String,
    pub agent_name: String,
    pub envelope_snapshot: AgentEnvelope,
    pub external_state: HashMap<String, Vec<u8>>,  // serialised tool outputs
    pub created_at: u64,
    pub description: String,
}

pub trait CheckpointStore: Send + Sync {
    fn save(&self, cp: Checkpoint) -> Result<Uuid>;
    fn load(&self, id: Uuid) -> Result<Checkpoint>;
    fn rollback(&self, id: Uuid) -> Result<AgentEnvelope>;
    fn list(&self, session_id: &str) -> Result<Vec<CheckpointMeta>>;
}

pub struct InMemoryCheckpointStore { /* Arc<RwLock<HashMap<Uuid, Checkpoint>>> */ }
```

```python
# Python
from truss import Session

async with Session(budget_usd=1.00, checkpoints=True) as session:
    cp1 = await session.checkpoint("after-planning")
    plan = await planner.run(session.envelope)

    try:
        research = await researcher.run(session.handoff(plan))
    except TrussError:
        session = await session.rollback(cp1)
        research = await researcher_v2.run(session.handoff(plan))
```

---

## Module 7: Fence — `crates/truss-core/src/fence/`

### The problem it solves

In multi-agent systems, two agents writing to the same state without coordination
creates race conditions. Truss provides a typed lock store with TTL-based expiry.

### Core types (Rust)

```rust
#[derive(Debug, Clone)]
pub struct LockHandle {
    pub key: String,
    pub owner: String,
    pub acquired_at_ms: u64,
    pub ttl_ms: u64,
}

impl LockHandle {
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms > self.acquired_at_ms + self.ttl_ms
    }
}

pub trait FenceStore: Send + Sync {
    /// Acquire lock. Returns FenceLockConflict if held by another non-expired owner.
    fn acquire(&self, key: &str, owner: &str, ttl_ms: u64, now_ms: u64) -> Result<()>;
    /// Release lock. No-op if key is not held by `owner`.
    fn release(&self, key: &str, owner: &str) -> Result<()>;
    /// True if the key is currently locked and not expired.
    fn is_locked(&self, key: &str, now_ms: u64) -> bool;
}

/// Phase 1 implementation: in-process, Arc<Mutex<HashMap>>.
pub struct InMemoryFence { /* ... */ }

// Phase 2: RedisFence using SET NX PX
```

### Public API (Python)

```python
from truss import Fence

fence = Fence()  # InMemoryFence by default; Phase 2 adds Fence(store=redis_store)

async with fence.lock("shared-doc-v3", owner="agent-writer-1", ttl_ms=30_000):
    doc = await read_doc()
    modified = transform(doc)
    await write_doc(modified)
# Lock released on __aexit__
```

---

## The `Session` Object — Unified Entry Point

All modules are accessible through a single `Session` object.

### `SessionReport` — returned by `session.report()`

```python
@dataclass
class SessionReport:
    tokens_before: int        # total tokens across all compress() calls (before)
    tokens_after: int         # total tokens after compression
    tokens_saved: int         # tokens_before - tokens_after
    cost_saved_usd: float     # estimated savings at ~$0.001/1k tokens
    budget_used_usd: float    # total recorded via record_usage()
    budget_limit_usd: float | None

    def __str__(self) -> str:
        # Example output:
        # Context: 12,400 → 4,200 tokens (saved 8,200)
        # Est. savings: $0.0082
        # Budget: $0.18 of $1.00 used
```

### Python

```python
from truss import Session, SessionConfig, BudgetConfig

# Minimal config (sane defaults everywhere)
session = Session()

# Full config
session = Session(
    budget_usd=1.00,
    target_tokens=8_000,
    compress_strategy="hybrid",
    preserve_recent=5,
)

async with session as s:
    # Compress context before each LLM call
    result = s.compress(blocks)

    # Record token usage after each LLM call
    s.record_usage(input_tokens=1200, output_tokens=400, model="claude-haiku")

    # Print summary
    print(s.report())
    # Context: 12,400 → 4,200 tokens (saved 8,200)
    # Est. savings: $0.0082
    # Budget: $0.18 of $1.00 used
```

### TypeScript (Phase 2)

```typescript
import { Session } from "truss-ai";

const session = new Session({
  budget: { perSession: { usd: 1.0 }, onExceeded: "fallback" },
  context: { targetTokens: 8_000, strategy: "hybrid" },
  checkpoints: true,
});

const envelope = await session.start({ task: "...", scope: ["searchWeb"] });
```

---

## Framework Adapters — Full Matrix

| Framework     | Language   | Adapter             | Phase | What it does                                     |
| ------------- | ---------- | ------------------- | ----- | ------------------------------------------------ |
| LangChain     | Python     | `TrussMemory`       | 1     | Drop-in memory with context surgery              |
| LangGraph     | Python     | `TrussNode`         | 2     | Node wrapper that checkpoints at each graph step |
| CrewAI        | Python     | `TrussHandoff`      | 2     | Packs/unpacks envelopes at task boundaries       |
| AutoGen       | Python     | `TrussMiddleware`   | 2     | Intercepts agent-to-agent messages               |
| Pydantic AI   | Python     | `TrussRunHooks`     | 2     | Budget + audit hooks                             |
| Vercel AI SDK | TypeScript | `trussMiddleware()` | 2     | Streaming budget + router                        |
| Mastra        | TypeScript | `TrussStep`         | 2     | Workflow step wrapper                            |
| Claude Code   | TypeScript | MCP server          | 3     | Exposes truss as an MCP tool                     |
| Raw API       | Both       | `Session`           | 1     | Works directly, no framework needed              |

---

## Tech Stack Summary

| Layer           | Technology                  | Rationale                                              |
| --------------- | --------------------------- | ------------------------------------------------------ |
| Core engine     | Rust 2021 edition           | Performance, safety, no GIL                            |
| Async runtime   | Tokio                       | Production-grade, dominant in Rust async               |
| Python bindings | PyO3 + Maturin              | Same toolchain as uv, Ruff, Pydantic v2                |
| TS/JS bindings  | napi-rs                     | Same pattern as SWC, Biome (Phase 2)                   |
| Token counting  | `estimate_tokens()` chars/4 | Provider-agnostic; pluggable `TokenCounter` in Phase 2 |
| Serialisation   | serde + serde_json          | Standard Rust, zero-copy where possible                |
| Storage         | SQLite (rusqlite bundled)   | Default embedded; Redis optional feature (Phase 2)     |
| Error handling  | thiserror                   | Derive macro, integrates cleanly with PyO3             |
| Build / CI      | GitHub Actions + cargo-dist | Cross-platform wheel builds                            |
| Python package  | `truss-ai` on PyPI          | Avoids collision with `truss` (ML deploy tool)         |
| Node package    | `truss-ai` on npm           | Same name, different registry                          |
| Rust crate      | `truss-core` on crates.io   | Core only; consumers use language bindings             |

---

## Build Phases

### Phase 1 — Rust core + Python surface (Weeks 1–8)

- [ ] `truss-core` scaffolded with Cargo workspace
- [ ] `TrussError` hierarchy (`thiserror`)
- [ ] `ContextBlock` + `AgentEnvelope` types with serde
- [ ] Context surgeon: `SlidingWindow` + `WeightedPrune` strategies
- [ ] `estimate_tokens()` — chars/4 provider-agnostic token counter
- [ ] Budget config types: `BudgetWindow`, `BudgetLimit`, `AlertConfig`
- [ ] Budget ledger: `InMemoryStore` (Arc<RwLock>) + `SqliteLedgerStore`
- [ ] Circuit breaker: `CircuitBreakerConfig` + `CircuitBreaker` runtime
- [ ] Agent handoff: `BudgetCarve`, `pack()`, `unpack()`
- [ ] Checkpoint: `CheckpointStore` trait + `InMemoryCheckpointStore`
- [ ] Fence: `FenceStore` trait + `InMemoryFence` with TTL
- [ ] PyO3 bindings for all of the above
- [ ] Maturin build: wheels for Linux, Mac, Windows
- [ ] `pip install truss-ai` works
- [ ] Python `Session` + `SessionReport` (`report()` method)
- [ ] LangChain `TrussMemory` adapter
- [ ] `examples/hermes-agent/` — first reference impl
- [ ] Criterion benchmarks: compress vs pure Python baseline
- [ ] Integration test suite (`pytest`)

### Phase 2 — Multi-LLM router + MCP + TS surface (Weeks 9–16)

- [ ] Multi-LLM router with keyword classifier
- [ ] Provider adapters: Anthropic, OpenAI, Google, Ollama
- [ ] MCP interceptor + `ScopeManifest` struct
- [ ] napi-rs bindings for all modules
- [ ] `npm install truss-ai` works
- [ ] Vercel AI SDK adapter
- [ ] Mastra adapter
- [ ] CrewAI + AutoGen adapters
- [ ] Semantic dedup compression strategy (embedding-based)
- [ ] Summarise compression strategy (LLM call — tracked against budget)
- [ ] `TokenCounter` trait + provider-specific implementations
- [ ] Redis store for ledger + checkpoint
- [ ] `examples/crewai-research/` and `examples/langchain-rag/`

### Phase 3 — Ecosystem + distribution (Weeks 17–24)

- [ ] Claude Code MCP server (truss as a tool)
- [ ] Trust registry for MCP servers
- [ ] Embedding-based complexity classifier
- [ ] cargo-dist for binary releases
- [ ] Docs site (mdBook or Docusaurus)
- [ ] `0.1.0` public release

---

## Naming + Package Identity

| Surface   | Package name         | Import                               |
| --------- | -------------------- | ------------------------------------ |
| PyPI      | `truss-ai`           | `import truss`                       |
| npm       | `truss-ai`           | `import { Session } from 'truss-ai'` |
| crates.io | `truss-core`         | `use truss_core::Session;`           |
| Docs      | `truss.dev` (target) | —                                    |
| GitHub    | `your-org/truss`     | —                                    |

Tagline: **The structural layer for agentic AI workflows.**

---

## Connection to ShieldMCP

Truss and ShieldMCP are separate products that share a Rust core:

| Module          | Truss                             | ShieldMCP                     |
| --------------- | --------------------------------- | ----------------------------- |
| MCP interceptor | `truss-core::mcp::interceptor`    | Re-exported, extended         |
| Scope manifest  | Same type, same parser            | Same                          |
| Audit log       | `truss-core::budget::LedgerEntry` | Extended with security fields |
| Token budget    | Full module                       | Subset (safety budget caps)   |

---

## What a Developer Does in 10 Minutes

```bash
pip install truss-ai
```

```python
from truss import Session, ContextBlock, ContextRole, ContextWeight

async with Session(budget_usd=1.00) as s:
    # Build context blocks from your agent's messages
    blocks = [
        ContextBlock(ContextRole.Task, ContextWeight.Critical, "your task here", "user"),
        # ... more blocks
    ]

    # Compress before sending to LLM
    result = s.compress(blocks)

    # Call your LLM, record usage
    response = await your_llm.complete(result.blocks)
    s.record_usage(response.usage.input_tokens, response.usage.output_tokens)

    print(s.report())
    # Context: 12,400 → 4,200 tokens (saved 8,200)
    # Est. savings: $0.0082
    # Budget: $0.18 of $1.00 used
```

---

_This spec is the source of truth for Phase 1 implementation._
_Rust core first. Python surface second. TypeScript surface third._
_Never let the surface dictate the core design._
