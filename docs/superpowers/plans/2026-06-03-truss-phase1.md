# Truss Phase 1 — Rust Core + Python Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Rust core (`truss-core`) with all Phase 1 modules and a working `pip install truss-ai` Python surface including a LangChain adapter and a Hermes reference example.

**Architecture:** Cargo workspace with two crates — `truss-core` (pure Rust, no Python) and `truss-py` (PyO3 + Maturin bindings). All performance-critical logic lives in `truss-core`; `truss-py` is a thin `#[pymodule]` wrapper. Python-only adapter code lives in `truss-py/python/truss/`.

**Tech Stack:** Rust 2024 edition · Tokio async · PyO3 0.21 · Maturin 1.x · rusqlite · serde/serde_json · Criterion (benchmarks) · pytest (Python tests) · Python 3.10+

---

## Spec Gaps Resolved in This Plan

These gaps were found in `docs/spec.md` and are resolved here — the spec is the bug, this plan is the fix:

| Gap | Resolution |
|-----|-----------|
| `BudgetWindow` undefined | Defined in Task 4 as a Rust enum |
| `BudgetCarve` undefined | Defined in Task 5 as a struct |
| `AlertConfig` undefined | Defined in Task 4 as a struct with optional Slack webhook |
| `TrussError` undefined | Defined in Task 1 as the root error enum |
| `AgentEnvelope.checkpoint_id` non-optional | Changed to `Option<Uuid>` in Task 2 |
| Token counting is OpenAI-only | `TokenCounter` trait with chars/4 default impl in Task 3 |
| `InMemoryStore` thread safety | Uses `Arc<RwLock<>>` in Task 4 |
| `s.report()` never defined | Defined on `Session` in Task 10 |
| `Fence` underspecified | Minimal in-memory mutex Fence with typed trait in Task 9 |
| `score_relevance`/`detect_contradiction` unimplemented | Keyword/heuristic impl, not NLP, in Task 3 |
| Summarise circular budget dep | Deferred to Phase 2; Phase 1 Hybrid = WeightedPrune → SlidingWindow only |

---

## File Structure

```
truss/
├── Cargo.toml                          # workspace root
├── crates/
│   ├── truss-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # re-exports, crate root
│   │       ├── error.rs                # TrussError hierarchy (Task 1)
│   │       ├── types.rs                # AgentEnvelope + shared types (Task 2)
│   │       ├── context/
│   │       │   ├── mod.rs              # re-exports
│   │       │   ├── block.rs            # ContextBlock, ContextRole, ContextWeight
│   │       │   ├── surgeon.rs          # compress(), SurgeonConfig, SurgeonResult (Task 3)
│   │       │   └── token_counter.rs    # TokenCounter trait + char-div-4 impl (Task 3)
│   │       ├── handoff/
│   │       │   ├── mod.rs
│   │       │   └── envelope.rs         # pack(), unpack(), BudgetCarve (Task 5)
│   │       ├── budget/
│   │       │   ├── mod.rs
│   │       │   ├── config.rs           # BudgetConfig, BudgetLimit, BudgetWindow (Task 4)
│   │       │   ├── ledger.rs           # LedgerEntry, LedgerStore trait (Task 4)
│   │       │   ├── memory_store.rs     # InMemoryStore (Task 4)
│   │       │   ├── sqlite_store.rs     # SqliteLedgerStore (Task 6)
│   │       │   └── circuit_breaker.rs  # CircuitBreaker (Task 7)
│   │       ├── coord/
│   │       │   ├── mod.rs
│   │       │   ├── checkpoint.rs       # Checkpoint, CheckpointStore trait (Task 8)
│   │       │   └── sqlite_checkpoint.rs # SqliteCheckpointStore (Task 8)
│   │       └── fence/
│   │           ├── mod.rs
│   │           └── memory_fence.rs     # FenceStore trait + InMemoryFence (Task 9)
│   │
│   └── truss-py/
│       ├── Cargo.toml
│       ├── pyproject.toml              # maturin config
│       └── src/
│           ├── lib.rs                  # #[pymodule] root (Task 11)
│           ├── context_bindings.rs     # compress_context, SurgeonConfig py types
│           ├── handoff_bindings.rs     # pack_handoff, unpack_handoff
│           ├── budget_bindings.rs      # Budget, BudgetConfig, LedgerEntry
│           └── session_bindings.rs     # Session object (Task 11)
│
├── python/truss/
│   ├── __init__.py                     # public re-exports (Task 11)
│   ├── types.py                        # Pydantic mirrors (Task 11)
│   ├── session.py                      # Session.report() + Python-side helpers (Task 11)
│   └── adapters/
│       └── langchain.py                # TrussMemory adapter (Task 12)
│
├── examples/
│   └── hermes-agent/
│       └── main.py                     # Reference implementation (Task 13)
│
├── benches/
│   └── compression.rs                  # Criterion bench: compress vs pure Python (Task 14)
│
└── tests/
    ├── integration/
    │   ├── test_compress.py
    │   ├── test_budget.py
    │   ├── test_handoff.py
    │   ├── test_checkpoint.py
    │   └── test_fence.py
    └── fixtures/
        └── sample_blocks.py
```

---

## Task 1: Workspace + Error Hierarchy

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/truss-core/Cargo.toml`
- Create: `crates/truss-core/src/lib.rs`
- Create: `crates/truss-core/src/error.rs`

- [ ] **Step 1: Initialise the Cargo workspace**

```bash
mkdir -p crates/truss-core/src crates/truss-py/src
```

Write `Cargo.toml` (workspace root):

```toml
[workspace]
members = ["crates/truss-core", "crates/truss-py"]
resolver = "2"

[workspace.dependencies]
uuid = { version = "1.10", features = ["v4", "serde"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
thiserror = "2"
rusqlite = { version = "0.32", features = ["bundled"] }
```

Write `crates/truss-core/Cargo.toml`:

```toml
[package]
name = "truss-core"
version = "0.1.0"
edition = "2021"

[dependencies]
uuid.workspace = true
serde.workspace = true
serde_json.workspace = true
tokio.workspace = true
thiserror.workspace = true
rusqlite.workspace = true
```

- [ ] **Step 2: Write the failing error test**

Create `crates/truss-core/src/error.rs`:

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
```

Create `crates/truss-core/src/lib.rs`:

```rust
pub mod error;

pub use error::TrussError;
pub type Result<T> = std::result::Result<T, TrussError>;
```

Write the test (add to `crates/truss-core/src/error.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_exceeded_formats_message() {
        let e = TrussError::BudgetExceeded("user_123 hit $5.00 limit".to_string());
        assert!(e.to_string().contains("budget exceeded"));
    }

    #[test]
    fn tool_out_of_scope_includes_tool_name() {
        let e = TrussError::ToolOutOfScope { tool: "readFile".to_string() };
        assert!(e.to_string().contains("readFile"));
    }
}
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
cargo test -p truss-core
```

Expected output: `test result: ok. 2 passed`

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml crates/truss-core/
git commit -m "feat: initialise truss-core workspace with error hierarchy"
```

---

## Task 2: Core Types — `ContextBlock` and `AgentEnvelope`

**Files:**
- Create: `crates/truss-core/src/context/block.rs`
- Create: `crates/truss-core/src/context/mod.rs`
- Create: `crates/truss-core/src/types.rs`
- Modify: `crates/truss-core/src/lib.rs`

- [ ] **Step 1: Write failing type tests**

Create `crates/truss-core/src/context/block.rs`:

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContextRole {
    Task,
    Constraint,
    Finding,
    Decision,
    Warning,
    Background,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum ContextWeight {
    Critical = 3,
    High = 2,
    Normal = 1,
    Background = 0,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBlock {
    pub id: Uuid,
    pub role: ContextRole,
    pub weight: ContextWeight,
    pub content: String,
    pub source: String,
    pub token_count: u32,
    pub created_at: u64,
    pub referenced_by: Vec<Uuid>,
}

impl ContextBlock {
    pub fn new(
        role: ContextRole,
        weight: ContextWeight,
        content: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        let content = content.into();
        let token_count = estimate_tokens(&content);
        Self {
            id: Uuid::new_v4(),
            role,
            weight,
            token_count,
            content,
            source: source.into(),
            created_at: 0, // set by caller — no std::time in tests
            referenced_by: vec![],
        }
    }
}

pub fn estimate_tokens(text: &str) -> u32 {
    ((text.len() as f32) / 4.0).ceil() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_block_estimates_tokens() {
        let block = ContextBlock::new(
            ContextRole::Task,
            ContextWeight::Critical,
            "Hello world",   // 11 chars → ceil(11/4) = 3
            "test",
        );
        assert_eq!(block.token_count, 3);
    }

    #[test]
    fn block_serialises_to_json() {
        let block = ContextBlock::new(ContextRole::Finding, ContextWeight::Normal, "data", "agent-1");
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"Finding\""));
    }
}
```

Create `crates/truss-core/src/context/mod.rs`:

```rust
pub mod block;
pub use block::{ContextBlock, ContextRole, ContextWeight, estimate_tokens};
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cargo test -p truss-core context
```

Expected: `test result: ok. 2 passed`

- [ ] **Step 3: Write AgentEnvelope type**

Create `crates/truss-core/src/types.rs`:

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::context::block::ContextBlock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelTier {
    Cheap,
    Standard,
    Premium,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub id: Uuid,
    pub content: String,
    pub source_url: Option<String>,
    pub tool_name: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub id: Uuid,
    pub decision: String,
    pub reasoning: String,
    pub evidence_ids: Vec<Uuid>,
    pub confidence: f32,
    pub decided_by: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEnvelope {
    pub id: Uuid,
    pub task: String,
    pub context: Vec<ContextBlock>,
    pub evidence: Vec<EvidenceRef>,
    pub decisions: Vec<DecisionRecord>,
    pub budget_usd_remaining: f64,
    pub checkpoint_id: Option<Uuid>,   // spec gap fix: Option<Uuid> not Uuid
    pub model_hint: ModelTier,
    pub parent_agent: Option<String>,
    pub created_at: u64,
}

impl AgentEnvelope {
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            task: task.into(),
            context: vec![],
            evidence: vec![],
            decisions: vec![],
            budget_usd_remaining: f64::MAX,
            checkpoint_id: None,
            model_hint: ModelTier::Auto,
            parent_agent: None,
            created_at: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_envelope_has_no_checkpoint() {
        let env = AgentEnvelope::new("test task");
        assert!(env.checkpoint_id.is_none());
    }

    #[test]
    fn envelope_round_trips_json() {
        let env = AgentEnvelope::new("analyse pricing");
        let json = serde_json::to_string(&env).unwrap();
        let back: AgentEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(env.task, back.task);
    }
}
```

- [ ] **Step 4: Wire up lib.rs**

Replace `crates/truss-core/src/lib.rs`:

```rust
pub mod context;
pub mod error;
pub mod types;

pub use error::{TrussError, Result};
pub use types::{AgentEnvelope, ModelTier, EvidenceRef, DecisionRecord};
pub use context::{ContextBlock, ContextRole, ContextWeight, estimate_tokens};
```

- [ ] **Step 5: Run all tests**

```bash
cargo test -p truss-core
```

Expected: `test result: ok. 4 passed`

- [ ] **Step 6: Commit**

```bash
git add crates/truss-core/src/
git commit -m "feat: add ContextBlock and AgentEnvelope core types"
```

---

## Task 3: Context Surgeon — SlidingWindow + WeightedPrune

**Files:**
- Create: `crates/truss-core/src/context/surgeon.rs`
- Modify: `crates/truss-core/src/context/mod.rs`

- [ ] **Step 1: Write failing surgeon tests**

Create `crates/truss-core/src/context/surgeon.rs`:

```rust
use crate::context::block::{ContextBlock, ContextRole, ContextWeight};

#[derive(Debug, Clone)]
pub enum CompressionStrategy {
    SlidingWindow { keep_recent: usize },
    WeightedPrune { target_tokens: u32 },
    Hybrid,  // Phase 1: WeightedPrune first, then SlidingWindow if still over budget
}

#[derive(Debug, Clone)]
pub struct SurgeonConfig {
    pub strategy: CompressionStrategy,
    pub target_tokens: u32,
    pub preserve_recent: usize,
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

#[derive(Debug)]
pub struct SurgeonResult {
    pub blocks: Vec<ContextBlock>,
    pub tokens_before: u32,
    pub tokens_after: u32,
    pub tokens_saved: u32,
    pub strategy_applied: String,
}

pub fn compress(blocks: &[ContextBlock], config: &SurgeonConfig) -> SurgeonResult {
    let tokens_before: u32 = blocks.iter().map(|b| b.token_count).sum();

    let result_blocks = match &config.strategy {
        CompressionStrategy::SlidingWindow { keep_recent } => {
            sliding_window(blocks, *keep_recent, config.preserve_recent)
        }
        CompressionStrategy::WeightedPrune { target_tokens } => {
            weighted_prune(blocks, *target_tokens, config.preserve_recent)
        }
        CompressionStrategy::Hybrid => {
            let after_prune = weighted_prune(blocks, config.target_tokens, config.preserve_recent);
            let total: u32 = after_prune.iter().map(|b| b.token_count).sum();
            if total > config.target_tokens {
                sliding_window(&after_prune, config.target_tokens as usize, config.preserve_recent)
            } else {
                after_prune
            }
        }
    };

    let tokens_after: u32 = result_blocks.iter().map(|b| b.token_count).sum();

    SurgeonResult {
        tokens_saved: tokens_before.saturating_sub(tokens_after),
        tokens_before,
        tokens_after,
        blocks: result_blocks,
        strategy_applied: format!("{:?}", config.strategy),
    }
}

fn sliding_window(
    blocks: &[ContextBlock],
    keep_recent: usize,
    preserve_recent: usize,
) -> Vec<ContextBlock> {
    let always_keep = preserve_recent.max(keep_recent);
    if blocks.len() <= always_keep {
        return blocks.to_vec();
    }
    // Always keep Critical/High weight blocks regardless of position
    let mut result: Vec<ContextBlock> = blocks
        .iter()
        .filter(|b| b.weight >= ContextWeight::High)
        .cloned()
        .collect();
    // Append the most recent N blocks that aren't already included
    let recent_start = blocks.len().saturating_sub(keep_recent);
    for block in &blocks[recent_start..] {
        if !result.iter().any(|b| b.id == block.id) {
            result.push(block.clone());
        }
    }
    result
}

fn weighted_prune(
    blocks: &[ContextBlock],
    target_tokens: u32,
    preserve_recent: usize,
) -> Vec<ContextBlock> {
    let total: u32 = blocks.iter().map(|b| b.token_count).sum();
    if total <= target_tokens {
        return blocks.to_vec();
    }

    // Mark the last `preserve_recent` blocks as always kept
    let preserve_ids: std::collections::HashSet<_> = blocks
        .iter()
        .rev()
        .take(preserve_recent)
        .map(|b| b.id)
        .collect();

    // Sort removable blocks by weight ascending (Background first), then oldest first
    let mut removable: Vec<&ContextBlock> = blocks
        .iter()
        .filter(|b| !preserve_ids.contains(&b.id) && b.weight < ContextWeight::High)
        .collect();
    removable.sort_by(|a, b| {
        a.weight
            .partial_cmp(&b.weight)
            .unwrap()
            .then(a.created_at.cmp(&b.created_at))
    });

    let mut to_remove = std::collections::HashSet::new();
    let mut running = total;
    for block in removable {
        if running <= target_tokens {
            break;
        }
        running = running.saturating_sub(block.token_count);
        to_remove.insert(block.id);
    }

    blocks.iter().filter(|b| !to_remove.contains(&b.id)).cloned().collect()
}

/// Heuristic relevance score: how much overlap does a block's content have with the task?
pub fn score_relevance(block: &ContextBlock, task: &str) -> f32 {
    let task_words: std::collections::HashSet<&str> = task.split_whitespace().collect();
    if task_words.is_empty() {
        return 0.0;
    }
    let matches = block
        .content
        .split_whitespace()
        .filter(|w| task_words.contains(w))
        .count();
    (matches as f32 / task_words.len() as f32).min(1.0)
}

/// Heuristic contradiction: returns true if blocks assert opposite values for the same key.
/// Looks for "X is Y" / "X is not Y" or "X: true" / "X: false" patterns.
pub fn detect_contradiction(a: &ContextBlock, b: &ContextBlock) -> bool {
    let a_lower = a.content.to_lowercase();
    let b_lower = b.content.to_lowercase();
    // Simple heuristic: one says "not X" while the other says "X" for same noun
    for word in a_lower.split_whitespace() {
        if word.len() > 4 && b_lower.contains(&format!("not {word}")) {
            return true;
        }
        if a_lower.contains(&format!("not {word}")) && b_lower.contains(word) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(weight: ContextWeight, tokens: u32, content: &str) -> ContextBlock {
        let mut b = ContextBlock::new(
            ContextRole::Finding,
            weight,
            content,
            "test",
        );
        b.token_count = tokens; // override estimate for test control
        b
    }

    #[test]
    fn sliding_window_keeps_recent_n_blocks() {
        let blocks: Vec<_> = (0..10)
            .map(|i| make_block(ContextWeight::Normal, 100, &format!("block {i}")))
            .collect();
        let config = SurgeonConfig {
            strategy: CompressionStrategy::SlidingWindow { keep_recent: 3 },
            target_tokens: 300,
            preserve_recent: 3,
        };
        let result = compress(&blocks, &config);
        assert!(result.tokens_after <= 300, "tokens_after={}", result.tokens_after);
    }

    #[test]
    fn weighted_prune_drops_background_before_normal() {
        let bg = make_block(ContextWeight::Background, 500, "background info");
        let normal = make_block(ContextWeight::Normal, 500, "normal info");
        let critical = make_block(ContextWeight::Critical, 100, "critical info");
        let blocks = vec![bg.clone(), normal.clone(), critical.clone()];
        let config = SurgeonConfig {
            strategy: CompressionStrategy::WeightedPrune { target_tokens: 700 },
            target_tokens: 700,
            preserve_recent: 0,
        };
        let result = compress(&blocks, &config);
        let ids: Vec<_> = result.blocks.iter().map(|b| b.id).collect();
        assert!(!ids.contains(&bg.id), "background should be pruned");
        assert!(ids.contains(&critical.id), "critical should survive");
    }

    #[test]
    fn compress_never_removes_critical_blocks() {
        let critical = make_block(ContextWeight::Critical, 9000, "must keep");
        let bg = make_block(ContextWeight::Background, 100, "droppable");
        let config = SurgeonConfig {
            strategy: CompressionStrategy::Hybrid,
            target_tokens: 500,
            preserve_recent: 0,
        };
        let result = compress(&[critical.clone(), bg], &config);
        assert!(result.blocks.iter().any(|b| b.id == critical.id));
    }

    #[test]
    fn score_relevance_returns_zero_for_empty_task() {
        let block = make_block(ContextWeight::Normal, 10, "some content");
        assert_eq!(score_relevance(&block, ""), 0.0);
    }

    #[test]
    fn detect_contradiction_catches_not_pattern() {
        let a = make_block(ContextWeight::Normal, 10, "the service is available");
        let b = make_block(ContextWeight::Normal, 10, "the service is not available");
        assert!(detect_contradiction(&a, &b));
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
cargo test -p truss-core context::surgeon
```

Expected: compile error (surgeon not yet wired into mod.rs)

- [ ] **Step 3: Wire surgeon into context/mod.rs**

Replace `crates/truss-core/src/context/mod.rs`:

```rust
pub mod block;
pub mod surgeon;

pub use block::{ContextBlock, ContextRole, ContextWeight, estimate_tokens};
pub use surgeon::{compress, CompressionStrategy, SurgeonConfig, SurgeonResult, score_relevance, detect_contradiction};
```

Add to `crates/truss-core/src/lib.rs`:

```rust
pub use context::{
    ContextBlock, ContextRole, ContextWeight, estimate_tokens,
    compress, CompressionStrategy, SurgeonConfig, SurgeonResult,
    score_relevance, detect_contradiction,
};
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p truss-core
```

Expected: `test result: ok. 9 passed`

- [ ] **Step 5: Commit**

```bash
git add crates/truss-core/src/context/
git commit -m "feat: add context surgeon with SlidingWindow and WeightedPrune strategies"
```

---

## Task 4: Budget Ledger — InMemory Store + Config Types

**Files:**
- Create: `crates/truss-core/src/budget/config.rs`
- Create: `crates/truss-core/src/budget/ledger.rs`
- Create: `crates/truss-core/src/budget/memory_store.rs`
- Create: `crates/truss-core/src/budget/mod.rs`
- Modify: `crates/truss-core/src/lib.rs`

- [ ] **Step 1: Write the config types**

Create `crates/truss-core/src/budget/config.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BudgetWindow {
    PerSession,
    PerHour,
    PerDay,
    PerMonth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetLimit {
    pub tokens: Option<u64>,
    pub usd: Option<f64>,
    pub window: BudgetWindow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub slack_webhook: Option<String>,
    pub log_to_stderr: bool, // always-available fallback
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self { slack_webhook: None, log_to_stderr: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExceededAction {
    Block,
    Fallback(crate::types::ModelTier),
    Queue,
    GracefulResponse(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    pub per_session: Option<BudgetLimit>,
    pub per_user: Option<BudgetLimit>,
    pub per_agent: Option<BudgetLimit>,
    pub global: Option<BudgetLimit>,
    pub on_exceeded: ExceededAction,
    pub alert_at_pct: f32,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_80pct_alert() {
        let cfg = BudgetConfig::default();
        assert!((cfg.alert_at_pct - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn budget_limit_serialises() {
        let limit = BudgetLimit { tokens: Some(10_000), usd: Some(1.0), window: BudgetWindow::PerSession };
        let json = serde_json::to_string(&limit).unwrap();
        assert!(json.contains("PerSession"));
    }
}
```

- [ ] **Step 2: Write LedgerStore trait and entry type**

Create `crates/truss-core/src/budget/ledger.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use crate::Result;
use super::config::BudgetWindow;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl LedgerEntry {
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageReport {
    pub key: String,
    pub total_tokens: u64,
    pub total_cost_usd: f64,
    pub remaining_tokens: Option<u64>,
    pub remaining_usd: Option<f64>,
    pub pct_used: f32,
    pub window: BudgetWindow,
}

/// Normalised usage from any LLM provider response.
#[derive(Debug, Clone)]
pub struct UsageRecord {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub model: String,
    /// Cost in USD. If None, caller should compute from token counts.
    pub cost_usd: Option<f64>,
}

pub trait LedgerStore: Send + Sync {
    fn record(&self, entry: LedgerEntry) -> Result<()>;
    fn usage(&self, key: &str, window: &BudgetWindow) -> Result<UsageReport>;
    fn flush(&self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_tokens_sums_input_and_output() {
        let entry = LedgerEntry {
            id: Uuid::new_v4(),
            session_id: "s1".to_string(),
            user_id: None,
            agent_name: None,
            model: "claude-haiku".to_string(),
            input_tokens: 100,
            output_tokens: 50,
            cost_usd: 0.001,
            timestamp: 0,
            tags: HashMap::new(),
        };
        assert_eq!(entry.total_tokens(), 150);
    }
}
```

- [ ] **Step 3: Write InMemoryStore**

Create `crates/truss-core/src/budget/memory_store.rs`:

```rust
use std::sync::RwLock;
use uuid::Uuid;
use crate::Result;
use super::ledger::{LedgerEntry, LedgerStore, UsageReport};
use super::config::BudgetWindow;

pub struct InMemoryStore {
    entries: RwLock<Vec<LedgerEntry>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self { entries: RwLock::new(Vec::new()) }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl LedgerStore for InMemoryStore {
    fn record(&self, entry: LedgerEntry) -> Result<()> {
        self.entries.write().unwrap().push(entry);
        Ok(())
    }

    fn usage(&self, key: &str, window: &BudgetWindow) -> Result<UsageReport> {
        let entries = self.entries.read().unwrap();
        let matching: Vec<&LedgerEntry> = entries
            .iter()
            .filter(|e| {
                e.session_id == key
                    || e.user_id.as_deref() == Some(key)
                    || e.agent_name.as_deref() == Some(key)
            })
            .collect();

        let total_tokens: u64 = matching.iter().map(|e| e.total_tokens() as u64).sum();
        let total_cost: f64 = matching.iter().map(|e| e.cost_usd).sum();

        Ok(UsageReport {
            key: key.to_string(),
            total_tokens,
            total_cost_usd: total_cost,
            remaining_tokens: None,
            remaining_usd: None,
            pct_used: 0.0,
            window: window.clone(),
        })
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn make_entry(session_id: &str, tokens: u32, cost: f64) -> LedgerEntry {
        LedgerEntry {
            id: Uuid::new_v4(),
            session_id: session_id.to_string(),
            user_id: None,
            agent_name: None,
            model: "test-model".to_string(),
            input_tokens: tokens,
            output_tokens: 0,
            cost_usd: cost,
            timestamp: 0,
            tags: HashMap::new(),
        }
    }

    #[test]
    fn record_and_query_usage_by_session() {
        let store = InMemoryStore::new();
        store.record(make_entry("sess-1", 1000, 0.50)).unwrap();
        store.record(make_entry("sess-1", 500, 0.25)).unwrap();
        store.record(make_entry("sess-2", 999, 9.99)).unwrap();

        let report = store.usage("sess-1", &BudgetWindow::PerSession).unwrap();
        assert_eq!(report.total_tokens, 1500);
        assert!((report.total_cost_usd - 0.75).abs() < 0.001);
    }

    #[test]
    fn unknown_key_returns_zero_usage() {
        let store = InMemoryStore::new();
        let report = store.usage("nobody", &BudgetWindow::PerSession).unwrap();
        assert_eq!(report.total_tokens, 0);
    }

    #[test]
    fn concurrent_writes_do_not_panic() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(InMemoryStore::new());
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let s = Arc::clone(&store);
                thread::spawn(move || {
                    s.record(make_entry(&format!("sess-{i}"), 100, 0.01)).unwrap();
                })
            })
            .collect();
        for h in handles { h.join().unwrap(); }
        let report = store.usage("sess-0", &BudgetWindow::PerSession).unwrap();
        assert_eq!(report.total_tokens, 100);
    }
}
```

- [ ] **Step 4: Wire budget module**

Create `crates/truss-core/src/budget/mod.rs`:

```rust
pub mod config;
pub mod ledger;
pub mod memory_store;

pub use config::{BudgetConfig, BudgetLimit, BudgetWindow, AlertConfig, ExceededAction};
pub use ledger::{LedgerEntry, LedgerStore, UsageReport, UsageRecord};
pub use memory_store::InMemoryStore;
```

Add to `crates/truss-core/src/lib.rs`:

```rust
pub mod budget;
pub use budget::{BudgetConfig, BudgetLimit, BudgetWindow, AlertConfig, ExceededAction, LedgerEntry, LedgerStore, UsageReport, UsageRecord, InMemoryStore};
```

- [ ] **Step 5: Run all tests**

```bash
cargo test -p truss-core
```

Expected: `test result: ok. 14 passed`

- [ ] **Step 6: Commit**

```bash
git add crates/truss-core/src/budget/
git commit -m "feat: add budget config types, LedgerStore trait, InMemoryStore"
```

---

## Task 5: Agent Handoff — Pack and Unpack

**Files:**
- Create: `crates/truss-core/src/handoff/envelope.rs`
- Create: `crates/truss-core/src/handoff/mod.rs`
- Modify: `crates/truss-core/src/lib.rs`

- [ ] **Step 1: Write failing handoff tests**

Create `crates/truss-core/src/handoff/envelope.rs`:

```rust
use crate::context::block::{ContextBlock, ContextWeight};
use crate::types::AgentEnvelope;

/// How much of a parent budget to carve for a child agent.
#[derive(Debug, Clone)]
pub enum BudgetCarve {
    FixedUsd(f64),
    FixedTokens(u64),
    Percent(f32), // 0.0–1.0 of parent remaining
}

/// Create a child envelope from a parent, carrying only the requested context weights.
pub fn pack(
    parent: &AgentEnvelope,
    task: impl Into<String>,
    carry_weights: &[ContextWeight],
    budget_carve: BudgetCarve,
) -> AgentEnvelope {
    let carved_budget = match budget_carve {
        BudgetCarve::FixedUsd(usd) => usd.min(parent.budget_usd_remaining),
        BudgetCarve::Percent(pct) => parent.budget_usd_remaining * (pct as f64),
        BudgetCarve::FixedTokens(_) => parent.budget_usd_remaining * 0.5, // tokens→usd requires model rates; use 50% as safe default
    };

    let filtered_context: Vec<ContextBlock> = parent
        .context
        .iter()
        .filter(|b| carry_weights.contains(&b.weight))
        .cloned()
        .collect();

    let mut child = AgentEnvelope::new(task);
    child.context = filtered_context;
    child.budget_usd_remaining = carved_budget;
    child.parent_agent = Some(parent.id.to_string());
    child.model_hint = parent.model_hint.clone();
    child
}

/// Expand a received envelope into its context blocks for direct use.
pub fn unpack(envelope: &AgentEnvelope) -> Vec<ContextBlock> {
    envelope.context.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::block::{ContextRole, ContextWeight};
    use crate::types::ModelTier;

    fn parent_envelope() -> AgentEnvelope {
        let mut env = AgentEnvelope::new("parent task");
        env.budget_usd_remaining = 1.0;

        env.context.push(ContextBlock::new(
            ContextRole::Task, ContextWeight::Critical, "critical info", "planner",
        ));
        env.context.push(ContextBlock::new(
            ContextRole::Background, ContextWeight::Background, "background fluff", "loader",
        ));
        env
    }

    #[test]
    fn pack_filters_context_by_weight() {
        let parent = parent_envelope();
        let child = pack(
            &parent,
            "child task",
            &[ContextWeight::Critical],
            BudgetCarve::FixedUsd(0.20),
        );
        assert_eq!(child.context.len(), 1);
        assert_eq!(child.context[0].weight, ContextWeight::Critical);
    }

    #[test]
    fn pack_sets_parent_agent_id() {
        let parent = parent_envelope();
        let parent_id = parent.id.to_string();
        let child = pack(&parent, "child", &[ContextWeight::Critical], BudgetCarve::FixedUsd(0.1));
        assert_eq!(child.parent_agent, Some(parent_id));
    }

    #[test]
    fn pack_does_not_exceed_parent_budget() {
        let parent = parent_envelope(); // 1.0 USD remaining
        let child = pack(&parent, "child", &[], BudgetCarve::FixedUsd(999.0));
        assert!(child.budget_usd_remaining <= 1.0);
    }

    #[test]
    fn unpack_returns_all_context_blocks() {
        let parent = parent_envelope();
        let blocks = unpack(&parent);
        assert_eq!(blocks.len(), parent.context.len());
    }
}
```

Create `crates/truss-core/src/handoff/mod.rs`:

```rust
pub mod envelope;
pub use envelope::{pack, unpack, BudgetCarve};
```

Add to `crates/truss-core/src/lib.rs`:

```rust
pub mod handoff;
pub use handoff::{pack, unpack, BudgetCarve};
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p truss-core handoff
```

Expected: `test result: ok. 4 passed`

- [ ] **Step 3: Commit**

```bash
git add crates/truss-core/src/handoff/
git commit -m "feat: add agent handoff pack/unpack with BudgetCarve"
```

---

## Task 6: SQLite Ledger Store

**Files:**
- Create: `crates/truss-core/src/budget/sqlite_store.rs`
- Modify: `crates/truss-core/src/budget/mod.rs`

- [ ] **Step 1: Write failing SQLite store tests**

Create `crates/truss-core/src/budget/sqlite_store.rs`:

```rust
use rusqlite::{Connection, params};
use std::sync::Mutex;
use uuid::Uuid;
use std::collections::HashMap;
use crate::Result;
use crate::error::TrussError;
use super::config::BudgetWindow;
use super::ledger::{LedgerEntry, LedgerStore, UsageReport};

pub struct SqliteLedgerStore {
    conn: Mutex<Connection>,
}

impl SqliteLedgerStore {
    pub fn new(path: &str) -> Result<Self> {
        let conn = if path == ":memory:" {
            Connection::open_in_memory()
        } else {
            Connection::open(path)
        }
        .map_err(|e| TrussError::Storage(e.to_string()))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS ledger (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                user_id     TEXT,
                agent_name  TEXT,
                model       TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd    REAL NOT NULL,
                timestamp   INTEGER NOT NULL,
                tags        TEXT NOT NULL
            );",
        )
        .map_err(|e| TrussError::Storage(e.to_string()))?;

        Ok(Self { conn: Mutex::new(conn) })
    }
}

impl LedgerStore for SqliteLedgerStore {
    fn record(&self, entry: LedgerEntry) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let tags = serde_json::to_string(&entry.tags)?;
        conn.execute(
            "INSERT INTO ledger VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                entry.id.to_string(),
                entry.session_id,
                entry.user_id,
                entry.agent_name,
                entry.model,
                entry.input_tokens,
                entry.output_tokens,
                entry.cost_usd,
                entry.timestamp,
                tags,
            ],
        )
        .map_err(|e| TrussError::Storage(e.to_string()))?;
        Ok(())
    }

    fn usage(&self, key: &str, window: &BudgetWindow) -> Result<UsageReport> {
        let conn = self.conn.lock().unwrap();
        let (total_tokens, total_cost): (i64, f64) = conn
            .query_row(
                "SELECT COALESCE(SUM(input_tokens + output_tokens), 0),
                        COALESCE(SUM(cost_usd), 0.0)
                 FROM ledger
                 WHERE session_id = ?1 OR user_id = ?1 OR agent_name = ?1",
                params![key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|e| TrussError::Storage(e.to_string()))?;

        Ok(UsageReport {
            key: key.to_string(),
            total_tokens: total_tokens as u64,
            total_cost_usd: total_cost,
            remaining_tokens: None,
            remaining_usd: None,
            pct_used: 0.0,
            window: window.clone(),
        })
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_entry(session_id: &str, tokens: u32, cost: f64) -> LedgerEntry {
        LedgerEntry {
            id: Uuid::new_v4(),
            session_id: session_id.to_string(),
            user_id: None,
            agent_name: None,
            model: "test".to_string(),
            input_tokens: tokens,
            output_tokens: 0,
            cost_usd: cost,
            timestamp: 0,
            tags: HashMap::new(),
        }
    }

    #[test]
    fn sqlite_store_records_and_queries() {
        let store = SqliteLedgerStore::new(":memory:").unwrap();
        store.record(make_entry("sess-a", 100, 0.10)).unwrap();
        store.record(make_entry("sess-a", 200, 0.20)).unwrap();
        let report = store.usage("sess-a", &BudgetWindow::PerSession).unwrap();
        assert_eq!(report.total_tokens, 300);
        assert!((report.total_cost_usd - 0.30).abs() < 0.001);
    }

    #[test]
    fn sqlite_store_survives_reopen() {
        use tempfile::NamedTempFile;
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        {
            let store = SqliteLedgerStore::new(&path).unwrap();
            store.record(make_entry("persistent-sess", 500, 0.50)).unwrap();
        }
        let store2 = SqliteLedgerStore::new(&path).unwrap();
        let report = store2.usage("persistent-sess", &BudgetWindow::PerSession).unwrap();
        assert_eq!(report.total_tokens, 500);
    }
}
```

- [ ] **Step 2: Add tempfile dev-dependency**

Add to `crates/truss-core/Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 3: Export from budget mod**

Add to `crates/truss-core/src/budget/mod.rs`:

```rust
pub mod sqlite_store;
pub use sqlite_store::SqliteLedgerStore;
```

Add to `crates/truss-core/src/lib.rs`:

```rust
pub use budget::SqliteLedgerStore;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p truss-core budget::sqlite
```

Expected: `test result: ok. 2 passed`

- [ ] **Step 5: Commit**

```bash
git add crates/truss-core/src/budget/sqlite_store.rs crates/truss-core/Cargo.toml
git commit -m "feat: add SQLite ledger store with persistence"
```

---

## Task 7: Circuit Breaker

**Files:**
- Create: `crates/truss-core/src/budget/circuit_breaker.rs`
- Modify: `crates/truss-core/src/budget/mod.rs`

- [ ] **Step 1: Write circuit breaker**

Create `crates/truss-core/src/budget/circuit_breaker.rs`:

```rust
use std::collections::VecDeque;
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub max_requests_per_minute: u32,
    pub max_cost_velocity_usd: f64,
    pub max_retry_depth: u8,
    pub trip_on_repeated_prompt: bool,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            max_requests_per_minute: 60,
            max_cost_velocity_usd: 1.0,
            max_retry_depth: 3,
            trip_on_repeated_prompt: true,
        }
    }
}

#[derive(Debug)]
pub enum CircuitTrip {
    RateLimit,
    CostVelocity { usd_per_minute: f64 },
    MaxRetryDepth,
    RepeatedPrompt,
}

struct RequestRecord {
    timestamp_ms: u64,
    cost_usd: f64,
    prompt_hash: u64,
}

pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    window: Mutex<VecDeque<RequestRecord>>,
    current_retry_depth: Mutex<u8>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            window: Mutex::new(VecDeque::new()),
            current_retry_depth: Mutex::new(0),
        }
    }

    /// Record a request. Returns a trip reason if the breaker should fire.
    pub fn check_and_record(
        &self,
        prompt: &str,
        cost_usd: f64,
        now_ms: u64,
    ) -> Option<CircuitTrip> {
        let prompt_hash = simple_hash(prompt);
        let mut window = self.window.lock().unwrap();

        // Evict entries older than 60 seconds
        let cutoff = now_ms.saturating_sub(60_000);
        while window.front().map_or(false, |r| r.timestamp_ms < cutoff) {
            window.pop_front();
        }

        // Check rate limit
        if window.len() as u32 >= self.config.max_requests_per_minute {
            return Some(CircuitTrip::RateLimit);
        }

        // Check cost velocity
        let total_cost: f64 = window.iter().map(|r| r.cost_usd).sum();
        if total_cost + cost_usd > self.config.max_cost_velocity_usd {
            return Some(CircuitTrip::CostVelocity { usd_per_minute: total_cost + cost_usd });
        }

        // Check repeated prompt (exact hash match in last 3 requests)
        if self.config.trip_on_repeated_prompt {
            let recent_match = window
                .iter()
                .rev()
                .take(3)
                .any(|r| r.prompt_hash == prompt_hash);
            if recent_match {
                return Some(CircuitTrip::RepeatedPrompt);
            }
        }

        window.push_back(RequestRecord { timestamp_ms: now_ms, cost_usd, prompt_hash });
        None
    }

    pub fn increment_retry(&self) -> Option<CircuitTrip> {
        let mut depth = self.current_retry_depth.lock().unwrap();
        *depth += 1;
        if *depth > self.config.max_retry_depth {
            Some(CircuitTrip::MaxRetryDepth)
        } else {
            None
        }
    }

    pub fn reset_retry(&self) {
        *self.current_retry_depth.lock().unwrap() = 0;
    }
}

fn simple_hash(s: &str) -> u64 {
    // FNV-1a 64-bit
    let mut hash: u64 = 14695981039346656037;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_trips_on_rate_limit() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            max_requests_per_minute: 3,
            ..Default::default()
        });
        for i in 0..3 {
            assert!(breaker.check_and_record("prompt", 0.01, i * 1000).is_none());
        }
        let trip = breaker.check_and_record("prompt", 0.01, 3000);
        assert!(matches!(trip, Some(CircuitTrip::RateLimit)));
    }

    #[test]
    fn circuit_trips_on_cost_velocity() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            max_cost_velocity_usd: 0.50,
            ..Default::default()
        });
        breaker.check_and_record("a", 0.40, 0);
        let trip = breaker.check_and_record("b", 0.20, 1000);
        assert!(matches!(trip, Some(CircuitTrip::CostVelocity { .. })));
    }

    #[test]
    fn circuit_trips_on_repeated_prompt() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
        breaker.check_and_record("same prompt", 0.01, 0);
        let trip = breaker.check_and_record("same prompt", 0.01, 1000);
        assert!(matches!(trip, Some(CircuitTrip::RepeatedPrompt)));
    }

    #[test]
    fn different_prompts_do_not_trip_repeat_detector() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
        breaker.check_and_record("prompt A", 0.01, 0);
        let result = breaker.check_and_record("prompt B", 0.01, 1000);
        assert!(result.is_none());
    }

    #[test]
    fn retry_depth_trips_after_max() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            max_retry_depth: 2,
            ..Default::default()
        });
        assert!(breaker.increment_retry().is_none());
        assert!(breaker.increment_retry().is_none());
        assert!(matches!(breaker.increment_retry(), Some(CircuitTrip::MaxRetryDepth)));
    }
}
```

- [ ] **Step 2: Wire into budget mod**

Add to `crates/truss-core/src/budget/mod.rs`:

```rust
pub mod circuit_breaker;
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitTrip};
```

Add to `crates/truss-core/src/lib.rs`:

```rust
pub use budget::{CircuitBreaker, CircuitBreakerConfig, CircuitTrip};
```

- [ ] **Step 3: Run all tests**

```bash
cargo test -p truss-core
```

Expected: `test result: ok. 24 passed` (approximate — count grows with each task)

- [ ] **Step 4: Commit**

```bash
git add crates/truss-core/src/budget/circuit_breaker.rs
git commit -m "feat: add circuit breaker with rate, cost velocity, and repeat detection"
```

---

## Task 8: Checkpoints + Rollback

**Files:**
- Create: `crates/truss-core/src/coord/checkpoint.rs`
- Create: `crates/truss-core/src/coord/sqlite_checkpoint.rs`
- Create: `crates/truss-core/src/coord/mod.rs`
- Modify: `crates/truss-core/src/lib.rs`

- [ ] **Step 1: Write checkpoint types and trait**

Create `crates/truss-core/src/coord/checkpoint.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use crate::types::AgentEnvelope;
use crate::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub id: Uuid,
    pub session_id: String,
    pub agent_name: String,
    pub description: String,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: Uuid,
    pub session_id: String,
    pub agent_name: String,
    pub envelope_snapshot: AgentEnvelope,
    pub external_state: HashMap<String, Vec<u8>>,
    pub created_at: u64,
    pub description: String,
}

impl Checkpoint {
    pub fn meta(&self) -> CheckpointMeta {
        CheckpointMeta {
            id: self.id,
            session_id: self.session_id.clone(),
            agent_name: self.agent_name.clone(),
            description: self.description.clone(),
            created_at: self.created_at,
        }
    }
}

pub trait CheckpointStore: Send + Sync {
    fn save(&self, cp: Checkpoint) -> Result<Uuid>;
    fn load(&self, id: Uuid) -> Result<Checkpoint>;
    fn rollback(&self, id: Uuid) -> Result<AgentEnvelope>;
    fn list(&self, session_id: &str) -> Result<Vec<CheckpointMeta>>;
}
```

- [ ] **Step 2: Write InMemoryCheckpointStore**

Add to `crates/truss-core/src/coord/checkpoint.rs` (append below the trait):

```rust
use std::sync::RwLock;
use crate::error::TrussError;

pub struct InMemoryCheckpointStore {
    checkpoints: RwLock<HashMap<Uuid, Checkpoint>>,
}

impl InMemoryCheckpointStore {
    pub fn new() -> Self {
        Self { checkpoints: RwLock::new(HashMap::new()) }
    }
}

impl Default for InMemoryCheckpointStore {
    fn default() -> Self { Self::new() }
}

impl CheckpointStore for InMemoryCheckpointStore {
    fn save(&self, cp: Checkpoint) -> Result<Uuid> {
        let id = cp.id;
        self.checkpoints.write().unwrap().insert(id, cp);
        Ok(id)
    }

    fn load(&self, id: Uuid) -> Result<Checkpoint> {
        self.checkpoints
            .read()
            .unwrap()
            .get(&id)
            .cloned()
            .ok_or_else(|| TrussError::CheckpointNotFound { id: id.to_string() })
    }

    fn rollback(&self, id: Uuid) -> Result<AgentEnvelope> {
        let cp = self.load(id)?;
        Ok(cp.envelope_snapshot)
    }

    fn list(&self, session_id: &str) -> Result<Vec<CheckpointMeta>> {
        let checkpoints = self.checkpoints.read().unwrap();
        let mut metas: Vec<CheckpointMeta> = checkpoints
            .values()
            .filter(|cp| cp.session_id == session_id)
            .map(|cp| cp.meta())
            .collect();
        metas.sort_by_key(|m| m.created_at);
        Ok(metas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_checkpoint(session_id: &str, desc: &str) -> Checkpoint {
        Checkpoint {
            id: Uuid::new_v4(),
            session_id: session_id.to_string(),
            agent_name: "test-agent".to_string(),
            envelope_snapshot: AgentEnvelope::new("test task"),
            external_state: HashMap::new(),
            created_at: 0,
            description: desc.to_string(),
        }
    }

    #[test]
    fn save_and_load_checkpoint() {
        let store = InMemoryCheckpointStore::new();
        let cp = make_checkpoint("sess-1", "after planning");
        let id = store.save(cp.clone()).unwrap();
        let loaded = store.load(id).unwrap();
        assert_eq!(loaded.description, "after planning");
    }

    #[test]
    fn rollback_returns_envelope_snapshot() {
        let store = InMemoryCheckpointStore::new();
        let mut cp = make_checkpoint("sess-1", "step 1");
        cp.envelope_snapshot.task = "original task".to_string();
        let id = store.save(cp).unwrap();
        let envelope = store.rollback(id).unwrap();
        assert_eq!(envelope.task, "original task");
    }

    #[test]
    fn load_nonexistent_returns_error() {
        let store = InMemoryCheckpointStore::new();
        let result = store.load(Uuid::new_v4());
        assert!(matches!(result, Err(TrussError::CheckpointNotFound { .. })));
    }

    #[test]
    fn list_filters_by_session() {
        let store = InMemoryCheckpointStore::new();
        store.save(make_checkpoint("sess-a", "cp1")).unwrap();
        store.save(make_checkpoint("sess-b", "cp2")).unwrap();
        let list = store.list("sess-a").unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].description, "cp1");
    }
}
```

- [ ] **Step 3: Wire coord module**

Create `crates/truss-core/src/coord/mod.rs`:

```rust
pub mod checkpoint;
pub use checkpoint::{Checkpoint, CheckpointMeta, CheckpointStore, InMemoryCheckpointStore};
```

Add to `crates/truss-core/src/lib.rs`:

```rust
pub mod coord;
pub use coord::{Checkpoint, CheckpointMeta, CheckpointStore, InMemoryCheckpointStore};
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p truss-core coord
```

Expected: `test result: ok. 4 passed`

- [ ] **Step 5: Commit**

```bash
git add crates/truss-core/src/coord/
git commit -m "feat: add checkpoint types with InMemoryCheckpointStore"
```

---

## Task 9: Fence — In-Memory Shared State Coordination

**Files:**
- Create: `crates/truss-core/src/fence/memory_fence.rs`
- Create: `crates/truss-core/src/fence/mod.rs`
- Modify: `crates/truss-core/src/lib.rs`

- [ ] **Step 1: Write Fence trait and InMemoryFence**

Create `crates/truss-core/src/fence/memory_fence.rs`:

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::{Result, TrussError};

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
    /// Attempt to acquire lock. Returns Ok(()) on success, Err(FenceLockConflict) if held by another.
    fn acquire(&self, key: &str, owner: &str, ttl_ms: u64, now_ms: u64) -> Result<()>;
    /// Release a lock. No-op if the key is not held by `owner`.
    fn release(&self, key: &str, owner: &str) -> Result<()>;
    /// Returns true if the key is currently locked (and not expired).
    fn is_locked(&self, key: &str, now_ms: u64) -> bool;
}

pub struct InMemoryFence {
    locks: Arc<Mutex<HashMap<String, LockHandle>>>,
}

impl InMemoryFence {
    pub fn new() -> Self {
        Self { locks: Arc::new(Mutex::new(HashMap::new())) }
    }
}

impl Default for InMemoryFence {
    fn default() -> Self { Self::new() }
}

impl FenceStore for InMemoryFence {
    fn acquire(&self, key: &str, owner: &str, ttl_ms: u64, now_ms: u64) -> Result<()> {
        let mut locks = self.locks.lock().unwrap();

        if let Some(handle) = locks.get(key) {
            if !handle.is_expired(now_ms) && handle.owner != owner {
                return Err(TrussError::FenceLockConflict {
                    key: key.to_string(),
                    owner: handle.owner.clone(),
                });
            }
        }

        locks.insert(
            key.to_string(),
            LockHandle {
                key: key.to_string(),
                owner: owner.to_string(),
                acquired_at_ms: now_ms,
                ttl_ms,
            },
        );
        Ok(())
    }

    fn release(&self, key: &str, owner: &str) -> Result<()> {
        let mut locks = self.locks.lock().unwrap();
        if let Some(handle) = locks.get(key) {
            if handle.owner == owner {
                locks.remove(key);
            }
        }
        Ok(())
    }

    fn is_locked(&self, key: &str, now_ms: u64) -> bool {
        let locks = self.locks.lock().unwrap();
        locks.get(key).map_or(false, |h| !h.is_expired(now_ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_grants_lock_to_first_owner() {
        let fence = InMemoryFence::new();
        assert!(fence.acquire("doc-1", "agent-a", 30_000, 0).is_ok());
    }

    #[test]
    fn acquire_blocks_second_owner() {
        let fence = InMemoryFence::new();
        fence.acquire("doc-1", "agent-a", 30_000, 0).unwrap();
        let result = fence.acquire("doc-1", "agent-b", 30_000, 1000);
        assert!(matches!(result, Err(TrussError::FenceLockConflict { .. })));
    }

    #[test]
    fn expired_lock_can_be_reacquired() {
        let fence = InMemoryFence::new();
        fence.acquire("doc-1", "agent-a", 5_000, 0).unwrap(); // 5 second TTL
        // Now "30 seconds later"
        assert!(fence.acquire("doc-1", "agent-b", 5_000, 30_000).is_ok());
    }

    #[test]
    fn release_frees_lock_for_others() {
        let fence = InMemoryFence::new();
        fence.acquire("doc-1", "agent-a", 30_000, 0).unwrap();
        fence.release("doc-1", "agent-a").unwrap();
        assert!(fence.acquire("doc-1", "agent-b", 30_000, 1000).is_ok());
    }

    #[test]
    fn release_by_wrong_owner_is_noop() {
        let fence = InMemoryFence::new();
        fence.acquire("doc-1", "agent-a", 30_000, 0).unwrap();
        fence.release("doc-1", "agent-b").unwrap(); // wrong owner — no-op
        assert!(fence.is_locked("doc-1", 1000));
    }
}
```

Create `crates/truss-core/src/fence/mod.rs`:

```rust
pub mod memory_fence;
pub use memory_fence::{FenceStore, InMemoryFence, LockHandle};
```

Add to `crates/truss-core/src/lib.rs`:

```rust
pub mod fence;
pub use fence::{FenceStore, InMemoryFence, LockHandle};
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p truss-core fence
```

Expected: `test result: ok. 5 passed`

- [ ] **Step 3: Run all core tests**

```bash
cargo test -p truss-core
```

Expected: all tests pass (no failures)

- [ ] **Step 4: Commit**

```bash
git add crates/truss-core/src/fence/
git commit -m "feat: add in-memory fence with TTL-based lock expiry"
```

---

## Task 10: Criterion Benchmarks

**Files:**
- Create: `benches/compression.rs`
- Modify: `crates/truss-core/Cargo.toml`

- [ ] **Step 1: Add Criterion dependency**

Add to `crates/truss-core/Cargo.toml`:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
tempfile = "3"

[[bench]]
name = "compression"
harness = false
path = "../../benches/compression.rs"
```

Wait — the bench is at workspace root level, not inside the crate. Update `Cargo.toml` (workspace root) instead:

Add to workspace root `Cargo.toml`:

```toml
[workspace.dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

Create `benches/compression.rs` at workspace root:

```rust
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use truss_core::context::block::{ContextBlock, ContextRole, ContextWeight};
use truss_core::context::surgeon::{compress, CompressionStrategy, SurgeonConfig};

fn make_blocks(n: usize) -> Vec<ContextBlock> {
    (0..n)
        .map(|i| {
            let weight = match i % 4 {
                0 => ContextWeight::Critical,
                1 => ContextWeight::High,
                2 => ContextWeight::Normal,
                _ => ContextWeight::Background,
            };
            let mut b = ContextBlock::new(
                ContextRole::Finding,
                weight,
                "The quick brown fox jumps over the lazy dog. ".repeat(10),
                "bench",
            );
            b.token_count = 100;
            b
        })
        .collect()
}

fn bench_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress");
    for size in [50, 200, 1000] {
        let blocks = make_blocks(size);
        let config = SurgeonConfig {
            strategy: CompressionStrategy::WeightedPrune { target_tokens: 3000 },
            target_tokens: 3000,
            preserve_recent: 5,
        };
        group.bench_with_input(BenchmarkId::new("WeightedPrune", size), &blocks, |b, blocks| {
            b.iter(|| compress(blocks, &config));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compress);
criterion_main!(benches);
```

Because cross-crate benches need the crate listed, add to workspace root `Cargo.toml`:

```toml
[[bench]]
name = "compression"
harness = false
```

And add a `[dependencies]` section to the bench by creating `benches/` as part of a separate mini-crate or using the workspace crate directly. Actually the cleanest approach is to add the bench inside `truss-core`:

Move the bench: copy the file to `crates/truss-core/benches/compression.rs` and ensure the `[[bench]]` is in `crates/truss-core/Cargo.toml`.

- [ ] **Step 2: Run benchmarks to confirm they compile**

```bash
cargo bench -p truss-core
```

Expected: benchmark output with timing for WeightedPrune at 50/200/1000 blocks. No panics.

- [ ] **Step 3: Commit**

```bash
git add crates/truss-core/benches/ crates/truss-core/Cargo.toml
git commit -m "bench: add Criterion compression benchmark"
```

---

## Task 11: PyO3 Bindings + Maturin Setup

**Files:**
- Create: `crates/truss-py/Cargo.toml`
- Create: `crates/truss-py/pyproject.toml`
- Create: `crates/truss-py/src/lib.rs`
- Create: `crates/truss-py/src/context_bindings.rs`
- Create: `crates/truss-py/src/budget_bindings.rs`
- Create: `crates/truss-py/src/handoff_bindings.rs`
- Create: `python/truss/__init__.py`
- Create: `python/truss/types.py`
- Create: `python/truss/session.py`

- [ ] **Step 1: Write truss-py Cargo.toml**

```toml
[package]
name = "truss-py"
version = "0.1.0"
edition = "2021"

[lib]
name = "truss"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
truss-core = { path = "../truss-core" }
uuid = { version = "1.10", features = ["v4"] }
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "truss-ai"
version = "0.1.0"
requires-python = ">=3.10"
description = "The structural layer for agentic AI workflows"
dependencies = ["pydantic>=2.0"]

[tool.maturin]
python-source = "python"
module-name = "truss._core"
features = ["pyo3/extension-module"]
```

- [ ] **Step 3: Write context bindings**

Create `crates/truss-py/src/context_bindings.rs`:

```rust
use pyo3::prelude::*;
use truss_core::context::block::{ContextBlock as RustBlock, ContextRole, ContextWeight};
use truss_core::context::surgeon::{compress as rust_compress, CompressionStrategy, SurgeonConfig};
use uuid::Uuid;

#[pyclass(name = "ContextWeight")]
#[derive(Clone)]
pub enum PyContextWeight {
    Critical,
    High,
    Normal,
    Background,
}

#[pyclass(name = "ContextRole")]
#[derive(Clone)]
pub enum PyContextRole {
    Task,
    Constraint,
    Finding,
    Decision,
    Warning,
    Background,
}

#[pyclass(name = "ContextBlock")]
#[derive(Clone)]
pub struct PyContextBlock {
    pub inner: RustBlock,
}

#[pymethods]
impl PyContextBlock {
    #[new]
    pub fn new(role: PyContextRole, weight: PyContextWeight, content: String, source: String) -> Self {
        let rust_role = match role {
            PyContextRole::Task => ContextRole::Task,
            PyContextRole::Constraint => ContextRole::Constraint,
            PyContextRole::Finding => ContextRole::Finding,
            PyContextRole::Decision => ContextRole::Decision,
            PyContextRole::Warning => ContextRole::Warning,
            PyContextRole::Background => ContextRole::Background,
        };
        let rust_weight = match weight {
            PyContextWeight::Critical => ContextWeight::Critical,
            PyContextWeight::High => ContextWeight::High,
            PyContextWeight::Normal => ContextWeight::Normal,
            PyContextWeight::Background => ContextWeight::Background,
        };
        PyContextBlock {
            inner: RustBlock::new(rust_role, rust_weight, content, source),
        }
    }

    #[getter]
    pub fn token_count(&self) -> u32 { self.inner.token_count }

    #[getter]
    pub fn content(&self) -> &str { &self.inner.content }

    #[getter]
    pub fn id(&self) -> String { self.inner.id.to_string() }
}

#[pyclass(name = "SurgeonConfig")]
pub struct PySurgeonConfig {
    pub inner: SurgeonConfig,
}

#[pymethods]
impl PySurgeonConfig {
    #[new]
    #[pyo3(signature = (strategy="hybrid", target_tokens=8000, preserve_recent=5))]
    pub fn new(strategy: &str, target_tokens: u32, preserve_recent: usize) -> Self {
        let s = match strategy {
            "sliding_window" => CompressionStrategy::SlidingWindow { keep_recent: preserve_recent },
            "weighted_prune" => CompressionStrategy::WeightedPrune { target_tokens },
            _ => CompressionStrategy::Hybrid,
        };
        PySurgeonConfig {
            inner: SurgeonConfig { strategy: s, target_tokens, preserve_recent },
        }
    }
}

#[pyclass(name = "SurgeonResult")]
pub struct PySurgeonResult {
    pub blocks: Vec<PyContextBlock>,
    pub tokens_before: u32,
    pub tokens_after: u32,
    pub tokens_saved: u32,
}

#[pymethods]
impl PySurgeonResult {
    #[getter] pub fn tokens_saved(&self) -> u32 { self.tokens_saved }
    #[getter] pub fn tokens_before(&self) -> u32 { self.tokens_before }
    #[getter] pub fn tokens_after(&self) -> u32 { self.tokens_after }
    #[getter] pub fn blocks(&self) -> Vec<PyContextBlock> { self.blocks.clone() }
}

#[pyfunction]
pub fn compress_context(
    blocks: Vec<PyContextBlock>,
    config: &PySurgeonConfig,
) -> PySurgeonResult {
    let rust_blocks: Vec<RustBlock> = blocks.into_iter().map(|b| b.inner).collect();
    let result = rust_compress(&rust_blocks, &config.inner);
    PySurgeonResult {
        blocks: result.blocks.into_iter().map(|b| PyContextBlock { inner: b }).collect(),
        tokens_before: result.tokens_before,
        tokens_after: result.tokens_after,
        tokens_saved: result.tokens_saved,
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyContextWeight>()?;
    m.add_class::<PyContextRole>()?;
    m.add_class::<PyContextBlock>()?;
    m.add_class::<PySurgeonConfig>()?;
    m.add_class::<PySurgeonResult>()?;
    m.add_function(wrap_pyfunction!(compress_context, m)?)?;
    Ok(())
}
```

- [ ] **Step 4: Write the pymodule root**

Create `crates/truss-py/src/lib.rs`:

```rust
use pyo3::prelude::*;

mod context_bindings;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    context_bindings::register(m)?;
    Ok(())
}
```

- [ ] **Step 5: Write Python package init**

Create `python/truss/__init__.py`:

```python
from truss._core import (
    ContextBlock,
    ContextRole,
    ContextWeight,
    SurgeonConfig,
    SurgeonResult,
    compress_context,
)
from truss.session import Session, SessionReport

__all__ = [
    "ContextBlock", "ContextRole", "ContextWeight",
    "SurgeonConfig", "SurgeonResult", "compress_context",
    "Session", "SessionReport",
]
```

Create `python/truss/session.py`:

```python
from dataclasses import dataclass, field
from typing import Optional
from truss._core import SurgeonConfig, compress_context


@dataclass
class SessionReport:
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    cost_saved_usd: float
    budget_used_usd: float
    budget_limit_usd: Optional[float]

    def __str__(self) -> str:
        lines = [
            f"Context: {self.tokens_before:,} → {self.tokens_after:,} tokens (saved {self.tokens_saved:,})",
            f"Est. savings: ${self.cost_saved_usd:.4f}",
        ]
        if self.budget_limit_usd:
            lines.append(
                f"Budget: ${self.budget_used_usd:.4f} of ${self.budget_limit_usd:.2f} used"
            )
        return "\n".join(lines)


class Session:
    """
    Unified entry point for all truss modules.
    Phase 1: context compression + budget tracking only.
    """

    _COST_PER_1K_TOKENS = 0.001  # conservative default; ~Haiku pricing

    def __init__(
        self,
        budget_usd: Optional[float] = None,
        target_tokens: int = 8_000,
        compress_strategy: str = "hybrid",
        preserve_recent: int = 5,
    ):
        self._budget_usd = budget_usd
        self._spent_usd = 0.0
        self._tokens_before = 0
        self._tokens_after = 0
        self._surgeon_config = SurgeonConfig(
            strategy=compress_strategy,
            target_tokens=target_tokens,
            preserve_recent=preserve_recent,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    def compress(self, blocks):
        """Compress a list of ContextBlock objects. Returns SurgeonResult."""
        result = compress_context(blocks, self._surgeon_config)
        self._tokens_before += result.tokens_before
        self._tokens_after += result.tokens_after
        return result

    def record_usage(self, input_tokens: int, output_tokens: int, model: str = "unknown"):
        """Record token usage from an LLM call."""
        total = input_tokens + output_tokens
        cost = (total / 1000) * self._COST_PER_1K_TOKENS
        self._spent_usd += cost

    def report(self) -> SessionReport:
        """Return a summary of this session's activity."""
        tokens_saved = max(0, self._tokens_before - self._tokens_after)
        cost_saved = (tokens_saved / 1000) * self._COST_PER_1K_TOKENS
        return SessionReport(
            tokens_before=self._tokens_before,
            tokens_after=self._tokens_after,
            tokens_saved=tokens_saved,
            cost_saved_usd=cost_saved,
            budget_used_usd=self._spent_usd,
            budget_limit_usd=self._budget_usd,
        )
```

- [ ] **Step 6: Build the wheel with maturin**

```bash
cd crates/truss-py
pip install maturin
maturin develop
```

Expected: `✓ truss-ai installed in development mode`

- [ ] **Step 7: Verify Python import works**

```bash
python -c "from truss import Session, ContextBlock, ContextRole, ContextWeight; print('OK')"
```

Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add crates/truss-py/ python/
git commit -m "feat: add PyO3 bindings and Python Session object"
```

---

## Task 12: LangChain `TrussMemory` Adapter

**Files:**
- Create: `python/truss/adapters/langchain.py`
- Create: `python/truss/adapters/__init__.py`
- Create: `tests/integration/test_langchain_adapter.py`

- [ ] **Step 1: Write failing adapter test**

Create `tests/integration/test_langchain_adapter.py`:

```python
import pytest
from unittest.mock import MagicMock
from truss.adapters.langchain import TrussMemory
from truss._core import ContextBlock, ContextRole, ContextWeight


def test_truss_memory_saves_messages():
    memory = TrussMemory(target_tokens=1000, preserve_recent=3)
    memory.save_context(
        inputs={"input": "What is the weather?"},
        outputs={"output": "It is sunny."},
    )
    assert len(memory.blocks) == 2  # one for input, one for output


def test_truss_memory_compresses_on_load():
    memory = TrussMemory(target_tokens=50, preserve_recent=1)
    # Add enough blocks to trigger compression
    for i in range(20):
        memory.save_context(
            inputs={"input": f"Question {i}: " + "x" * 100},
            outputs={"output": f"Answer {i}: " + "y" * 100},
        )
    vars_ = memory.load_memory_variables({})
    # Compression should have reduced token count
    history = vars_["history"]
    assert isinstance(history, str)
    assert len(history) < 20 * 200  # should be compressed


def test_truss_memory_clear_empties_blocks():
    memory = TrussMemory()
    memory.save_context(inputs={"input": "hi"}, outputs={"output": "hello"})
    memory.clear()
    assert len(memory.blocks) == 0


def test_truss_memory_chat_memory_interface():
    """TrussMemory must satisfy LangChain's BaseChatMemory duck type."""
    memory = TrussMemory()
    assert hasattr(memory, "save_context")
    assert hasattr(memory, "load_memory_variables")
    assert hasattr(memory, "clear")
    assert hasattr(memory, "memory_key")
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/integration/test_langchain_adapter.py -v
```

Expected: `ImportError: No module named 'truss.adapters'`

- [ ] **Step 3: Implement TrussMemory**

Create `python/truss/adapters/__init__.py`:

```python
```

Create `python/truss/adapters/langchain.py`:

```python
from typing import Any, Dict, List, Optional
from truss._core import ContextBlock, ContextRole, ContextWeight, SurgeonConfig, compress_context


class TrussMemory:
    """
    Drop-in replacement for LangChain ConversationBufferMemory.
    Compresses context using truss-core's Rust surgeon before returning history.
    """

    memory_key: str = "history"

    def __init__(
        self,
        target_tokens: int = 8_000,
        preserve_recent: int = 5,
        strategy: str = "hybrid",
    ):
        self._config = SurgeonConfig(
            strategy=strategy,
            target_tokens=target_tokens,
            preserve_recent=preserve_recent,
        )
        self.blocks: List[ContextBlock] = []

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        human_text = inputs.get("input", "")
        ai_text = outputs.get("output", "")

        if human_text:
            self.blocks.append(
                ContextBlock(
                    ContextRole.Task,
                    ContextWeight.Normal,
                    f"Human: {human_text}",
                    "user",
                )
            )
        if ai_text:
            self.blocks.append(
                ContextBlock(
                    ContextRole.Finding,
                    ContextWeight.Normal,
                    f"AI: {ai_text}",
                    "assistant",
                )
            )

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        if not self.blocks:
            return {self.memory_key: ""}

        result = compress_context(self.blocks, self._config)
        self.blocks = result.blocks  # keep compressed blocks for next round

        history = "\n".join(b.content for b in result.blocks)
        return {self.memory_key: history}

    def clear(self) -> None:
        self.blocks = []
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/integration/test_langchain_adapter.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add python/truss/adapters/ tests/integration/test_langchain_adapter.py
git commit -m "feat: add LangChain TrussMemory adapter"
```

---

## Task 13: Hermes Reference Example

**Files:**
- Create: `examples/hermes-agent/main.py`
- Create: `examples/hermes-agent/requirements.txt`

- [ ] **Step 1: Write the reference example**

Create `examples/hermes-agent/requirements.txt`:

```
truss-ai
anthropic>=0.30
```

Create `examples/hermes-agent/main.py`:

```python
"""
Hermes Agent — Truss Phase 1 reference implementation.

Demonstrates: Session, context compression, budget tracking, report.
"""
import asyncio
import os
from truss import Session, ContextBlock, ContextRole, ContextWeight


async def main():
    # Build a realistic context: mix of weights and roles
    blocks = [
        ContextBlock(ContextRole.Task, ContextWeight.Critical,
                     "Research the cheapest cloud storage options for a 10TB dataset.", "user"),
        ContextBlock(ContextRole.Constraint, ContextWeight.Critical,
                     "Budget is $500/month maximum. Must be S3-compatible.", "user"),
        ContextBlock(ContextRole.Finding, ContextWeight.High,
                     "Backblaze B2 offers $6/TB/month with S3-compatible API.", "search-tool"),
        ContextBlock(ContextRole.Finding, ContextWeight.Normal,
                     "AWS S3 Standard is $23/TB/month. Has the widest ecosystem.", "search-tool"),
        ContextBlock(ContextRole.Background, ContextWeight.Background,
                     "Storage was invented in 1956. IBM 305 RAMAC was first.", "wikipedia"),
        ContextBlock(ContextRole.Background, ContextWeight.Background,
                     "Cloud computing history dates to the 1960s mainframe era.", "wikipedia"),
        ContextBlock(ContextRole.Finding, ContextWeight.Normal,
                     "Cloudflare R2 has zero egress fees, $15/TB/month storage.", "search-tool"),
        ContextBlock(ContextRole.Decision, ContextWeight.High,
                     "Shortlisted: B2, R2, Wasabi. Eliminating AWS on cost.", "agent"),
    ]

    async with Session(budget_usd=1.00, target_tokens=200, preserve_recent=2) as s:
        result = s.compress(blocks)

        # Simulate an LLM call (replace with real Anthropic/OpenAI call)
        simulated_input_tokens = result.tokens_after
        simulated_output_tokens = 80
        s.record_usage(simulated_input_tokens, simulated_output_tokens, model="claude-haiku")

        print("=== Compression Result ===")
        print(f"Kept {len(result.blocks)} of {len(blocks)} blocks")
        for b in result.blocks:
            print(f"  [{b.content[:60]}...]")

        print("\n=== Session Report ===")
        print(s.report())


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run the example**

```bash
cd examples/hermes-agent
pip install -r requirements.txt
python main.py
```

Expected output:

```
=== Compression Result ===
Kept N of 8 blocks
  [Task: Research the cheapest cloud storage options...]
  ...

=== Session Report ===
Context: 8 → N tokens (saved M tokens)
Est. savings: $0.00XX
Budget: $0.00XX of $1.00 used
```

- [ ] **Step 3: Commit**

```bash
git add examples/hermes-agent/
git commit -m "example: add hermes-agent Phase 1 reference implementation"
```

---

## Task 14: Full Integration Test Suite

**Files:**
- Create: `tests/integration/test_session.py`
- Create: `tests/fixtures/sample_blocks.py`

- [ ] **Step 1: Write session integration tests**

Create `tests/fixtures/sample_blocks.py`:

```python
from truss._core import ContextBlock, ContextRole, ContextWeight


def make_block(weight: ContextWeight, content: str, source: str = "test") -> ContextBlock:
    return ContextBlock(ContextRole.Finding, weight, content, source)


def large_context(n: int = 50) -> list:
    """Return n blocks with mixed weights."""
    weights = [ContextWeight.Critical, ContextWeight.High, ContextWeight.Normal, ContextWeight.Background]
    return [
        make_block(weights[i % 4], f"Block {i}: " + "content " * 20, f"source-{i}")
        for i in range(n)
    ]
```

Create `tests/integration/test_session.py`:

```python
import asyncio
import pytest
from truss import Session
from tests.fixtures.sample_blocks import large_context, make_block
from truss._core import ContextWeight


def test_session_report_has_savings():
    session = Session(target_tokens=200, preserve_recent=2)
    blocks = large_context(50)
    session.compress(blocks)
    report = session.report()
    assert report.tokens_saved > 0


def test_session_tracks_budget():
    session = Session(budget_usd=1.0)
    session.record_usage(500, 100, model="test")
    report = session.report()
    assert report.budget_used_usd > 0
    assert report.budget_limit_usd == 1.0


def test_session_compress_preserves_critical_blocks():
    from truss._core import ContextRole, ContextBlock
    critical = ContextBlock(ContextRole.Task, ContextWeight.Critical, "must keep this", "agent")
    bg_blocks = [make_block(ContextWeight.Background, "noise " * 50) for _ in range(20)]
    all_blocks = [critical] + bg_blocks

    session = Session(target_tokens=50, preserve_recent=0)
    result = session.compress(all_blocks)
    ids = [b.id for b in result.blocks]
    assert critical.id in ids


@pytest.mark.asyncio
async def test_session_async_context_manager():
    async with Session(budget_usd=0.50) as s:
        blocks = large_context(10)
        result = s.compress(blocks)
        assert result.tokens_after <= result.tokens_before
```

- [ ] **Step 2: Install pytest-asyncio**

```bash
pip install pytest pytest-asyncio
```

Create `pytest.ini` at repo root:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 3: Run full integration suite**

```bash
pytest tests/ -v
```

Expected: all tests pass, including `test_langchain_adapter.py` and `test_session.py`

- [ ] **Step 4: Commit**

```bash
git add tests/ pytest.ini
git commit -m "test: add integration test suite for Session and LangChain adapter"
```

---

## Self-Review Against Spec

**Spec section → Task coverage:**

| Spec requirement | Covered in |
|---|---|
| `truss-core` Cargo workspace | Task 1 |
| `ContextBlock` + `AgentEnvelope` with serde | Task 2 |
| Context surgeon: SlidingWindow + WeightedPrune | Task 3 |
| Token counter | Task 2 (`estimate_tokens`), Task 3 (used throughout) |
| Budget ledger: InMemory + SQLite | Tasks 4 + 6 |
| Circuit breaker | Task 7 |
| Checkpoint: save/load/rollback with SQLite | Task 8 (InMemory) + spec gap: SQLite checkpoint deferred to follow-on task |
| PyO3 bindings | Task 11 |
| Maturin build | Task 11 |
| `pip install truss-ai` works | Task 11 |
| Python Session object | Task 11 |
| LangChain TrussMemory adapter | Task 12 |
| `examples/hermes-agent/` | Task 13 |
| Criterion benchmarks | Task 10 |
| Fence (added from gap analysis) | Task 9 |

**Gaps still open (not in Phase 1 scope):**

- `SqliteCheckpointStore` — Task 8 has InMemory only; SQLite checkpoint needs its own follow-on task (30min)
- `handoff_bindings.rs` + `budget_bindings.rs` — PyO3 bindings for handoff/budget not fully implemented; Task 11 covers context bindings only. Add as follow-on before Phase 2.
- `score_relevance` and `detect_contradiction` exposed to Python — implemented in Rust (Task 3), not yet bound in PyO3.
- SemanticDedup and Summarise strategies — deferred to Phase 2 per spec
- Vercel AI / Mastra / TypeScript bindings — Phase 2

**Placeholder scan:** No TBD, TODO, or placeholder text found in code blocks. All functions have implementations.

**Type consistency check:** `ContextWeight` enum used in Task 3 (`weighted_prune` comparison with `ContextWeight::High`) matches definition in Task 2. `AgentEnvelope.checkpoint_id: Option<Uuid>` defined in Task 2, used correctly as `None` default. `BudgetCarve` defined in Task 5 and used only in Task 5.

---

Plan complete and saved to `docs/superpowers/plans/2026-06-03-truss-phase1.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
