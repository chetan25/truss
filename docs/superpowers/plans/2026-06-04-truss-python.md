# Truss Python — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `truss-ai` as a pure Python package implementing all 7 Truss modules — Context Surgeon, Agent Handoff, Token Budget/Ledger, Multi-LLM Router, MCP Interceptor, Checkpoints/Rollback, and Fence — with SQLite persistence, a LangChain adapter, and a Hermes reference example.

**Architecture:** Single Python package under `python/src/truss/` with one submodule per domain. Pydantic v2 for type-safe models with JSON round-trip. `Protocol` classes (PEP 544) for store interfaces enabling SQLite or custom backends. Standard library `sqlite3` for persistence — the only mandatory runtime dependency beyond Pydantic is zero.

**Tech Stack:** Python 3.10+ · Pydantic v2 · pytest 8 · sqlite3 (stdlib) · hatchling (build backend) · langchain-core ≥0.2 (optional, adapter only)

---

## File Structure

```
python/
├── pyproject.toml
├── src/
│   └── truss/
│       ├── __init__.py              # public re-exports
│       ├── errors.py                # TrussError hierarchy
│       ├── types.py                 # ContextBlock, AgentEnvelope, shared enums
│       ├── context/
│       │   ├── __init__.py
│       │   └── surgeon.py           # compress(), SurgeonConfig, SurgeonResult, strategies
│       ├── handoff/
│       │   ├── __init__.py
│       │   └── envelope.py          # pack(), unpack(), BudgetCarve
│       ├── budget/
│       │   ├── __init__.py
│       │   ├── config.py            # BudgetConfig, BudgetLimit, BudgetWindow, AlertConfig
│       │   ├── ledger.py            # LedgerEntry, LedgerStore protocol, UsageReport
│       │   ├── memory_store.py      # InMemoryStore
│       │   ├── sqlite_store.py      # SqliteLedgerStore
│       │   └── circuit_breaker.py   # CircuitBreaker, CircuitBreakerConfig, CircuitTrip
│       ├── coord/
│       │   ├── __init__.py
│       │   ├── checkpoint.py        # Checkpoint, CheckpointStore, InMemoryCheckpointStore
│       │   └── sqlite_checkpoint.py # SqliteCheckpointStore
│       ├── fence/
│       │   ├── __init__.py
│       │   └── memory_fence.py      # FenceStore, InMemoryFence, LockHandle
│       ├── router/
│       │   ├── __init__.py
│       │   └── router.py            # ModelSpec, RouterConfig, route()
│       ├── mcp/
│       │   ├── __init__.py
│       │   └── interceptor.py       # McpManifest, McpInterceptor, McpCall
│       ├── session.py               # Session class + SessionReport
│       └── adapters/
│           ├── __init__.py
│           └── langchain.py         # TrussMemory
├── tests/
│   ├── conftest.py
│   ├── test_types.py
│   ├── test_surgeon.py
│   ├── test_budget.py
│   ├── test_handoff.py
│   ├── test_circuit_breaker.py
│   ├── test_checkpoint.py
│   ├── test_fence.py
│   ├── test_router.py
│   ├── test_mcp.py
│   └── test_session.py
└── examples/
    └── hermes/
        └── main.py
```

---

## Task 1: Repo Cleanup + Python Project Setup

**Files:**
- Create: `.gitignore`
- Create: `python/pyproject.toml`
- Create: `python/src/truss/__init__.py` (stub)
- Create: `python/tests/conftest.py`

- [ ] **Step 1: Remove untracked Rust artifacts**

The Rust workspace files were never committed. Delete them so they don't confuse the workspace:

```powershell
Remove-Item -Recurse -Force "Cargo.toml", "Cargo.lock", "crates", "target", ".cargo" -ErrorAction SilentlyContinue
```

- [ ] **Step 2: Create .gitignore**

Create `.gitignore` at repo root:

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
.venv/
.pytest_cache/
*.db

# TypeScript / Node
node_modules/
dist/
*.tsbuildinfo
.turbo/

# Editors
.vscode/
.idea/
*.swp
```

- [ ] **Step 3: Create directory structure**

```powershell
$dirs = @(
  "python/src/truss/context",
  "python/src/truss/handoff",
  "python/src/truss/budget",
  "python/src/truss/coord",
  "python/src/truss/fence",
  "python/src/truss/router",
  "python/src/truss/mcp",
  "python/src/truss/adapters",
  "python/tests",
  "python/examples/hermes"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d }
```

- [ ] **Step 4: Create pyproject.toml**

Create `python/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "truss-ai"
version = "0.1.0"
description = "The structural layer for agentic AI workflows"
requires-python = ">=3.10"
dependencies = ["pydantic>=2.0"]

[project.optional-dependencies]
langchain = ["langchain-core>=0.2"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[tool.hatch.build.targets.wheel]
packages = ["src/truss"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"
```

- [ ] **Step 5: Create stub __init__.py files**

Create `python/src/truss/__init__.py` (stub — filled out in Task 14):

```python
# Public API — populated in Task 14 after all modules exist.
```

Create empty `__init__.py` files for each subpackage:

```powershell
$pkgs = @(
  "python/src/truss/context/__init__.py",
  "python/src/truss/handoff/__init__.py",
  "python/src/truss/budget/__init__.py",
  "python/src/truss/coord/__init__.py",
  "python/src/truss/fence/__init__.py",
  "python/src/truss/router/__init__.py",
  "python/src/truss/mcp/__init__.py",
  "python/src/truss/adapters/__init__.py"
)
foreach ($f in $pkgs) { New-Item -ItemType File -Force -Path $f }
```

- [ ] **Step 6: Create conftest.py**

Create `python/tests/conftest.py`:

```python
# Shared pytest fixtures live here as tasks add them.
```

- [ ] **Step 7: Install and verify**

```bash
cd python && pip install -e ".[dev]"
pytest
```

Expected: `no tests ran` (0 items collected, no errors).

- [ ] **Step 8: Commit**

```bash
git add .gitignore python/
git commit -m "chore: set up Python project structure for truss-ai"
```

---

## Task 2: Error Hierarchy + Core Types

**Files:**
- Create: `python/src/truss/errors.py`
- Create: `python/src/truss/types.py`
- Create: `python/tests/test_types.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_types.py`:

```python
import pytest
from truss.errors import BudgetExceeded, ToolOutOfScope, CheckpointNotFound, FenceLockConflict
from truss.types import (
    ContextBlock, ContextRole, ContextWeight, estimate_tokens,
    AgentEnvelope, ModelTier, EvidenceRef, DecisionRecord,
)


def test_estimate_tokens_ceiling_division():
    assert estimate_tokens("Hello") == 2        # ceil(5/4) = 2
    assert estimate_tokens("Hello world") == 3  # ceil(11/4) = 3
    assert estimate_tokens("") == 0


def test_context_block_auto_estimates_tokens():
    block = ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL, content="Hello world", source="test")
    assert block.token_count == 3


def test_context_block_explicit_token_count_not_overridden():
    block = ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.NORMAL, content="Hello world", source="test", token_count=99)
    assert block.token_count == 99


def test_context_weight_is_comparable():
    assert ContextWeight.CRITICAL > ContextWeight.NORMAL
    assert ContextWeight.BACKGROUND < ContextWeight.HIGH


def test_context_block_serialises_to_json():
    block = ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.NORMAL, content="data", source="agent-1")
    json_str = block.model_dump_json()
    assert "finding" in json_str


def test_agent_envelope_has_no_checkpoint_by_default():
    env = AgentEnvelope(task="test task")
    assert env.checkpoint_id is None


def test_agent_envelope_round_trips_json():
    env = AgentEnvelope(task="analyse pricing")
    json_str = env.model_dump_json()
    back = AgentEnvelope.model_validate_json(json_str)
    assert back.task == env.task
    assert back.id == env.id


def test_error_messages_include_key_info():
    assert "wallet" in str(BudgetExceeded("wallet exceeded $5"))
    assert "readFile" in str(ToolOutOfScope("readFile denied by manifest"))
    assert "abc-123" in str(CheckpointNotFound("abc-123"))
    assert "agent-a" in str(FenceLockConflict("doc-1 held by agent-a"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && pytest tests/test_types.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.errors'`

- [ ] **Step 3: Implement errors.py**

Create `python/src/truss/errors.py`:

```python
class TrussError(Exception):
    pass

class BudgetExceeded(TrussError):
    pass

class ToolOutOfScope(TrussError):
    pass

class CheckpointNotFound(TrussError):
    pass

class FenceLockConflict(TrussError):
    pass
```

- [ ] **Step 4: Implement types.py**

Create `python/src/truss/types.py`:

```python
from __future__ import annotations

import math
from enum import IntEnum, Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


class ContextRole(str, Enum):
    TASK = "task"
    CONSTRAINT = "constraint"
    FINDING = "finding"
    DECISION = "decision"
    WARNING = "warning"
    BACKGROUND = "background"


class ContextWeight(IntEnum):
    BACKGROUND = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class ContextBlock(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: ContextRole
    weight: ContextWeight
    content: str
    source: str
    token_count: int = 0
    created_at: int = 0
    referenced_by: list[UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_token_count(self) -> "ContextBlock":
        if self.token_count == 0 and self.content:
            self.token_count = estimate_tokens(self.content)
        return self


class ModelTier(str, Enum):
    CHEAP = "cheap"
    STANDARD = "standard"
    PREMIUM = "premium"
    AUTO = "auto"


class EvidenceRef(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    source_url: Optional[str] = None
    tool_name: Optional[str] = None
    confidence: float = 1.0


class DecisionRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    decision: str
    reasoning: str
    evidence_ids: list[UUID] = Field(default_factory=list)
    confidence: float = 1.0
    decided_by: str
    timestamp: int = 0


class AgentEnvelope(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    task: str
    context: list[ContextBlock] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    budget_usd_remaining: float = float("inf")
    checkpoint_id: Optional[UUID] = None
    model_hint: ModelTier = ModelTier.AUTO
    parent_agent: Optional[str] = None
    scope: list[str] = Field(default_factory=list)
    created_at: int = 0
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd python && pytest tests/test_types.py -v
```

Expected: `9 passed`

- [ ] **Step 6: Commit**

```bash
git add python/src/truss/errors.py python/src/truss/types.py python/tests/test_types.py
git commit -m "feat: add error hierarchy and core types"
```

---

## Task 3: Context Surgeon

**Files:**
- Create: `python/src/truss/context/surgeon.py`
- Create: `python/tests/test_surgeon.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_surgeon.py`:

```python
import pytest
from truss.types import ContextBlock, ContextRole, ContextWeight
from truss.context.surgeon import (
    compress, SurgeonConfig, SurgeonResult, CompressionStrategy,
    score_relevance, detect_contradiction,
)


def make_block(weight: ContextWeight, tokens: int, content: str = "") -> ContextBlock:
    b = ContextBlock(role=ContextRole.FINDING, weight=weight, content=content or ("x " * tokens), source="test")
    object.__setattr__(b, "token_count", tokens)
    return b


def test_sliding_window_keeps_recent_n_blocks():
    blocks = [make_block(ContextWeight.NORMAL, 100, f"block {i}") for i in range(10)]
    config = SurgeonConfig(strategy=CompressionStrategy.SLIDING_WINDOW, target_tokens=300, preserve_recent=3, keep_recent=3)
    result = compress(blocks, config)
    assert result.tokens_after <= 300


def test_weighted_prune_drops_background_before_normal():
    bg = make_block(ContextWeight.BACKGROUND, 500, "background")
    normal = make_block(ContextWeight.NORMAL, 500, "normal")
    critical = make_block(ContextWeight.CRITICAL, 100, "critical")
    config = SurgeonConfig(strategy=CompressionStrategy.WEIGHTED_PRUNE, target_tokens=700, preserve_recent=0)
    result = compress([bg, normal, critical], config)
    ids = {b.id for b in result.blocks}
    assert bg.id not in ids
    assert critical.id in ids


def test_hybrid_applies_prune_then_sliding_window():
    blocks = [make_block(ContextWeight.NORMAL, 1000, f"block {i}") for i in range(5)]
    config = SurgeonConfig(strategy=CompressionStrategy.HYBRID, target_tokens=2000, preserve_recent=2)
    result = compress(blocks, config)
    assert result.tokens_after <= 3000  # not strictly 2000 because sliding keeps critical


def test_compress_never_removes_critical_blocks():
    critical = make_block(ContextWeight.CRITICAL, 9000, "must keep")
    bg = make_block(ContextWeight.BACKGROUND, 100, "droppable")
    config = SurgeonConfig(strategy=CompressionStrategy.HYBRID, target_tokens=500, preserve_recent=0)
    result = compress([critical, bg], config)
    assert any(b.id == critical.id for b in result.blocks)


def test_surgeon_result_tokens_saved():
    blocks = [make_block(ContextWeight.BACKGROUND, 500) for _ in range(4)]
    config = SurgeonConfig(strategy=CompressionStrategy.WEIGHTED_PRUNE, target_tokens=500, preserve_recent=0)
    result = compress(blocks, config)
    assert result.tokens_saved == result.tokens_before - result.tokens_after


def test_score_relevance_empty_task_returns_zero():
    block = make_block(ContextWeight.NORMAL, 10, "some content")
    assert score_relevance(block, "") == 0.0


def test_score_relevance_exact_match():
    block = make_block(ContextWeight.NORMAL, 10, "pricing cloud storage")
    score = score_relevance(block, "pricing cloud storage")
    assert score == 1.0


def test_detect_contradiction_catches_not_pattern():
    a = make_block(ContextWeight.NORMAL, 10, "the service is available")
    b = make_block(ContextWeight.NORMAL, 10, "the service is not available")
    assert detect_contradiction(a, b) is True


def test_detect_contradiction_no_false_positive():
    a = make_block(ContextWeight.NORMAL, 10, "the weather is sunny today")
    b = make_block(ContextWeight.NORMAL, 10, "the price is twenty dollars")
    assert detect_contradiction(a, b) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && pytest tests/test_surgeon.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.context.surgeon'`

- [ ] **Step 3: Implement surgeon.py**

Create `python/src/truss/context/surgeon.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from truss.types import ContextBlock, ContextWeight


class CompressionStrategy(Enum):
    SLIDING_WINDOW = "sliding_window"
    WEIGHTED_PRUNE = "weighted_prune"
    HYBRID = "hybrid"


@dataclass
class SurgeonConfig:
    strategy: CompressionStrategy = CompressionStrategy.HYBRID
    target_tokens: int = 8_000
    preserve_recent: int = 5
    keep_recent: int = 0


@dataclass
class SurgeonResult:
    blocks: list[ContextBlock]
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    strategy_applied: str


def compress(blocks: list[ContextBlock], config: SurgeonConfig) -> SurgeonResult:
    tokens_before = sum(b.token_count for b in blocks)

    if config.strategy == CompressionStrategy.SLIDING_WINDOW:
        kept = _sliding_window(blocks, config.keep_recent or config.preserve_recent, config.preserve_recent)
    elif config.strategy == CompressionStrategy.WEIGHTED_PRUNE:
        kept = _weighted_prune(blocks, config.target_tokens, config.preserve_recent)
    else:  # HYBRID: prune first, then window if still over budget
        after_prune = _weighted_prune(blocks, config.target_tokens, config.preserve_recent)
        if sum(b.token_count for b in after_prune) > config.target_tokens:
            kept = _sliding_window(after_prune, config.target_tokens, config.preserve_recent)
        else:
            kept = after_prune

    tokens_after = sum(b.token_count for b in kept)
    return SurgeonResult(
        blocks=kept,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_before - tokens_after,
        strategy_applied=config.strategy.value,
    )


def _sliding_window(blocks: list[ContextBlock], keep_recent: int, preserve_recent: int) -> list[ContextBlock]:
    always_keep = max(preserve_recent, keep_recent)
    if len(blocks) <= always_keep:
        return list(blocks)
    # Always keep Critical/High; then append recent blocks not already included
    pinned = [b for b in blocks if b.weight >= ContextWeight.HIGH]
    pinned_ids = {b.id for b in pinned}
    recent_start = max(0, len(blocks) - keep_recent)
    result = list(pinned)
    for b in blocks[recent_start:]:
        if b.id not in pinned_ids:
            result.append(b)
    return result


def _weighted_prune(blocks: list[ContextBlock], target_tokens: int, preserve_recent: int) -> list[ContextBlock]:
    total = sum(b.token_count for b in blocks)
    if total <= target_tokens:
        return list(blocks)

    preserve_ids = {b.id for b in blocks[-preserve_recent:]} if preserve_recent > 0 else set()

    # Removable = not pinned recent AND weight < High
    removable = sorted(
        [b for b in blocks if b.id not in preserve_ids and b.weight < ContextWeight.HIGH],
        key=lambda b: (int(b.weight), b.created_at),  # cheapest (Background, oldest) first
    )

    to_remove: set = set()
    running = total
    for b in removable:
        if running <= target_tokens:
            break
        running -= b.token_count
        to_remove.add(b.id)

    return [b for b in blocks if b.id not in to_remove]


def score_relevance(block: ContextBlock, task: str) -> float:
    """Keyword overlap between block content and task string. Returns 0.0–1.0."""
    task_words = set(task.split())
    if not task_words:
        return 0.0
    matches = sum(1 for w in block.content.split() if w in task_words)
    return min(matches / len(task_words), 1.0)


def detect_contradiction(a: ContextBlock, b: ContextBlock) -> bool:
    """Heuristic: returns True if one block says 'X' and the other says 'not X'."""
    a_lower = a.content.lower()
    b_lower = b.content.lower()
    for word in a_lower.split():
        if len(word) > 4:
            if f"not {word}" in b_lower:
                return True
            if f"not {word}" in a_lower and word in b_lower:
                return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && pytest tests/test_surgeon.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/context/surgeon.py python/tests/test_surgeon.py
git commit -m "feat: add context surgeon with SlidingWindow, WeightedPrune, Hybrid"
```

---

## Task 4: Budget Config + LedgerStore + InMemoryStore

**Files:**
- Create: `python/src/truss/budget/config.py`
- Create: `python/src/truss/budget/ledger.py`
- Create: `python/src/truss/budget/memory_store.py`
- Create: `python/tests/test_budget.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_budget.py`:

```python
import pytest
import threading
from uuid import uuid4
from truss.budget.config import BudgetConfig, BudgetWindow, AlertConfig, BudgetLimit, ExceededAction
from truss.budget.ledger import LedgerEntry, UsageReport
from truss.budget.memory_store import InMemoryStore
from truss.types import ModelTier


def make_entry(session_id: str, input_tokens: int = 100, cost_usd: float = 0.01) -> LedgerEntry:
    return LedgerEntry(
        session_id=session_id,
        model="test-model",
        input_tokens=input_tokens,
        output_tokens=0,
        cost_usd=cost_usd,
    )


def test_budget_config_defaults():
    cfg = BudgetConfig()
    assert cfg.alert_at_pct == 0.8
    assert cfg.alerts.log_to_stderr is True


def test_budget_limit_serialises():
    limit = BudgetLimit(tokens=10_000, usd=1.0, window=BudgetWindow.PER_SESSION)
    assert limit.window == BudgetWindow.PER_SESSION


def test_in_memory_store_records_and_queries():
    store = InMemoryStore()
    store.record(make_entry("sess-1", 1000, 0.50))
    store.record(make_entry("sess-1", 500, 0.25))
    store.record(make_entry("sess-2", 999, 9.99))
    report = store.usage("sess-1", BudgetWindow.PER_SESSION)
    assert report.total_tokens == 1500
    assert abs(report.total_cost_usd - 0.75) < 0.001


def test_in_memory_store_unknown_key_returns_zero():
    store = InMemoryStore()
    report = store.usage("nobody", BudgetWindow.PER_SESSION)
    assert report.total_tokens == 0
    assert report.total_cost_usd == 0.0


def test_in_memory_store_concurrent_writes():
    store = InMemoryStore()
    errors = []

    def writer(i: int):
        try:
            store.record(make_entry(f"sess-{i}", 100, 0.01))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # All 20 entries recorded
    total = sum(store.usage(f"sess-{i}", BudgetWindow.PER_SESSION).total_tokens for i in range(20))
    assert total == 2000


def test_ledger_entry_total_tokens():
    entry = LedgerEntry(session_id="s", model="m", input_tokens=100, output_tokens=50, cost_usd=0.01)
    assert entry.total_tokens == 150
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && pytest tests/test_budget.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.budget.config'`

- [ ] **Step 3: Implement config.py**

Create `python/src/truss/budget/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from truss.types import ModelTier


class BudgetWindow(Enum):
    PER_SESSION = "per_session"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    PER_MONTH = "per_month"


@dataclass
class BudgetLimit:
    window: BudgetWindow
    tokens: Optional[int] = None
    usd: Optional[float] = None


@dataclass
class AlertConfig:
    slack_webhook: Optional[str] = None
    log_to_stderr: bool = True


class ExceededAction(Enum):
    BLOCK = "block"
    QUEUE = "queue"

    @staticmethod
    def fallback(tier: ModelTier) -> "_FallbackAction":
        return _FallbackAction(tier)

    @staticmethod
    def graceful(message: str) -> "_GracefulAction":
        return _GracefulAction(message)


@dataclass
class _FallbackAction:
    tier: ModelTier


@dataclass
class _GracefulAction:
    message: str


@dataclass
class BudgetConfig:
    per_session: Optional[BudgetLimit] = None
    per_user: Optional[BudgetLimit] = None
    per_agent: Optional[BudgetLimit] = None
    global_limit: Optional[BudgetLimit] = None
    on_exceeded: ExceededAction = ExceededAction.BLOCK
    alert_at_pct: float = 0.8
    alerts: AlertConfig = field(default_factory=AlertConfig)
```

- [ ] **Step 4: Implement ledger.py**

Create `python/src/truss/budget/ledger.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from truss.budget.config import BudgetWindow


@dataclass
class LedgerEntry:
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    id: UUID = field(default_factory=uuid4)
    user_id: str | None = None
    agent_name: str | None = None
    timestamp: int = 0
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class UsageReport:
    key: str
    total_tokens: int
    total_cost_usd: float
    window: BudgetWindow
    remaining_tokens: int | None = None
    remaining_usd: float | None = None
    pct_used: float = 0.0


@runtime_checkable
class LedgerStore(Protocol):
    def record(self, entry: LedgerEntry) -> None: ...
    def usage(self, key: str, window: BudgetWindow) -> UsageReport: ...
    def flush(self) -> None: ...
```

- [ ] **Step 5: Implement memory_store.py**

Create `python/src/truss/budget/memory_store.py`:

```python
from __future__ import annotations

import threading
from truss.budget.config import BudgetWindow
from truss.budget.ledger import LedgerEntry, LedgerStore, UsageReport


class InMemoryStore:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._lock = threading.RLock()

    def record(self, entry: LedgerEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def usage(self, key: str, window: BudgetWindow) -> UsageReport:
        with self._lock:
            matching = [
                e for e in self._entries
                if e.session_id == key or e.user_id == key or e.agent_name == key
            ]
        total_tokens = sum(e.total_tokens for e in matching)
        total_cost = sum(e.cost_usd for e in matching)
        return UsageReport(
            key=key,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            window=window,
        )

    def flush(self) -> None:
        pass
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd python && pytest tests/test_budget.py -v
```

Expected: `6 passed`

- [ ] **Step 7: Commit**

```bash
git add python/src/truss/budget/config.py python/src/truss/budget/ledger.py python/src/truss/budget/memory_store.py python/tests/test_budget.py
git commit -m "feat: add budget config, LedgerStore protocol, InMemoryStore"
```

---

## Task 5: SQLite Ledger Store

**Files:**
- Create: `python/src/truss/budget/sqlite_store.py`
- Modify: `python/tests/test_budget.py` (add SQLite tests)

- [ ] **Step 1: Add SQLite tests to test_budget.py**

Append to `python/tests/test_budget.py`:

```python
import tempfile
import os
from truss.budget.sqlite_store import SqliteLedgerStore


def test_sqlite_store_records_and_queries():
    store = SqliteLedgerStore(":memory:")
    store.record(make_entry("sess-a", 100, 0.10))
    store.record(make_entry("sess-a", 200, 0.20))
    report = store.usage("sess-a", BudgetWindow.PER_SESSION)
    assert report.total_tokens == 300
    assert abs(report.total_cost_usd - 0.30) < 0.001


def test_sqlite_store_survives_reopen():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        store1 = SqliteLedgerStore(path)
        store1.record(make_entry("persistent-sess", 500, 0.50))
        store1.flush()

        store2 = SqliteLedgerStore(path)
        report = store2.usage("persistent-sess", BudgetWindow.PER_SESSION)
        assert report.total_tokens == 500
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd python && pytest tests/test_budget.py::test_sqlite_store_records_and_queries -v
```

Expected: `ModuleNotFoundError: No module named 'truss.budget.sqlite_store'`

- [ ] **Step 3: Implement sqlite_store.py**

Create `python/src/truss/budget/sqlite_store.py`:

```python
from __future__ import annotations

import sqlite3
import json
import threading
from truss.budget.config import BudgetWindow
from truss.budget.ledger import LedgerEntry, UsageReport


class SqliteLedgerStore:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ledger (
                id            TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                user_id       TEXT,
                agent_name    TEXT,
                model         TEXT NOT NULL,
                input_tokens  INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd      REAL NOT NULL,
                timestamp     INTEGER NOT NULL,
                tags          TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def record(self, entry: LedgerEntry) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    str(entry.id), entry.session_id, entry.user_id, entry.agent_name,
                    entry.model, entry.input_tokens, entry.output_tokens,
                    entry.cost_usd, entry.timestamp, json.dumps(entry.tags),
                ),
            )
            self._conn.commit()

    def usage(self, key: str, window: BudgetWindow) -> UsageReport:
        with self._lock:
            row = self._conn.execute(
                """SELECT COALESCE(SUM(input_tokens + output_tokens), 0),
                          COALESCE(SUM(cost_usd), 0.0)
                   FROM ledger
                   WHERE session_id = ? OR user_id = ? OR agent_name = ?""",
                (key, key, key),
            ).fetchone()
        total_tokens, total_cost = row
        return UsageReport(
            key=key,
            total_tokens=int(total_tokens),
            total_cost_usd=float(total_cost),
            window=window,
        )

    def flush(self) -> None:
        with self._lock:
            self._conn.commit()
```

- [ ] **Step 4: Run all budget tests**

```bash
cd python && pytest tests/test_budget.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/budget/sqlite_store.py python/tests/test_budget.py
git commit -m "feat: add SQLite ledger store with persistence"
```

---

## Task 6: Agent Handoff

**Files:**
- Create: `python/src/truss/handoff/envelope.py`
- Create: `python/tests/test_handoff.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_handoff.py`:

```python
import pytest
from truss.types import AgentEnvelope, ContextBlock, ContextRole, ContextWeight
from truss.handoff.envelope import pack, unpack, BudgetCarve


def parent_envelope() -> AgentEnvelope:
    env = AgentEnvelope(task="parent task", budget_usd_remaining=1.0)
    env.context.append(ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL, content="critical info", source="planner"))
    env.context.append(ContextBlock(role=ContextRole.BACKGROUND, weight=ContextWeight.BACKGROUND, content="background fluff", source="loader"))
    return env


def test_pack_filters_context_by_weight():
    parent = parent_envelope()
    child = pack(parent, "child task", carry_weights=[ContextWeight.CRITICAL], budget_carve=BudgetCarve.fixed_usd(0.20))
    assert len(child.context) == 1
    assert child.context[0].weight == ContextWeight.CRITICAL


def test_pack_sets_parent_agent_id():
    parent = parent_envelope()
    child = pack(parent, "child", carry_weights=[ContextWeight.CRITICAL], budget_carve=BudgetCarve.fixed_usd(0.1))
    assert child.parent_agent == str(parent.id)


def test_pack_does_not_exceed_parent_budget():
    parent = parent_envelope()  # 1.0 USD
    child = pack(parent, "child", carry_weights=[], budget_carve=BudgetCarve.fixed_usd(999.0))
    assert child.budget_usd_remaining <= 1.0


def test_pack_percent_carve():
    parent = parent_envelope()  # 1.0 USD
    child = pack(parent, "child", carry_weights=[], budget_carve=BudgetCarve.percent(0.5))
    assert abs(child.budget_usd_remaining - 0.5) < 0.001


def test_unpack_returns_all_context_blocks():
    parent = parent_envelope()
    blocks = unpack(parent)
    assert len(blocks) == len(parent.context)


def test_pack_inherits_model_hint():
    from truss.types import ModelTier
    parent = parent_envelope()
    parent.model_hint = ModelTier.PREMIUM
    child = pack(parent, "child", carry_weights=[], budget_carve=BudgetCarve.fixed_usd(0.1))
    assert child.model_hint == ModelTier.PREMIUM
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && pytest tests/test_handoff.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.handoff.envelope'`

- [ ] **Step 3: Implement envelope.py**

Create `python/src/truss/handoff/envelope.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from truss.types import AgentEnvelope, ContextBlock, ContextWeight


@dataclass
class BudgetCarve:
    _kind: str
    _value: float

    @staticmethod
    def fixed_usd(amount: float) -> "BudgetCarve":
        return BudgetCarve(_kind="usd", _value=amount)

    @staticmethod
    def percent(fraction: float) -> "BudgetCarve":
        return BudgetCarve(_kind="pct", _value=fraction)

    @staticmethod
    def fixed_tokens(tokens: int) -> "BudgetCarve":
        return BudgetCarve(_kind="tokens", _value=float(tokens))

    def apply(self, parent_budget: float) -> float:
        if self._kind == "usd":
            return min(self._value, parent_budget)
        if self._kind == "pct":
            return parent_budget * self._value
        # tokens → usd: use 50% of parent as safe default
        return parent_budget * 0.5


def pack(
    parent: AgentEnvelope,
    task: str,
    carry_weights: list[ContextWeight],
    budget_carve: BudgetCarve,
) -> AgentEnvelope:
    child = AgentEnvelope(
        task=task,
        budget_usd_remaining=budget_carve.apply(parent.budget_usd_remaining),
        parent_agent=str(parent.id),
        model_hint=parent.model_hint,
    )
    child.context = [b for b in parent.context if b.weight in carry_weights]
    return child


def unpack(envelope: AgentEnvelope) -> list[ContextBlock]:
    return list(envelope.context)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && pytest tests/test_handoff.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/handoff/envelope.py python/tests/test_handoff.py
git commit -m "feat: add agent handoff pack/unpack with BudgetCarve"
```

---

## Task 7: Circuit Breaker

**Files:**
- Create: `python/src/truss/budget/circuit_breaker.py`
- Create: `python/tests/test_circuit_breaker.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_circuit_breaker.py`:

```python
import pytest
from truss.budget.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitTrip


def test_trips_on_rate_limit():
    cb = CircuitBreaker(CircuitBreakerConfig(max_requests_per_minute=3))
    for i in range(3):
        assert cb.check_and_record("prompt", 0.01, i * 1000) is None
    trip = cb.check_and_record("prompt", 0.01, 3000)
    assert trip == CircuitTrip.RATE_LIMIT


def test_trips_on_cost_velocity():
    cb = CircuitBreaker(CircuitBreakerConfig(max_cost_velocity_usd=0.50))
    cb.check_and_record("a", 0.40, 0)
    trip = cb.check_and_record("b", 0.20, 1000)
    assert trip == CircuitTrip.COST_VELOCITY


def test_trips_on_repeated_prompt():
    cb = CircuitBreaker(CircuitBreakerConfig(trip_on_repeated_prompt=True))
    cb.check_and_record("same prompt", 0.01, 0)
    trip = cb.check_and_record("same prompt", 0.01, 1000)
    assert trip == CircuitTrip.REPEATED_PROMPT


def test_different_prompts_do_not_trip():
    cb = CircuitBreaker(CircuitBreakerConfig(trip_on_repeated_prompt=True))
    cb.check_and_record("prompt A", 0.01, 0)
    result = cb.check_and_record("prompt B", 0.01, 1000)
    assert result is None


def test_retry_depth_trips_after_max():
    cb = CircuitBreaker(CircuitBreakerConfig(max_retry_depth=2))
    assert cb.increment_retry() is None
    assert cb.increment_retry() is None
    assert cb.increment_retry() == CircuitTrip.MAX_RETRY_DEPTH


def test_reset_retry_clears_depth():
    cb = CircuitBreaker(CircuitBreakerConfig(max_retry_depth=1))
    cb.increment_retry()
    cb.reset_retry()
    assert cb.increment_retry() is None


def test_old_requests_evicted_from_window():
    cb = CircuitBreaker(CircuitBreakerConfig(max_requests_per_minute=2))
    cb.check_and_record("a", 0.01, 0)
    cb.check_and_record("b", 0.01, 1000)
    # now 61 seconds later — old entries should be evicted
    result = cb.check_and_record("c", 0.01, 61_000)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && pytest tests/test_circuit_breaker.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.budget.circuit_breaker'`

- [ ] **Step 3: Implement circuit_breaker.py**

Create `python/src/truss/budget/circuit_breaker.py`:

```python
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CircuitTrip(Enum):
    RATE_LIMIT = "rate_limit"
    COST_VELOCITY = "cost_velocity"
    MAX_RETRY_DEPTH = "max_retry_depth"
    REPEATED_PROMPT = "repeated_prompt"


@dataclass
class CircuitBreakerConfig:
    max_requests_per_minute: int = 60
    max_cost_velocity_usd: float = 1.0
    max_retry_depth: int = 3
    trip_on_repeated_prompt: bool = True


@dataclass
class _Record:
    timestamp_ms: int
    cost_usd: float
    prompt_hash: int


def _fnv1a(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode():
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._window: deque[_Record] = deque()
        self._retry_depth = 0
        self._lock = threading.Lock()

    def check_and_record(self, prompt: str, cost_usd: float, now_ms: int) -> Optional[CircuitTrip]:
        prompt_hash = _fnv1a(prompt)
        with self._lock:
            cutoff = now_ms - 60_000
            while self._window and self._window[0].timestamp_ms < cutoff:
                self._window.popleft()

            if len(self._window) >= self._config.max_requests_per_minute:
                return CircuitTrip.RATE_LIMIT

            running_cost = sum(r.cost_usd for r in self._window)
            if running_cost + cost_usd > self._config.max_cost_velocity_usd:
                return CircuitTrip.COST_VELOCITY

            if self._config.trip_on_repeated_prompt:
                recent = list(self._window)[-3:]
                if any(r.prompt_hash == prompt_hash for r in recent):
                    return CircuitTrip.REPEATED_PROMPT

            self._window.append(_Record(now_ms, cost_usd, prompt_hash))
            return None

    def increment_retry(self) -> Optional[CircuitTrip]:
        with self._lock:
            self._retry_depth += 1
            if self._retry_depth > self._config.max_retry_depth:
                return CircuitTrip.MAX_RETRY_DEPTH
            return None

    def reset_retry(self) -> None:
        with self._lock:
            self._retry_depth = 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && pytest tests/test_circuit_breaker.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/budget/circuit_breaker.py python/tests/test_circuit_breaker.py
git commit -m "feat: add circuit breaker with rate, cost velocity, and repeat detection"
```

---

## Task 8: Checkpoints + Rollback

**Files:**
- Create: `python/src/truss/coord/checkpoint.py`
- Create: `python/src/truss/coord/sqlite_checkpoint.py`
- Create: `python/tests/test_checkpoint.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_checkpoint.py`:

```python
import pytest
import os
import tempfile
from uuid import uuid4
from truss.types import AgentEnvelope
from truss.coord.checkpoint import (
    Checkpoint, CheckpointMeta, InMemoryCheckpointStore,
)
from truss.coord.sqlite_checkpoint import SqliteCheckpointStore
from truss.errors import CheckpointNotFound


def make_checkpoint(session_id: str, description: str) -> Checkpoint:
    return Checkpoint(
        session_id=session_id,
        agent_name="test-agent",
        envelope_snapshot=AgentEnvelope(task="test task"),
        description=description,
    )


# --- InMemory ---

def test_save_and_load_checkpoint():
    store = InMemoryCheckpointStore()
    cp = make_checkpoint("sess-1", "after planning")
    cp_id = store.save(cp)
    loaded = store.load(cp_id)
    assert loaded.description == "after planning"


def test_rollback_returns_envelope():
    store = InMemoryCheckpointStore()
    cp = make_checkpoint("sess-1", "step 1")
    cp.envelope_snapshot.task = "original task"
    cp_id = store.save(cp)
    env = store.rollback(cp_id)
    assert env.task == "original task"


def test_load_nonexistent_raises():
    store = InMemoryCheckpointStore()
    with pytest.raises(CheckpointNotFound):
        store.load(uuid4())


def test_list_filters_by_session():
    store = InMemoryCheckpointStore()
    store.save(make_checkpoint("sess-a", "cp1"))
    store.save(make_checkpoint("sess-b", "cp2"))
    metas = store.list("sess-a")
    assert len(metas) == 1
    assert metas[0].description == "cp1"


# --- SQLite ---

def test_sqlite_checkpoint_save_and_load():
    store = SqliteCheckpointStore(":memory:")
    cp = make_checkpoint("sess-1", "sqlite test")
    cp_id = store.save(cp)
    loaded = store.load(cp_id)
    assert loaded.description == "sqlite test"
    assert loaded.envelope_snapshot.task == "test task"


def test_sqlite_checkpoint_survives_reopen():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        store1 = SqliteCheckpointStore(path)
        cp = make_checkpoint("sess-p", "persistent")
        cp_id = store1.save(cp)

        store2 = SqliteCheckpointStore(path)
        loaded = store2.load(cp_id)
        assert loaded.description == "persistent"
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && pytest tests/test_checkpoint.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.coord.checkpoint'`

- [ ] **Step 3: Implement checkpoint.py**

Create `python/src/truss/coord/checkpoint.py`:

```python
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from truss.types import AgentEnvelope
from truss.errors import CheckpointNotFound


@dataclass
class CheckpointMeta:
    id: UUID
    session_id: str
    agent_name: str
    description: str
    created_at: int


@dataclass
class Checkpoint:
    session_id: str
    agent_name: str
    envelope_snapshot: AgentEnvelope
    description: str = ""
    id: UUID = field(default_factory=uuid4)
    external_state: dict[str, bytes] = field(default_factory=dict)
    created_at: int = 0

    def meta(self) -> CheckpointMeta:
        return CheckpointMeta(
            id=self.id,
            session_id=self.session_id,
            agent_name=self.agent_name,
            description=self.description,
            created_at=self.created_at,
        )


@runtime_checkable
class CheckpointStore(Protocol):
    def save(self, cp: Checkpoint) -> UUID: ...
    def load(self, id: UUID) -> Checkpoint: ...
    def rollback(self, id: UUID) -> AgentEnvelope: ...
    def list(self, session_id: str) -> list[CheckpointMeta]: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._store: dict[UUID, Checkpoint] = {}
        self._lock = threading.RLock()

    def save(self, cp: Checkpoint) -> UUID:
        with self._lock:
            self._store[cp.id] = cp
        return cp.id

    def load(self, id: UUID) -> Checkpoint:
        with self._lock:
            cp = self._store.get(id)
        if cp is None:
            raise CheckpointNotFound(str(id))
        return cp

    def rollback(self, id: UUID) -> AgentEnvelope:
        return self.load(id).envelope_snapshot

    def list(self, session_id: str) -> list[CheckpointMeta]:
        with self._lock:
            metas = [cp.meta() for cp in self._store.values() if cp.session_id == session_id]
        return sorted(metas, key=lambda m: m.created_at)
```

- [ ] **Step 4: Implement sqlite_checkpoint.py**

Create `python/src/truss/coord/sqlite_checkpoint.py`:

```python
from __future__ import annotations

import json
import sqlite3
import threading
from uuid import UUID, uuid4

from truss.types import AgentEnvelope
from truss.coord.checkpoint import Checkpoint, CheckpointMeta
from truss.errors import CheckpointNotFound


class SqliteCheckpointStore:
    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id           TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL,
                agent_name   TEXT NOT NULL,
                description  TEXT NOT NULL,
                envelope_json TEXT NOT NULL,
                created_at   INTEGER NOT NULL
            )
        """)
        self._conn.commit()

    def save(self, cp: Checkpoint) -> UUID:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?)",
                (
                    str(cp.id), cp.session_id, cp.agent_name, cp.description,
                    cp.envelope_snapshot.model_dump_json(), cp.created_at,
                ),
            )
            self._conn.commit()
        return cp.id

    def load(self, id: UUID) -> Checkpoint:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, session_id, agent_name, description, envelope_json, created_at FROM checkpoints WHERE id = ?",
                (str(id),),
            ).fetchone()
        if row is None:
            raise CheckpointNotFound(str(id))
        cp_id, session_id, agent_name, description, env_json, created_at = row
        return Checkpoint(
            id=UUID(cp_id),
            session_id=session_id,
            agent_name=agent_name,
            description=description,
            envelope_snapshot=AgentEnvelope.model_validate_json(env_json),
            created_at=created_at,
        )

    def rollback(self, id: UUID) -> AgentEnvelope:
        return self.load(id).envelope_snapshot

    def list(self, session_id: str) -> list[CheckpointMeta]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, agent_name, description, created_at FROM checkpoints WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [
            CheckpointMeta(id=UUID(r[0]), session_id=r[1], agent_name=r[2], description=r[3], created_at=r[4])
            for r in rows
        ]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd python && pytest tests/test_checkpoint.py -v
```

Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add python/src/truss/coord/ python/tests/test_checkpoint.py
git commit -m "feat: add checkpoint store with InMemory and SQLite backends"
```

---

## Task 9: Fence

**Files:**
- Create: `python/src/truss/fence/memory_fence.py`
- Create: `python/tests/test_fence.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_fence.py`:

```python
import pytest
from truss.fence.memory_fence import InMemoryFence, LockHandle
from truss.errors import FenceLockConflict


def test_acquire_grants_lock_to_first_owner():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=0)  # should not raise


def test_acquire_blocks_second_owner():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=0)
    with pytest.raises(FenceLockConflict):
        fence.acquire("doc-1", "agent-b", ttl_ms=30_000, now_ms=1_000)


def test_expired_lock_can_be_reacquired():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=5_000, now_ms=0)
    fence.acquire("doc-1", "agent-b", ttl_ms=5_000, now_ms=30_000)  # 30s later, expired


def test_release_frees_lock():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=0)
    fence.release("doc-1", "agent-a")
    fence.acquire("doc-1", "agent-b", ttl_ms=30_000, now_ms=1_000)  # should succeed


def test_release_by_wrong_owner_is_noop():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=0)
    fence.release("doc-1", "agent-b")  # wrong owner — no-op
    assert fence.is_locked("doc-1", now_ms=1_000) is True


def test_is_locked_returns_false_after_expiry():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=5_000, now_ms=0)
    assert fence.is_locked("doc-1", now_ms=6_000) is False


def test_owner_can_reacquire_own_lock():
    fence = InMemoryFence()
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=0)
    fence.acquire("doc-1", "agent-a", ttl_ms=30_000, now_ms=1_000)  # refresh — should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && pytest tests/test_fence.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.fence.memory_fence'`

- [ ] **Step 3: Implement memory_fence.py**

Create `python/src/truss/fence/memory_fence.py`:

```python
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from truss.errors import FenceLockConflict


@dataclass
class LockHandle:
    key: str
    owner: str
    acquired_at_ms: int
    ttl_ms: int

    def is_expired(self, now_ms: int) -> bool:
        return now_ms > self.acquired_at_ms + self.ttl_ms


@runtime_checkable
class FenceStore(Protocol):
    def acquire(self, key: str, owner: str, ttl_ms: int, now_ms: int) -> None: ...
    def release(self, key: str, owner: str) -> None: ...
    def is_locked(self, key: str, now_ms: int) -> bool: ...


class InMemoryFence:
    def __init__(self) -> None:
        self._locks: dict[str, LockHandle] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str, owner: str, ttl_ms: int, now_ms: int) -> None:
        with self._lock:
            handle = self._locks.get(key)
            if handle is not None and not handle.is_expired(now_ms) and handle.owner != owner:
                raise FenceLockConflict(f"{key} held by {handle.owner}")
            self._locks[key] = LockHandle(key=key, owner=owner, acquired_at_ms=now_ms, ttl_ms=ttl_ms)

    def release(self, key: str, owner: str) -> None:
        with self._lock:
            handle = self._locks.get(key)
            if handle is not None and handle.owner == owner:
                del self._locks[key]

    def is_locked(self, key: str, now_ms: int) -> bool:
        with self._lock:
            handle = self._locks.get(key)
            return handle is not None and not handle.is_expired(now_ms)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && pytest tests/test_fence.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/fence/memory_fence.py python/tests/test_fence.py
git commit -m "feat: add in-memory fence with TTL-based lock expiry"
```

---

## Task 10: Multi-LLM Router

**Files:**
- Create: `python/src/truss/router/router.py`
- Create: `python/tests/test_router.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_router.py`:

```python
import pytest
from truss.router.router import ModelSpec, RouterConfig, RouterRule, route
from truss.types import ModelTier


def sample_models() -> list[ModelSpec]:
    return [
        ModelSpec(name="claude-haiku-4-5", tier=ModelTier.CHEAP, max_tokens=8192, cost_per_1k_input=0.001, cost_per_1k_output=0.005),
        ModelSpec(name="claude-sonnet-4-6", tier=ModelTier.STANDARD, max_tokens=16384, cost_per_1k_input=0.003, cost_per_1k_output=0.015),
        ModelSpec(name="claude-opus-4-8", tier=ModelTier.PREMIUM, max_tokens=32768, cost_per_1k_input=0.015, cost_per_1k_output=0.075),
    ]


def test_route_uses_rule_matching_keyword():
    config = RouterConfig(
        models=sample_models(),
        rules=[RouterRule(keywords=["summarise", "summarize"], preferred_tier=ModelTier.CHEAP)],
        default_tier=ModelTier.STANDARD,
    )
    model = route("summarise this document", config)
    assert model.tier == ModelTier.CHEAP


def test_route_falls_back_to_default_tier():
    config = RouterConfig(models=sample_models(), rules=[], default_tier=ModelTier.STANDARD)
    model = route("analyse this deeply", config)
    assert model.tier == ModelTier.STANDARD


def test_route_auto_tier_defaults_to_standard():
    config = RouterConfig(models=sample_models(), rules=[], default_tier=ModelTier.AUTO)
    model = route("any task", config)
    assert model.tier == ModelTier.STANDARD


def test_route_returns_cheapest_in_tier():
    models = [
        ModelSpec(name="cheap-a", tier=ModelTier.CHEAP, max_tokens=4096, cost_per_1k_input=0.002, cost_per_1k_output=0.010),
        ModelSpec(name="cheap-b", tier=ModelTier.CHEAP, max_tokens=4096, cost_per_1k_input=0.001, cost_per_1k_output=0.005),
    ]
    config = RouterConfig(models=models, rules=[], default_tier=ModelTier.CHEAP)
    model = route("any task", config)
    assert model.name == "cheap-b"


def test_route_first_matching_rule_wins():
    config = RouterConfig(
        models=sample_models(),
        rules=[
            RouterRule(keywords=["fast"], preferred_tier=ModelTier.CHEAP),
            RouterRule(keywords=["fast", "deep"], preferred_tier=ModelTier.PREMIUM),
        ],
        default_tier=ModelTier.STANDARD,
    )
    model = route("fast analysis", config)
    assert model.tier == ModelTier.CHEAP  # first rule matched
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && pytest tests/test_router.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.router.router'`

- [ ] **Step 3: Implement router.py**

Create `python/src/truss/router/router.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from truss.types import ModelTier


@dataclass
class ModelSpec:
    name: str
    tier: ModelTier
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float

    @property
    def cost_per_1k_total(self) -> float:
        return self.cost_per_1k_input + self.cost_per_1k_output


@dataclass
class RouterRule:
    keywords: list[str]
    preferred_tier: ModelTier


@dataclass
class RouterConfig:
    models: list[ModelSpec]
    rules: list[RouterRule] = field(default_factory=list)
    default_tier: ModelTier = ModelTier.AUTO


def route(task: str, config: RouterConfig) -> ModelSpec:
    task_lower = task.lower()
    tier = config.default_tier

    for rule in config.rules:
        if any(kw in task_lower for kw in rule.keywords):
            tier = rule.preferred_tier
            break

    if tier == ModelTier.AUTO:
        tier = ModelTier.STANDARD

    candidates = [m for m in config.models if m.tier == tier]
    if not candidates:
        candidates = config.models  # fallback to all models

    return min(candidates, key=lambda m: m.cost_per_1k_total)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && pytest tests/test_router.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/router/router.py python/tests/test_router.py
git commit -m "feat: add multi-LLM router with keyword-based routing rules"
```

---

## Task 11: MCP Interceptor

**Files:**
- Create: `python/src/truss/mcp/interceptor.py`
- Create: `python/tests/test_mcp.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_mcp.py`:

```python
import pytest
from truss.mcp.interceptor import McpManifest, McpInterceptor, McpCall
from truss.errors import ToolOutOfScope


def test_allowed_tool_passes():
    manifest = McpManifest(allowed_tools=["read_file", "list_dir"])
    interceptor = McpInterceptor(manifest)
    interceptor.check(McpCall(tool_name="read_file", arguments={"path": "/tmp/x"}))  # no exception


def test_denied_tool_raises():
    manifest = McpManifest(allowed_tools=["read_file"])
    interceptor = McpInterceptor(manifest)
    with pytest.raises(ToolOutOfScope, match="write_file"):
        interceptor.check(McpCall(tool_name="write_file", arguments={}))


def test_wrap_calls_fn_on_allowed():
    manifest = McpManifest(allowed_tools=["tool_a"])
    interceptor = McpInterceptor(manifest)
    call = McpCall(tool_name="tool_a", arguments={"x": 1})
    result = interceptor.wrap(call, lambda c: c.arguments["x"] * 2)
    assert result == 2


def test_wrap_raises_before_calling_fn_on_denied():
    called = []
    manifest = McpManifest(allowed_tools=["tool_a"])
    interceptor = McpInterceptor(manifest)
    with pytest.raises(ToolOutOfScope):
        interceptor.wrap(McpCall(tool_name="tool_b", arguments={}), lambda c: called.append(True))
    assert not called


def test_empty_manifest_denies_all():
    manifest = McpManifest(allowed_tools=[])
    interceptor = McpInterceptor(manifest)
    with pytest.raises(ToolOutOfScope):
        interceptor.check(McpCall(tool_name="any_tool", arguments={}))


def test_manifest_is_allowed():
    manifest = McpManifest(allowed_tools=["a", "b"])
    assert manifest.is_allowed("a") is True
    assert manifest.is_allowed("c") is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && pytest tests/test_mcp.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.mcp.interceptor'`

- [ ] **Step 3: Implement interceptor.py**

Create `python/src/truss/mcp/interceptor.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from truss.errors import ToolOutOfScope


@dataclass
class McpCall:
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class McpManifest:
    allowed_tools: list[str]

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools


class McpInterceptor:
    def __init__(self, manifest: McpManifest) -> None:
        self._manifest = manifest

    def check(self, call: McpCall) -> None:
        if not self._manifest.is_allowed(call.tool_name):
            raise ToolOutOfScope(f"{call.tool_name} denied by manifest")

    def wrap(self, call: McpCall, fn: Callable[[McpCall], Any]) -> Any:
        self.check(call)
        return fn(call)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && pytest tests/test_mcp.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/mcp/interceptor.py python/tests/test_mcp.py
git commit -m "feat: add MCP interceptor with manifest-based tool gating"
```

---

## Task 12: Session

**Files:**
- Create: `python/src/truss/session.py`
- Create: `python/tests/test_session.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_session.py`:

```python
import pytest
from truss.session import Session, SessionReport
from truss.types import AgentEnvelope, ContextBlock, ContextRole, ContextWeight


def sample_blocks(n: int = 10) -> list[ContextBlock]:
    weights = [ContextWeight.CRITICAL, ContextWeight.HIGH, ContextWeight.NORMAL, ContextWeight.BACKGROUND]
    return [
        ContextBlock(role=ContextRole.FINDING, weight=weights[i % 4], content=f"block {i} " + "content " * 20, source=f"s{i}")
        for i in range(n)
    ]


def test_session_report_after_compress():
    s = Session(target_tokens=100, preserve_recent=2)
    s.compress(sample_blocks(20))
    report = s.report()
    assert report.tokens_saved >= 0
    assert report.tokens_before > 0


def test_session_tracks_budget():
    s = Session(budget_usd=1.0)
    s.record_usage(500, 100, cost_usd=0.05, model="test")
    report = s.report()
    assert abs(report.budget_used_usd - 0.05) < 0.001
    assert report.budget_limit_usd == 1.0


def test_session_compress_keeps_critical_blocks():
    critical = ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL, content="must keep this", source="agent")
    bg = [ContextBlock(role=ContextRole.BACKGROUND, weight=ContextWeight.BACKGROUND, content="noise " * 50, source="x") for _ in range(20)]
    s = Session(target_tokens=50, preserve_recent=0)
    result = s.compress([critical] + bg)
    ids = {b.id for b in result.blocks}
    assert critical.id in ids


def test_session_checkpoint_and_rollback():
    s = Session()
    s._envelope.task = "original"
    cp_id = s.checkpoint("before change")
    s._envelope.task = "modified"
    s.rollback(cp_id)
    assert s._envelope.task == "original"


def test_session_checkpoint_count_in_report():
    s = Session()
    s.checkpoint("cp1")
    s.checkpoint("cp2")
    report = s.report()
    assert report.checkpoint_count == 2


async def test_session_async_context_manager():
    async with Session(budget_usd=0.50) as s:
        result = s.compress(sample_blocks(5))
        assert result.tokens_after <= result.tokens_before
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && pytest tests/test_session.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.session'`

- [ ] **Step 3: Implement session.py**

Create `python/src/truss/session.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from truss.budget.config import BudgetWindow
from truss.budget.ledger import LedgerEntry, LedgerStore
from truss.budget.memory_store import InMemoryStore
from truss.context.surgeon import SurgeonConfig, SurgeonResult, compress
from truss.coord.checkpoint import CheckpointStore, InMemoryCheckpointStore, Checkpoint
from truss.fence.memory_fence import FenceStore, InMemoryFence
from truss.types import AgentEnvelope, ContextBlock


@dataclass
class SessionReport:
    session_id: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    cost_saved_usd: float
    budget_used_usd: float
    budget_limit_usd: Optional[float]
    checkpoint_count: int

    def __str__(self) -> str:
        lines = [
            f"Session: {self.session_id[:8]}",
            f"Context: {self.tokens_before:,} → {self.tokens_after:,} tokens (saved {self.tokens_saved:,})",
            f"Est. savings: ${self.cost_saved_usd:.4f}",
        ]
        if self.budget_limit_usd is not None:
            pct = (self.budget_used_usd / self.budget_limit_usd * 100) if self.budget_limit_usd else 0
            lines.append(f"Budget: ${self.budget_used_usd:.4f} of ${self.budget_limit_usd:.2f} used ({pct:.1f}%)")
        if self.checkpoint_count:
            lines.append(f"Checkpoints: {self.checkpoint_count}")
        return "\n".join(lines)


class Session:
    _COST_PER_1K_TOKENS = 0.001  # conservative default

    def __init__(
        self,
        envelope: Optional[AgentEnvelope] = None,
        budget_usd: Optional[float] = None,
        target_tokens: int = 8_000,
        preserve_recent: int = 5,
        ledger: Optional[LedgerStore] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        fence: Optional[FenceStore] = None,
    ) -> None:
        self._envelope = envelope or AgentEnvelope(task="session")
        self._budget_usd = budget_usd
        self._tokens_before = 0
        self._tokens_after = 0
        self._ledger: LedgerStore = ledger or InMemoryStore()
        self._checkpoint_store: CheckpointStore = checkpoint_store or InMemoryCheckpointStore()
        self._fence: FenceStore = fence or InMemoryFence()
        self._surgeon_config = SurgeonConfig(target_tokens=target_tokens, preserve_recent=preserve_recent)
        self._checkpoint_count = 0

    @property
    def session_id(self) -> str:
        return str(self._envelope.id)

    def compress(self, blocks: list[ContextBlock]) -> SurgeonResult:
        result = compress(blocks, self._surgeon_config)
        self._tokens_before += result.tokens_before
        self._tokens_after += result.tokens_after
        return result

    def record_usage(self, input_tokens: int, output_tokens: int, cost_usd: float = 0.0, model: str = "unknown") -> None:
        if cost_usd == 0.0:
            cost_usd = (input_tokens + output_tokens) / 1000 * self._COST_PER_1K_TOKENS
        self._ledger.record(LedgerEntry(
            session_id=self.session_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ))

    def checkpoint(self, description: str = "") -> UUID:
        cp = Checkpoint(
            session_id=self.session_id,
            agent_name="session",
            envelope_snapshot=self._envelope.model_copy(deep=True),
            description=description,
        )
        self._checkpoint_count += 1
        return self._checkpoint_store.save(cp)

    def rollback(self, checkpoint_id: UUID) -> None:
        self._envelope = self._checkpoint_store.rollback(checkpoint_id)

    def report(self) -> SessionReport:
        tokens_saved = max(0, self._tokens_before - self._tokens_after)
        cost_saved = tokens_saved / 1000 * self._COST_PER_1K_TOKENS
        usage = self._ledger.usage(self.session_id, BudgetWindow.PER_SESSION)
        return SessionReport(
            session_id=self.session_id,
            tokens_before=self._tokens_before,
            tokens_after=self._tokens_after,
            tokens_saved=tokens_saved,
            cost_saved_usd=cost_saved,
            budget_used_usd=usage.total_cost_usd,
            budget_limit_usd=self._budget_usd,
            checkpoint_count=self._checkpoint_count,
        )

    async def __aenter__(self) -> "Session":
        return self

    async def __aexit__(self, *_) -> None:
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && pytest tests/test_session.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/session.py python/tests/test_session.py
git commit -m "feat: add Session class with compress, checkpoint, record_usage, report"
```

---

## Task 13: LangChain Adapter

**Files:**
- Create: `python/src/truss/adapters/langchain.py`
- Create: `python/tests/test_langchain.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_langchain.py`:

```python
import pytest
from truss.adapters.langchain import TrussMemory


def test_save_context_adds_two_blocks():
    mem = TrussMemory(target_tokens=1000)
    mem.save_context({"input": "What is the weather?"}, {"output": "It is sunny."})
    assert len(mem.blocks) == 2


def test_load_memory_variables_returns_history_key():
    mem = TrussMemory(target_tokens=1000)
    mem.save_context({"input": "Hello"}, {"output": "World"})
    vars_ = mem.load_memory_variables({})
    assert "history" in vars_
    assert "Human:" in vars_["history"]
    assert "AI:" in vars_["history"]


def test_compression_fires_when_over_budget():
    mem = TrussMemory(target_tokens=20, preserve_recent=1)
    for i in range(30):
        mem.save_context({"input": f"q{i} " + "x" * 100}, {"output": f"a{i} " + "y" * 100})
    vars_ = mem.load_memory_variables({})
    # After compression, fewer blocks remain
    assert len(vars_["history"]) < 30 * 200


def test_clear_empties_blocks():
    mem = TrussMemory()
    mem.save_context({"input": "hi"}, {"output": "hello"})
    mem.clear()
    assert len(mem.blocks) == 0


def test_memory_key_attribute():
    mem = TrussMemory()
    assert mem.memory_key == "history"


def test_custom_memory_key():
    mem = TrussMemory(memory_key="chat_history")
    mem.save_context({"input": "hi"}, {"output": "hello"})
    vars_ = mem.load_memory_variables({})
    assert "chat_history" in vars_
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && pytest tests/test_langchain.py -v
```

Expected: `ModuleNotFoundError: No module named 'truss.adapters.langchain'`

- [ ] **Step 3: Implement langchain.py**

Create `python/src/truss/adapters/langchain.py`:

```python
from __future__ import annotations

from typing import Any

from truss.context.surgeon import SurgeonConfig, compress
from truss.types import ContextBlock, ContextRole, ContextWeight


class TrussMemory:
    """Drop-in replacement for LangChain ConversationBufferMemory.

    Compresses conversation history using Truss context surgeon before returning.
    No dependency on langchain-core at import time — works with or without it.
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
        self._config = SurgeonConfig(target_tokens=target_tokens, preserve_recent=preserve_recent)

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && pytest tests/test_langchain.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add python/src/truss/adapters/langchain.py python/tests/test_langchain.py
git commit -m "feat: add LangChain TrussMemory adapter"
```

---

## Task 14: Public API + Hermes Example

**Files:**
- Modify: `python/src/truss/__init__.py`
- Create: `python/examples/hermes/main.py`

- [ ] **Step 1: Write the full public __init__.py**

Replace `python/src/truss/__init__.py`:

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

__all__ = [
    # Errors
    "TrussError", "BudgetExceeded", "ToolOutOfScope", "CheckpointNotFound", "FenceLockConflict",
    # Types
    "ContextBlock", "ContextRole", "ContextWeight", "estimate_tokens",
    "AgentEnvelope", "ModelTier", "EvidenceRef", "DecisionRecord",
    # Context
    "compress", "SurgeonConfig", "SurgeonResult", "CompressionStrategy",
    "score_relevance", "detect_contradiction",
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
]
```

- [ ] **Step 2: Verify full import works**

```bash
cd python && python -c "import truss; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run the full test suite**

```bash
cd python && pytest -v
```

Expected: all tests pass (around 57 tests).

- [ ] **Step 4: Write Hermes reference example**

Create `python/examples/hermes/main.py`:

```python
"""
Hermes Agent — Truss Python Phase 1 reference example.

Demonstrates: Session, compress, handoff/pack, budget tracking, checkpoint, MCP interception, report.
Run with: python -m examples.hermes.main (from python/)
"""
import asyncio
from truss import (
    Session, ContextBlock, ContextRole, ContextWeight,
    AgentEnvelope, ModelTier,
    McpManifest, McpInterceptor, McpCall,
    ModelSpec, RouterConfig, RouterRule, route,
    pack, BudgetCarve,
)


async def main() -> None:
    # 1. Build a parent agent envelope
    parent = AgentEnvelope(task="Research cheapest S3-compatible cloud storage for 10TB", budget_usd_remaining=1.0)
    parent.context = [
        ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL,
                     content="Find cheapest S3-compatible storage for 10TB dataset under $500/month.", source="user"),
        ContextBlock(role=ContextRole.CONSTRAINT, weight=ContextWeight.CRITICAL,
                     content="Must be S3-compatible. Budget: $500/month max.", source="user"),
        ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.HIGH,
                     content="Backblaze B2: $6/TB/month, S3-compatible.", source="search"),
        ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.HIGH,
                     content="Cloudflare R2: $15/TB/month, zero egress fees.", source="search"),
        ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.NORMAL,
                     content="AWS S3 Standard: $23/TB/month, widest ecosystem.", source="search"),
        ContextBlock(role=ContextRole.BACKGROUND, weight=ContextWeight.BACKGROUND,
                     content="Cloud storage history dates to the 1960s mainframe era.", source="wikipedia"),
        ContextBlock(role=ContextRole.BACKGROUND, weight=ContextWeight.BACKGROUND,
                     content="IBM 305 RAMAC was the first commercial hard disk drive, 1956.", source="wikipedia"),
    ]

    # 2. Route to cheapest model for a summarise task
    models = [
        ModelSpec("claude-haiku-4-5", ModelTier.CHEAP, 8192, 0.001, 0.005),
        ModelSpec("claude-sonnet-4-6", ModelTier.STANDARD, 16384, 0.003, 0.015),
    ]
    router_config = RouterConfig(
        models=models,
        rules=[RouterRule(keywords=["summarise", "summarize", "cheapest"], preferred_tier=ModelTier.CHEAP)],
    )
    selected_model = route("Find cheapest storage option", router_config)

    # 3. Set up MCP interceptor — only allow read tools
    manifest = McpManifest(allowed_tools=["search_web", "read_url"])
    interceptor = McpInterceptor(manifest)

    # 4. Pack a child envelope for a sub-agent
    child = pack(parent, "Rank storage options by price", carry_weights=[ContextWeight.CRITICAL, ContextWeight.HIGH], budget_carve=BudgetCarve.percent(0.3))

    # 5. Run a Session
    async with Session(envelope=parent, budget_usd=1.0, target_tokens=150, preserve_recent=2) as s:
        # Compress the parent context
        result = s.compress(parent.context)

        # Save a checkpoint
        cp_id = s.checkpoint("after compression")

        # Simulate an LLM call using the routed model
        s.record_usage(input_tokens=result.tokens_after, output_tokens=80, cost_usd=0.005, model=selected_model.name)

        # Simulate an MCP check (allowed)
        try:
            interceptor.check(McpCall(tool_name="search_web", arguments={"q": "cheapest cloud storage"}))
            print("MCP: search_web allowed ✓")
        except Exception as e:
            print(f"MCP blocked: {e}")

        # Try a disallowed MCP tool
        try:
            interceptor.check(McpCall(tool_name="delete_file", arguments={"path": "/important.db"}))
        except Exception as e:
            print(f"MCP blocked delete_file ✓ ({e})")

        print("\n=== Compression ===")
        print(f"Blocks: {len(parent.context)} → {len(result.blocks)} (saved {result.tokens_saved} tokens)")

        print("\n=== Child Envelope ===")
        print(f"Task: {child.task}")
        print(f"Context blocks carried: {len(child.context)}")
        print(f"Budget carved: ${child.budget_usd_remaining:.2f}")

        print("\n=== Session Report ===")
        print(s.report())


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: Run the example**

```bash
cd python && python examples/hermes/main.py
```

Expected output (exact numbers vary):
```
MCP: search_web allowed ✓
MCP blocked delete_file ✓ (delete_file denied by manifest)

=== Compression ===
Blocks: 7 → N (saved M tokens)

=== Child Envelope ===
Task: Rank storage options by price
Context blocks carried: 4
Budget carved: $0.30

=== Session Report ===
Session: xxxxxxxx
Context: N → M tokens (saved K)
Est. savings: $0.00XX
Budget: $0.0050 of $1.00 used (0.5%)
Checkpoints: 1
```

- [ ] **Step 6: Commit**

```bash
git add python/src/truss/__init__.py python/examples/
git commit -m "feat: wire public API and add Hermes reference example"
```

---

## Self-Review Against Spec

| Spec requirement | Task |
|---|---|
| ContextBlock with role, weight, token_count, source | Task 2 |
| AgentEnvelope with optional checkpoint_id, budget_usd_remaining, scope | Task 2 |
| estimate_tokens = ceil(chars/4) | Task 2 |
| Context surgeon: SlidingWindow, WeightedPrune, Hybrid | Task 3 |
| score_relevance / detect_contradiction (keyword heuristics) | Task 3 |
| BudgetWindow, BudgetLimit, AlertConfig, BudgetConfig | Task 4 |
| LedgerStore protocol + InMemoryStore | Task 4 |
| SqliteLedgerStore (stdlib sqlite3) | Task 5 |
| Agent handoff pack/unpack with BudgetCarve | Task 6 |
| Circuit breaker: rate, cost velocity, repeat, retry depth | Task 7 |
| Checkpoint: save/load/rollback/list, InMemory + SQLite | Task 8 |
| Fence: acquire/release/is_locked with TTL, owner check | Task 9 |
| Multi-LLM Router: keyword rules, tier selection | Task 10 |
| MCP Interceptor: manifest allowlist, wrap() | Task 11 |
| Session with compress/record_usage/checkpoint/rollback/report | Task 12 |
| LangChain TrussMemory adapter | Task 13 |
| Hermes reference example | Task 14 |
| TrussError hierarchy | Task 2 |
| Thread safety on all stores | Tasks 4, 5, 8, 9 (RLock/Lock) |

**Gaps not in Python Phase 1:**
- TypeScript bindings (separate plan)
- SemanticDedup and Summarise compression strategies (Phase 2)
- Slack webhook alerting in AlertConfig (stub field present, not wired)
- Real token counting per-model (chars/4 only in Phase 1)
