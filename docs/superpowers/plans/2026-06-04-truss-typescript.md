# Truss TypeScript — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `truss-ai` as a pure TypeScript/Node.js package implementing all 7 Truss modules — Context Surgeon, Agent Handoff, Token Budget/Ledger, Multi-LLM Router, MCP Interceptor, Checkpoints/Rollback, and Fence — with SQLite persistence and a Hermes reference example.

**Architecture:** ESM-first TypeScript package under `typescript/src/` with one directory per domain. Native TypeScript interfaces for all types (no runtime schema library). `better-sqlite3` for persistence. `vitest` for testing. Built with `tsup` for dual CJS+ESM output.

**Tech Stack:** Node.js 18+ · TypeScript 5.x · vitest 1.x · better-sqlite3 9.x · tsup 8.x · uuid 9.x

---

## File Structure

```
typescript/
├── package.json
├── tsconfig.json
├── tsup.config.ts
├── src/
│   ├── index.ts                 # public re-exports
│   ├── errors.ts                # TrussError hierarchy
│   ├── types.ts                 # ContextBlock, AgentEnvelope, shared enums
│   ├── context/
│   │   ├── index.ts
│   │   └── surgeon.ts           # compress(), SurgeonConfig, strategies
│   ├── handoff/
│   │   ├── index.ts
│   │   └── envelope.ts          # pack(), unpack(), BudgetCarve
│   ├── budget/
│   │   ├── index.ts
│   │   ├── config.ts            # BudgetConfig, BudgetWindow, AlertConfig
│   │   ├── ledger.ts            # LedgerEntry, LedgerStore interface, UsageReport
│   │   ├── memory-store.ts      # InMemoryStore
│   │   ├── sqlite-store.ts      # SqliteLedgerStore
│   │   └── circuit-breaker.ts   # CircuitBreaker, CircuitBreakerConfig, CircuitTrip
│   ├── coord/
│   │   ├── index.ts
│   │   ├── checkpoint.ts        # Checkpoint, CheckpointStore, InMemoryCheckpointStore
│   │   └── sqlite-checkpoint.ts # SqliteCheckpointStore
│   ├── fence/
│   │   ├── index.ts
│   │   └── memory-fence.ts      # FenceStore, InMemoryFence, LockHandle
│   ├── router/
│   │   ├── index.ts
│   │   └── router.ts            # ModelSpec, RouterConfig, route()
│   ├── mcp/
│   │   ├── index.ts
│   │   └── interceptor.ts       # McpManifest, McpInterceptor, McpCall
│   └── session.ts               # Session class + SessionReport
├── tests/
│   ├── types.test.ts
│   ├── surgeon.test.ts
│   ├── budget.test.ts
│   ├── handoff.test.ts
│   ├── circuit-breaker.test.ts
│   ├── checkpoint.test.ts
│   ├── fence.test.ts
│   ├── router.test.ts
│   ├── mcp.test.ts
│   └── session.test.ts
└── examples/
    └── hermes.ts
```

---

## Task 1: TypeScript Project Setup

**Files:**
- Create: `typescript/package.json`
- Create: `typescript/tsconfig.json`
- Create: `typescript/tsup.config.ts`
- Create: `typescript/src/index.ts` (stub)

- [ ] **Step 1: Create directory structure**

```powershell
$dirs = @(
  "typescript/src/context",
  "typescript/src/handoff",
  "typescript/src/budget",
  "typescript/src/coord",
  "typescript/src/fence",
  "typescript/src/router",
  "typescript/src/mcp",
  "typescript/tests",
  "typescript/examples"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d }
```

- [ ] **Step 2: Create package.json**

Create `typescript/package.json`:

```json
{
  "name": "truss-ai",
  "version": "0.1.0",
  "description": "The structural layer for agentic AI workflows",
  "type": "module",
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "require": "./dist/index.cjs",
      "types": "./dist/index.d.ts"
    }
  },
  "scripts": {
    "build": "tsup",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "better-sqlite3": "^9.0.0",
    "uuid": "^9.0.0"
  },
  "devDependencies": {
    "@types/better-sqlite3": "^7.6.0",
    "@types/node": "^20.0.0",
    "@types/uuid": "^9.0.0",
    "tsup": "^8.0.0",
    "typescript": "^5.0.0",
    "vitest": "^1.0.0"
  }
}
```

- [ ] **Step 3: Create tsconfig.json**

Create `typescript/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2022"],
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

- [ ] **Step 4: Create tsup.config.ts**

Create `typescript/tsup.config.ts`:

```typescript
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  sourcemap: true,
});
```

- [ ] **Step 5: Create stub index.ts**

Create `typescript/src/index.ts`:

```typescript
// Public API — populated in Task 12 after all modules exist.
export {};
```

- [ ] **Step 6: Create empty index.ts barrel files**

```powershell
$barrels = @(
  "typescript/src/context/index.ts",
  "typescript/src/handoff/index.ts",
  "typescript/src/budget/index.ts",
  "typescript/src/coord/index.ts",
  "typescript/src/fence/index.ts",
  "typescript/src/router/index.ts",
  "typescript/src/mcp/index.ts"
)
foreach ($f in $barrels) { New-Item -ItemType File -Force -Path $f; Set-Content $f "export {};" }
```

- [ ] **Step 7: Install dependencies**

```bash
cd typescript && npm install
```

Expected: `node_modules` created, no errors.

- [ ] **Step 8: Verify tests run (empty)**

```bash
cd typescript && npm test
```

Expected: `No test files found` or `0 tests passed`.

- [ ] **Step 9: Commit**

```bash
git add typescript/
git commit -m "chore: set up TypeScript project structure for truss-ai"
```

---

## Task 2: Error Hierarchy + Core Types

**Files:**
- Create: `typescript/src/errors.ts`
- Create: `typescript/src/types.ts`
- Create: `typescript/tests/types.test.ts`

- [ ] **Step 1: Write failing tests**

Create `typescript/tests/types.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { BudgetExceeded, ToolOutOfScope, CheckpointNotFound, FenceLockConflict } from '../src/errors.js';
import {
  estimateTokens, makeContextBlock, makeEnvelope,
  ContextRole, ContextWeight, ModelTier,
} from '../src/types.js';

describe('estimateTokens', () => {
  it('ceiling-divides by 4', () => {
    expect(estimateTokens('Hello')).toBe(2);        // ceil(5/4)
    expect(estimateTokens('Hello world')).toBe(3);  // ceil(11/4)
    expect(estimateTokens('')).toBe(0);
  });
});

describe('makeContextBlock', () => {
  it('auto-estimates token count', () => {
    const b = makeContextBlock(ContextRole.Task, ContextWeight.Critical, 'Hello world', 'test');
    expect(b.tokenCount).toBe(3);
  });

  it('has unique id', () => {
    const a = makeContextBlock(ContextRole.Task, ContextWeight.Normal, 'a', 'test');
    const b = makeContextBlock(ContextRole.Task, ContextWeight.Normal, 'a', 'test');
    expect(a.id).not.toBe(b.id);
  });
});

describe('ContextWeight', () => {
  it('is numerically comparable', () => {
    expect(ContextWeight.Critical).toBeGreaterThan(ContextWeight.Normal);
    expect(ContextWeight.Background).toBeLessThan(ContextWeight.High);
  });
});

describe('makeEnvelope', () => {
  it('starts with no checkpoint', () => {
    const env = makeEnvelope('test task');
    expect(env.checkpointId).toBeUndefined();
  });

  it('is JSON-serializable', () => {
    const env = makeEnvelope('analyse pricing');
    const json = JSON.stringify(env);
    const back = JSON.parse(json);
    expect(back.task).toBe(env.task);
    expect(back.id).toBe(env.id);
  });
});

describe('errors', () => {
  it('BudgetExceeded is instanceof Error', () => {
    const e = new BudgetExceeded('wallet exceeded $5');
    expect(e).toBeInstanceOf(Error);
    expect(e.message).toContain('wallet');
  });

  it('ToolOutOfScope includes tool name', () => {
    const e = new ToolOutOfScope('readFile denied');
    expect(e.message).toContain('readFile');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd typescript && npm test tests/types.test.ts
```

Expected: `Cannot find module '../src/errors.js'`

- [ ] **Step 3: Implement errors.ts**

Create `typescript/src/errors.ts`:

```typescript
export class TrussError extends Error {
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}

export class BudgetExceeded extends TrussError {}
export class ToolOutOfScope extends TrussError {}
export class CheckpointNotFound extends TrussError {}
export class FenceLockConflict extends TrussError {}
```

- [ ] **Step 4: Implement types.ts**

Create `typescript/src/types.ts`:

```typescript
import { v4 as uuidv4 } from 'uuid';

export type UUID = string;

export function estimateTokens(text: string): number {
  return text.length === 0 ? 0 : Math.ceil(text.length / 4);
}

export enum ContextRole {
  Task = 'task',
  Constraint = 'constraint',
  Finding = 'finding',
  Decision = 'decision',
  Warning = 'warning',
  Background = 'background',
}

export enum ContextWeight {
  Background = 0,
  Normal = 1,
  High = 2,
  Critical = 3,
}

export interface ContextBlock {
  id: UUID;
  role: ContextRole;
  weight: ContextWeight;
  content: string;
  source: string;
  tokenCount: number;
  createdAt: number;
  referencedBy: UUID[];
}

export function makeContextBlock(
  role: ContextRole,
  weight: ContextWeight,
  content: string,
  source: string,
): ContextBlock {
  return {
    id: uuidv4(),
    role,
    weight,
    content,
    source,
    tokenCount: estimateTokens(content),
    createdAt: 0,
    referencedBy: [],
  };
}

export enum ModelTier {
  Cheap = 'cheap',
  Standard = 'standard',
  Premium = 'premium',
  Auto = 'auto',
}

export interface EvidenceRef {
  id: UUID;
  content: string;
  sourceUrl?: string;
  toolName?: string;
  confidence: number;
}

export interface DecisionRecord {
  id: UUID;
  decision: string;
  reasoning: string;
  evidenceIds: UUID[];
  confidence: number;
  decidedBy: string;
  timestamp: number;
}

export interface AgentEnvelope {
  id: UUID;
  task: string;
  context: ContextBlock[];
  evidence: EvidenceRef[];
  decisions: DecisionRecord[];
  budgetUsdRemaining: number;
  checkpointId?: UUID;
  modelHint: ModelTier;
  parentAgent?: string;
  scope: string[];
  createdAt: number;
}

export function makeEnvelope(task: string): AgentEnvelope {
  return {
    id: uuidv4(),
    task,
    context: [],
    evidence: [],
    decisions: [],
    budgetUsdRemaining: Infinity,
    modelHint: ModelTier.Auto,
    scope: [],
    createdAt: 0,
  };
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd typescript && npm test tests/types.test.ts
```

Expected: `5 tests passed`

- [ ] **Step 6: Commit**

```bash
git add typescript/src/errors.ts typescript/src/types.ts typescript/tests/types.test.ts
git commit -m "feat: add TypeScript error hierarchy and core types"
```

---

## Task 3: Context Surgeon

**Files:**
- Create: `typescript/src/context/surgeon.ts`
- Modify: `typescript/src/context/index.ts`
- Create: `typescript/tests/surgeon.test.ts`

- [ ] **Step 1: Write failing tests**

Create `typescript/tests/surgeon.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { makeContextBlock, ContextRole, ContextWeight } from '../src/types.js';
import { compress, SurgeonConfig, CompressionStrategy, scoreRelevance, detectContradiction } from '../src/context/surgeon.js';

function makeBlock(weight: ContextWeight, tokens: number, content = ''): ReturnType<typeof makeContextBlock> {
  const b = makeContextBlock(ContextRole.Finding, weight, content || 'x '.repeat(tokens), 'test');
  b.tokenCount = tokens;
  return b;
}

describe('sliding window', () => {
  it('keeps recent N blocks within token budget', () => {
    const blocks = Array.from({ length: 10 }, (_, i) => makeBlock(ContextWeight.Normal, 100, `block ${i}`));
    const config: SurgeonConfig = { strategy: CompressionStrategy.SlidingWindow, targetTokens: 300, preserveRecent: 3, keepRecent: 3 };
    const result = compress(blocks, config);
    expect(result.tokensAfter).toBeLessThanOrEqual(300);
  });
});

describe('weighted prune', () => {
  it('drops background before normal', () => {
    const bg = makeBlock(ContextWeight.Background, 500, 'background');
    const normal = makeBlock(ContextWeight.Normal, 500, 'normal');
    const critical = makeBlock(ContextWeight.Critical, 100, 'critical');
    const config: SurgeonConfig = { strategy: CompressionStrategy.WeightedPrune, targetTokens: 700, preserveRecent: 0 };
    const result = compress([bg, normal, critical], config);
    const ids = new Set(result.blocks.map(b => b.id));
    expect(ids.has(bg.id)).toBe(false);
    expect(ids.has(critical.id)).toBe(true);
  });

  it('never removes critical blocks', () => {
    const critical = makeBlock(ContextWeight.Critical, 9000, 'must keep');
    const bg = makeBlock(ContextWeight.Background, 100, 'droppable');
    const config: SurgeonConfig = { strategy: CompressionStrategy.Hybrid, targetTokens: 500, preserveRecent: 0 };
    const result = compress([critical, bg], config);
    expect(result.blocks.some(b => b.id === critical.id)).toBe(true);
  });
});

describe('SurgeonResult', () => {
  it('tokensSaved equals before minus after', () => {
    const blocks = Array.from({ length: 4 }, () => makeBlock(ContextWeight.Background, 500));
    const config: SurgeonConfig = { strategy: CompressionStrategy.WeightedPrune, targetTokens: 500, preserveRecent: 0 };
    const result = compress(blocks, config);
    expect(result.tokensSaved).toBe(result.tokensBefore - result.tokensAfter);
  });
});

describe('scoreRelevance', () => {
  it('returns 0 for empty task', () => {
    expect(scoreRelevance(makeBlock(ContextWeight.Normal, 10, 'content'), '')).toBe(0);
  });

  it('returns 1 for exact match', () => {
    const b = makeBlock(ContextWeight.Normal, 10, 'pricing cloud storage');
    expect(scoreRelevance(b, 'pricing cloud storage')).toBe(1.0);
  });
});

describe('detectContradiction', () => {
  it('catches not-X pattern', () => {
    const a = makeBlock(ContextWeight.Normal, 10, 'the service is available');
    const b = makeBlock(ContextWeight.Normal, 10, 'the service is not available');
    expect(detectContradiction(a, b)).toBe(true);
  });

  it('no false positive for unrelated blocks', () => {
    const a = makeBlock(ContextWeight.Normal, 10, 'the weather is sunny today');
    const b = makeBlock(ContextWeight.Normal, 10, 'the price is twenty dollars');
    expect(detectContradiction(a, b)).toBe(false);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd typescript && npm test tests/surgeon.test.ts
```

Expected: `Cannot find module '../src/context/surgeon.js'`

- [ ] **Step 3: Implement surgeon.ts**

Create `typescript/src/context/surgeon.ts`:

```typescript
import { ContextBlock, ContextWeight } from '../types.js';

export enum CompressionStrategy {
  SlidingWindow = 'sliding_window',
  WeightedPrune = 'weighted_prune',
  Hybrid = 'hybrid',
}

export interface SurgeonConfig {
  strategy: CompressionStrategy;
  targetTokens: number;
  preserveRecent: number;
  keepRecent?: number;
}

export interface SurgeonResult {
  blocks: ContextBlock[];
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  strategyApplied: string;
}

export function compress(blocks: ContextBlock[], config: SurgeonConfig): SurgeonResult {
  const tokensBefore = blocks.reduce((s, b) => s + b.tokenCount, 0);
  let kept: ContextBlock[];

  if (config.strategy === CompressionStrategy.SlidingWindow) {
    kept = slidingWindow(blocks, config.keepRecent ?? config.preserveRecent, config.preserveRecent);
  } else if (config.strategy === CompressionStrategy.WeightedPrune) {
    kept = weightedPrune(blocks, config.targetTokens, config.preserveRecent);
  } else {
    const afterPrune = weightedPrune(blocks, config.targetTokens, config.preserveRecent);
    const total = afterPrune.reduce((s, b) => s + b.tokenCount, 0);
    kept = total > config.targetTokens
      ? slidingWindow(afterPrune, config.targetTokens, config.preserveRecent)
      : afterPrune;
  }

  const tokensAfter = kept.reduce((s, b) => s + b.tokenCount, 0);
  return { blocks: kept, tokensBefore, tokensAfter, tokensSaved: tokensBefore - tokensAfter, strategyApplied: config.strategy };
}

function slidingWindow(blocks: ContextBlock[], keepRecent: number, preserveRecent: number): ContextBlock[] {
  const alwaysKeep = Math.max(preserveRecent, keepRecent);
  if (blocks.length <= alwaysKeep) return [...blocks];

  const pinned = blocks.filter(b => b.weight >= ContextWeight.High);
  const pinnedIds = new Set(pinned.map(b => b.id));
  const recentStart = Math.max(0, blocks.length - keepRecent);
  const result = [...pinned];
  for (const b of blocks.slice(recentStart)) {
    if (!pinnedIds.has(b.id)) result.push(b);
  }
  return result;
}

function weightedPrune(blocks: ContextBlock[], targetTokens: number, preserveRecent: number): ContextBlock[] {
  const total = blocks.reduce((s, b) => s + b.tokenCount, 0);
  if (total <= targetTokens) return [...blocks];

  const preserveIds = new Set(blocks.slice(-preserveRecent).map(b => b.id));
  const removable = blocks
    .filter(b => !preserveIds.has(b.id) && b.weight < ContextWeight.High)
    .sort((a, b) => a.weight !== b.weight ? a.weight - b.weight : a.createdAt - b.createdAt);

  const toRemove = new Set<string>();
  let running = total;
  for (const b of removable) {
    if (running <= targetTokens) break;
    running -= b.tokenCount;
    toRemove.add(b.id);
  }
  return blocks.filter(b => !toRemove.has(b.id));
}

export function scoreRelevance(block: ContextBlock, task: string): number {
  const taskWords = new Set(task.split(/\s+/).filter(Boolean));
  if (taskWords.size === 0) return 0;
  const matches = block.content.split(/\s+/).filter(w => taskWords.has(w)).length;
  return Math.min(matches / taskWords.size, 1.0);
}

export function detectContradiction(a: ContextBlock, b: ContextBlock): boolean {
  const aLow = a.content.toLowerCase();
  const bLow = b.content.toLowerCase();
  for (const word of aLow.split(/\s+/)) {
    if (word.length > 4) {
      if (bLow.includes(`not ${word}`)) return true;
      if (aLow.includes(`not ${word}`) && bLow.includes(word)) return true;
    }
  }
  return false;
}
```

Update `typescript/src/context/index.ts`:

```typescript
export { compress, SurgeonConfig, SurgeonResult, CompressionStrategy, scoreRelevance, detectContradiction } from './surgeon.js';
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd typescript && npm test tests/surgeon.test.ts
```

Expected: `8 tests passed`

- [ ] **Step 5: Commit**

```bash
git add typescript/src/context/ typescript/tests/surgeon.test.ts
git commit -m "feat: add TypeScript context surgeon"
```

---

## Task 4: Budget Config + LedgerStore + InMemoryStore

**Files:**
- Create: `typescript/src/budget/config.ts`
- Create: `typescript/src/budget/ledger.ts`
- Create: `typescript/src/budget/memory-store.ts`
- Modify: `typescript/src/budget/index.ts`
- Create: `typescript/tests/budget.test.ts`

- [ ] **Step 1: Write failing tests**

Create `typescript/tests/budget.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { BudgetWindow, BudgetConfig } from '../src/budget/config.js';
import { LedgerEntry, makeEntry } from '../src/budget/ledger.js';
import { InMemoryStore } from '../src/budget/memory-store.js';
import { v4 as uuidv4 } from 'uuid';

function entry(sessionId: string, tokens = 100, cost = 0.01): LedgerEntry {
  return { id: uuidv4(), sessionId, userId: undefined, agentName: undefined, model: 'test', inputTokens: tokens, outputTokens: 0, costUsd: cost, timestamp: 0, tags: {} };
}

describe('BudgetConfig defaults', () => {
  it('has 80% alert threshold', () => {
    const cfg = new BudgetConfig();
    expect(cfg.alertAtPct).toBe(0.8);
    expect(cfg.alerts.logToStderr).toBe(true);
  });
});

describe('InMemoryStore', () => {
  it('records and queries by session', () => {
    const store = new InMemoryStore();
    store.record(entry('sess-1', 1000, 0.50));
    store.record(entry('sess-1', 500, 0.25));
    store.record(entry('sess-2', 999, 9.99));
    const report = store.usage('sess-1', BudgetWindow.PerSession);
    expect(report.totalTokens).toBe(1500);
    expect(report.totalCostUsd).toBeCloseTo(0.75, 3);
  });

  it('returns zero for unknown key', () => {
    const store = new InMemoryStore();
    const report = store.usage('nobody', BudgetWindow.PerSession);
    expect(report.totalTokens).toBe(0);
  });
});

describe('LedgerEntry.totalTokens', () => {
  it('sums input and output', () => {
    const e = entry('s', 100, 0.01);
    e.outputTokens = 50;
    expect(e.inputTokens + e.outputTokens).toBe(150);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd typescript && npm test tests/budget.test.ts
```

Expected: `Cannot find module '../src/budget/config.js'`

- [ ] **Step 3: Implement config.ts**

Create `typescript/src/budget/config.ts`:

```typescript
import { ModelTier } from '../types.js';

export enum BudgetWindow {
  PerSession = 'per_session',
  PerHour = 'per_hour',
  PerDay = 'per_day',
  PerMonth = 'per_month',
}

export interface BudgetLimit {
  window: BudgetWindow;
  tokens?: number;
  usd?: number;
}

export interface AlertConfig {
  slackWebhook?: string;
  logToStderr: boolean;
}

export class BudgetConfig {
  perSession?: BudgetLimit;
  perUser?: BudgetLimit;
  perAgent?: BudgetLimit;
  globalLimit?: BudgetLimit;
  alertAtPct: number = 0.8;
  alerts: AlertConfig = { logToStderr: true };
}
```

- [ ] **Step 4: Implement ledger.ts**

Create `typescript/src/budget/ledger.ts`:

```typescript
import { BudgetWindow } from './config.js';
import { UUID } from '../types.js';

export interface LedgerEntry {
  id: UUID;
  sessionId: string;
  userId?: string;
  agentName?: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  timestamp: number;
  tags: Record<string, string>;
}

export interface UsageReport {
  key: string;
  totalTokens: number;
  totalCostUsd: number;
  window: BudgetWindow;
  remainingTokens?: number;
  remainingUsd?: number;
  pctUsed: number;
}

export interface LedgerStore {
  record(entry: LedgerEntry): void;
  usage(key: string, window: BudgetWindow): UsageReport;
  flush(): void;
}

export function makeEntry(partial: Omit<LedgerEntry, 'id' | 'timestamp' | 'tags'> & Partial<Pick<LedgerEntry, 'id' | 'timestamp' | 'tags'>>): LedgerEntry {
  const { v4: uuidv4 } = require('uuid');
  return { id: uuidv4(), timestamp: 0, tags: {}, ...partial };
}
```

- [ ] **Step 5: Implement memory-store.ts**

Create `typescript/src/budget/memory-store.ts`:

```typescript
import { LedgerEntry, LedgerStore, UsageReport } from './ledger.js';
import { BudgetWindow } from './config.js';

export class InMemoryStore implements LedgerStore {
  private entries: LedgerEntry[] = [];

  record(entry: LedgerEntry): void {
    this.entries.push(entry);
  }

  usage(key: string, window: BudgetWindow): UsageReport {
    const matching = this.entries.filter(
      e => e.sessionId === key || e.userId === key || e.agentName === key,
    );
    const totalTokens = matching.reduce((s, e) => s + e.inputTokens + e.outputTokens, 0);
    const totalCostUsd = matching.reduce((s, e) => s + e.costUsd, 0);
    return { key, totalTokens, totalCostUsd, window, pctUsed: 0 };
  }

  flush(): void {}
}
```

Update `typescript/src/budget/index.ts`:

```typescript
export { BudgetWindow, BudgetLimit, AlertConfig, BudgetConfig } from './config.js';
export { LedgerEntry, UsageReport, LedgerStore } from './ledger.js';
export { InMemoryStore } from './memory-store.js';
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd typescript && npm test tests/budget.test.ts
```

Expected: `4 tests passed`

- [ ] **Step 7: Commit**

```bash
git add typescript/src/budget/config.ts typescript/src/budget/ledger.ts typescript/src/budget/memory-store.ts typescript/src/budget/index.ts typescript/tests/budget.test.ts
git commit -m "feat: add TypeScript budget config, LedgerStore, InMemoryStore"
```

---

## Task 5: SQLite Ledger Store

**Files:**
- Create: `typescript/src/budget/sqlite-store.ts`
- Modify: `typescript/src/budget/index.ts`
- Modify: `typescript/tests/budget.test.ts`

- [ ] **Step 1: Add SQLite tests to budget.test.ts**

Append to `typescript/tests/budget.test.ts`:

```typescript
import { SqliteLedgerStore } from '../src/budget/sqlite-store.js';
import { tmpdir } from 'os';
import { join } from 'path';
import { unlinkSync, existsSync } from 'fs';

describe('SqliteLedgerStore', () => {
  it('records and queries in-memory', () => {
    const store = new SqliteLedgerStore(':memory:');
    store.record(entry('sess-a', 100, 0.10));
    store.record(entry('sess-a', 200, 0.20));
    const report = store.usage('sess-a', BudgetWindow.PerSession);
    expect(report.totalTokens).toBe(300);
    expect(report.totalCostUsd).toBeCloseTo(0.30, 3);
  });

  it('survives reopen', () => {
    const path = join(tmpdir(), `truss-test-${Date.now()}.db`);
    try {
      const store1 = new SqliteLedgerStore(path);
      store1.record(entry('persistent', 500, 0.50));
      store1.flush();

      const store2 = new SqliteLedgerStore(path);
      const report = store2.usage('persistent', BudgetWindow.PerSession);
      expect(report.totalTokens).toBe(500);
    } finally {
      if (existsSync(path)) unlinkSync(path);
    }
  });
});
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd typescript && npm test tests/budget.test.ts
```

Expected: `Cannot find module '../src/budget/sqlite-store.js'`

- [ ] **Step 3: Implement sqlite-store.ts**

Create `typescript/src/budget/sqlite-store.ts`:

```typescript
import Database from 'better-sqlite3';
import { LedgerEntry, LedgerStore, UsageReport } from './ledger.js';
import { BudgetWindow } from './config.js';

export class SqliteLedgerStore implements LedgerStore {
  private db: Database.Database;

  constructor(path: string) {
    this.db = new Database(path);
    this.db.exec(`
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
    `);
  }

  record(entry: LedgerEntry): void {
    this.db.prepare(
      `INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?,?,?)`
    ).run(
      entry.id, entry.sessionId, entry.userId ?? null, entry.agentName ?? null,
      entry.model, entry.inputTokens, entry.outputTokens,
      entry.costUsd, entry.timestamp, JSON.stringify(entry.tags),
    );
  }

  usage(key: string, window: BudgetWindow): UsageReport {
    const row = this.db.prepare(
      `SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens,
              COALESCE(SUM(cost_usd), 0.0) AS cost
       FROM ledger
       WHERE session_id = ? OR user_id = ? OR agent_name = ?`
    ).get(key, key, key) as { tokens: number; cost: number };

    return {
      key,
      totalTokens: row.tokens,
      totalCostUsd: row.cost,
      window,
      pctUsed: 0,
    };
  }

  flush(): void {
    // better-sqlite3 auto-commits synchronous writes
  }
}
```

Update `typescript/src/budget/index.ts` — add:

```typescript
export { SqliteLedgerStore } from './sqlite-store.js';
```

- [ ] **Step 4: Run all budget tests**

```bash
cd typescript && npm test tests/budget.test.ts
```

Expected: `6 tests passed`

- [ ] **Step 5: Commit**

```bash
git add typescript/src/budget/sqlite-store.ts typescript/src/budget/index.ts typescript/tests/budget.test.ts
git commit -m "feat: add TypeScript SQLite ledger store"
```

---

## Task 6: Agent Handoff

**Files:**
- Create: `typescript/src/handoff/envelope.ts`
- Modify: `typescript/src/handoff/index.ts`
- Create: `typescript/tests/handoff.test.ts`

- [ ] **Step 1: Write failing tests**

Create `typescript/tests/handoff.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { makeEnvelope, makeContextBlock, ContextRole, ContextWeight, ModelTier } from '../src/types.js';
import { pack, unpack, BudgetCarve } from '../src/handoff/envelope.js';

function parentEnvelope() {
  const env = makeEnvelope('parent task');
  env.budgetUsdRemaining = 1.0;
  env.context.push(makeContextBlock(ContextRole.Task, ContextWeight.Critical, 'critical info', 'planner'));
  env.context.push(makeContextBlock(ContextRole.Background, ContextWeight.Background, 'background fluff', 'loader'));
  return env;
}

describe('pack', () => {
  it('filters context by weight', () => {
    const parent = parentEnvelope();
    const child = pack(parent, 'child task', [ContextWeight.Critical], BudgetCarve.fixedUsd(0.20));
    expect(child.context).toHaveLength(1);
    expect(child.context[0].weight).toBe(ContextWeight.Critical);
  });

  it('sets parentAgent to parent id', () => {
    const parent = parentEnvelope();
    const child = pack(parent, 'child', [ContextWeight.Critical], BudgetCarve.fixedUsd(0.1));
    expect(child.parentAgent).toBe(parent.id);
  });

  it('does not exceed parent budget', () => {
    const parent = parentEnvelope(); // 1.0 USD
    const child = pack(parent, 'child', [], BudgetCarve.fixedUsd(999));
    expect(child.budgetUsdRemaining).toBeLessThanOrEqual(1.0);
  });

  it('percent carve works', () => {
    const parent = parentEnvelope(); // 1.0 USD
    const child = pack(parent, 'child', [], BudgetCarve.percent(0.5));
    expect(child.budgetUsdRemaining).toBeCloseTo(0.5, 3);
  });

  it('inherits modelHint', () => {
    const parent = parentEnvelope();
    parent.modelHint = ModelTier.Premium;
    const child = pack(parent, 'child', [], BudgetCarve.fixedUsd(0.1));
    expect(child.modelHint).toBe(ModelTier.Premium);
  });
});

describe('unpack', () => {
  it('returns all context blocks', () => {
    const parent = parentEnvelope();
    expect(unpack(parent)).toHaveLength(parent.context.length);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd typescript && npm test tests/handoff.test.ts
```

Expected: `Cannot find module '../src/handoff/envelope.js'`

- [ ] **Step 3: Implement envelope.ts**

Create `typescript/src/handoff/envelope.ts`:

```typescript
import { AgentEnvelope, ContextBlock, ContextWeight, makeEnvelope } from '../types.js';

export class BudgetCarve {
  private constructor(private kind: 'usd' | 'pct' | 'tokens', private value: number) {}

  static fixedUsd(amount: number): BudgetCarve { return new BudgetCarve('usd', amount); }
  static percent(fraction: number): BudgetCarve { return new BudgetCarve('pct', fraction); }
  static fixedTokens(tokens: number): BudgetCarve { return new BudgetCarve('tokens', tokens); }

  apply(parentBudget: number): number {
    if (this.kind === 'usd') return Math.min(this.value, parentBudget);
    if (this.kind === 'pct') return parentBudget * this.value;
    return parentBudget * 0.5; // tokens→usd: safe 50% default
  }
}

export function pack(
  parent: AgentEnvelope,
  task: string,
  carryWeights: ContextWeight[],
  budgetCarve: BudgetCarve,
): AgentEnvelope {
  const child = makeEnvelope(task);
  child.context = parent.context.filter(b => carryWeights.includes(b.weight));
  child.budgetUsdRemaining = budgetCarve.apply(parent.budgetUsdRemaining);
  child.parentAgent = parent.id;
  child.modelHint = parent.modelHint;
  return child;
}

export function unpack(envelope: AgentEnvelope): ContextBlock[] {
  return [...envelope.context];
}
```

Update `typescript/src/handoff/index.ts`:

```typescript
export { pack, unpack, BudgetCarve } from './envelope.js';
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd typescript && npm test tests/handoff.test.ts
```

Expected: `6 tests passed`

- [ ] **Step 5: Commit**

```bash
git add typescript/src/handoff/ typescript/tests/handoff.test.ts
git commit -m "feat: add TypeScript agent handoff pack/unpack"
```

---

## Task 7: Circuit Breaker

**Files:**
- Create: `typescript/src/budget/circuit-breaker.ts`
- Modify: `typescript/src/budget/index.ts`
- Create: `typescript/tests/circuit-breaker.test.ts`

- [ ] **Step 1: Write failing tests**

Create `typescript/tests/circuit-breaker.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { CircuitBreaker, CircuitBreakerConfig, CircuitTrip } from '../src/budget/circuit-breaker.js';

describe('CircuitBreaker', () => {
  it('trips on rate limit', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxRequestsPerMinute: 3 }));
    expect(cb.checkAndRecord('p', 0.01, 0)).toBeNull();
    expect(cb.checkAndRecord('p', 0.01, 1000)).toBeNull();
    expect(cb.checkAndRecord('p', 0.01, 2000)).toBeNull();
    expect(cb.checkAndRecord('p', 0.01, 3000)).toBe(CircuitTrip.RateLimit);
  });

  it('trips on cost velocity', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxCostVelocityUsd: 0.50 }));
    cb.checkAndRecord('a', 0.40, 0);
    expect(cb.checkAndRecord('b', 0.20, 1000)).toBe(CircuitTrip.CostVelocity);
  });

  it('trips on repeated prompt', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ tripOnRepeatedPrompt: true }));
    cb.checkAndRecord('same prompt', 0.01, 0);
    expect(cb.checkAndRecord('same prompt', 0.01, 1000)).toBe(CircuitTrip.RepeatedPrompt);
  });

  it('different prompts do not trip', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ tripOnRepeatedPrompt: true }));
    cb.checkAndRecord('prompt A', 0.01, 0);
    expect(cb.checkAndRecord('prompt B', 0.01, 1000)).toBeNull();
  });

  it('trips on max retry depth', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxRetryDepth: 2 }));
    expect(cb.incrementRetry()).toBeNull();
    expect(cb.incrementRetry()).toBeNull();
    expect(cb.incrementRetry()).toBe(CircuitTrip.MaxRetryDepth);
  });

  it('resetRetry clears depth', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxRetryDepth: 1 }));
    cb.incrementRetry();
    cb.resetRetry();
    expect(cb.incrementRetry()).toBeNull();
  });

  it('evicts old requests from 60s window', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxRequestsPerMinute: 2 }));
    cb.checkAndRecord('a', 0.01, 0);
    cb.checkAndRecord('b', 0.01, 1000);
    expect(cb.checkAndRecord('c', 0.01, 61_000)).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd typescript && npm test tests/circuit-breaker.test.ts
```

Expected: `Cannot find module '../src/budget/circuit-breaker.js'`

- [ ] **Step 3: Implement circuit-breaker.ts**

Create `typescript/src/budget/circuit-breaker.ts`:

```typescript
export enum CircuitTrip {
  RateLimit = 'rate_limit',
  CostVelocity = 'cost_velocity',
  MaxRetryDepth = 'max_retry_depth',
  RepeatedPrompt = 'repeated_prompt',
}

export class CircuitBreakerConfig {
  maxRequestsPerMinute: number;
  maxCostVelocityUsd: number;
  maxRetryDepth: number;
  tripOnRepeatedPrompt: boolean;

  constructor(partial: Partial<CircuitBreakerConfig> = {}) {
    this.maxRequestsPerMinute = partial.maxRequestsPerMinute ?? 60;
    this.maxCostVelocityUsd = partial.maxCostVelocityUsd ?? 1.0;
    this.maxRetryDepth = partial.maxRetryDepth ?? 3;
    this.tripOnRepeatedPrompt = partial.tripOnRepeatedPrompt ?? true;
  }
}

interface Record { timestampMs: number; costUsd: number; promptHash: bigint }

function fnv1a(s: string): bigint {
  let h = 14695981039346656037n;
  for (const b of new TextEncoder().encode(s)) {
    h ^= BigInt(b);
    h = BigInt.asUintN(64, h * 1099511628211n);
  }
  return h;
}

export class CircuitBreaker {
  private window: Record[] = [];
  private retryDepth = 0;

  constructor(private config: CircuitBreakerConfig) {}

  checkAndRecord(prompt: string, costUsd: number, nowMs: number): CircuitTrip | null {
    const hash = fnv1a(prompt);
    const cutoff = nowMs - 60_000;
    this.window = this.window.filter(r => r.timestampMs >= cutoff);

    if (this.window.length >= this.config.maxRequestsPerMinute) return CircuitTrip.RateLimit;

    const runningCost = this.window.reduce((s, r) => s + r.costUsd, 0);
    if (runningCost + costUsd > this.config.maxCostVelocityUsd) return CircuitTrip.CostVelocity;

    if (this.config.tripOnRepeatedPrompt) {
      const recent = this.window.slice(-3);
      if (recent.some(r => r.promptHash === hash)) return CircuitTrip.RepeatedPrompt;
    }

    this.window.push({ timestampMs: nowMs, costUsd, promptHash: hash });
    return null;
  }

  incrementRetry(): CircuitTrip | null {
    this.retryDepth += 1;
    return this.retryDepth > this.config.maxRetryDepth ? CircuitTrip.MaxRetryDepth : null;
  }

  resetRetry(): void { this.retryDepth = 0; }
}
```

Append to `typescript/src/budget/index.ts`:

```typescript
export { CircuitBreaker, CircuitBreakerConfig, CircuitTrip } from './circuit-breaker.js';
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd typescript && npm test tests/circuit-breaker.test.ts
```

Expected: `7 tests passed`

- [ ] **Step 5: Commit**

```bash
git add typescript/src/budget/circuit-breaker.ts typescript/src/budget/index.ts typescript/tests/circuit-breaker.test.ts
git commit -m "feat: add TypeScript circuit breaker"
```

---

## Task 8: Checkpoints + Rollback

**Files:**
- Create: `typescript/src/coord/checkpoint.ts`
- Create: `typescript/src/coord/sqlite-checkpoint.ts`
- Modify: `typescript/src/coord/index.ts`
- Create: `typescript/tests/checkpoint.test.ts`

- [ ] **Step 1: Write failing tests**

Create `typescript/tests/checkpoint.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { v4 as uuidv4 } from 'uuid';
import { makeEnvelope } from '../src/types.js';
import { Checkpoint, InMemoryCheckpointStore } from '../src/coord/checkpoint.js';
import { SqliteCheckpointStore } from '../src/coord/sqlite-checkpoint.js';
import { CheckpointNotFound } from '../src/errors.js';
import { tmpdir } from 'os';
import { join } from 'path';
import { unlinkSync, existsSync } from 'fs';

function makeCheckpoint(sessionId: string, description: string): Checkpoint {
  return { id: uuidv4(), sessionId, agentName: 'test-agent', envelopeSnapshot: makeEnvelope('test task'), externalState: {}, createdAt: 0, description };
}

describe('InMemoryCheckpointStore', () => {
  it('save and load', () => {
    const store = new InMemoryCheckpointStore();
    const cp = makeCheckpoint('sess-1', 'after planning');
    const id = store.save(cp);
    expect(store.load(id).description).toBe('after planning');
  });

  it('rollback returns envelope', () => {
    const store = new InMemoryCheckpointStore();
    const cp = makeCheckpoint('sess-1', 'step 1');
    cp.envelopeSnapshot.task = 'original task';
    const id = store.save(cp);
    expect(store.rollback(id).task).toBe('original task');
  });

  it('load nonexistent throws', () => {
    const store = new InMemoryCheckpointStore();
    expect(() => store.load(uuidv4())).toThrow(CheckpointNotFound);
  });

  it('list filters by session', () => {
    const store = new InMemoryCheckpointStore();
    store.save(makeCheckpoint('sess-a', 'cp1'));
    store.save(makeCheckpoint('sess-b', 'cp2'));
    const list = store.list('sess-a');
    expect(list).toHaveLength(1);
    expect(list[0].description).toBe('cp1');
  });
});

describe('SqliteCheckpointStore', () => {
  it('save and load in-memory', () => {
    const store = new SqliteCheckpointStore(':memory:');
    const cp = makeCheckpoint('sess-1', 'sqlite test');
    const id = store.save(cp);
    const loaded = store.load(id);
    expect(loaded.description).toBe('sqlite test');
    expect(loaded.envelopeSnapshot.task).toBe('test task');
  });

  it('survives reopen', () => {
    const path = join(tmpdir(), `truss-cp-${Date.now()}.db`);
    try {
      const store1 = new SqliteCheckpointStore(path);
      const cp = makeCheckpoint('sess-p', 'persistent');
      const id = store1.save(cp);

      const store2 = new SqliteCheckpointStore(path);
      expect(store2.load(id).description).toBe('persistent');
    } finally {
      if (existsSync(path)) unlinkSync(path);
    }
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd typescript && npm test tests/checkpoint.test.ts
```

Expected: `Cannot find module '../src/coord/checkpoint.js'`

- [ ] **Step 3: Implement checkpoint.ts**

Create `typescript/src/coord/checkpoint.ts`:

```typescript
import { AgentEnvelope, UUID } from '../types.js';
import { CheckpointNotFound } from '../errors.js';

export interface CheckpointMeta {
  id: UUID;
  sessionId: string;
  agentName: string;
  description: string;
  createdAt: number;
}

export interface Checkpoint {
  id: UUID;
  sessionId: string;
  agentName: string;
  envelopeSnapshot: AgentEnvelope;
  externalState: Record<string, Uint8Array>;
  createdAt: number;
  description: string;
}

export interface CheckpointStore {
  save(cp: Checkpoint): UUID;
  load(id: UUID): Checkpoint;
  rollback(id: UUID): AgentEnvelope;
  list(sessionId: string): CheckpointMeta[];
}

export class InMemoryCheckpointStore implements CheckpointStore {
  private store = new Map<UUID, Checkpoint>();

  save(cp: Checkpoint): UUID {
    this.store.set(cp.id, cp);
    return cp.id;
  }

  load(id: UUID): Checkpoint {
    const cp = this.store.get(id);
    if (!cp) throw new CheckpointNotFound(id);
    return cp;
  }

  rollback(id: UUID): AgentEnvelope {
    return this.load(id).envelopeSnapshot;
  }

  list(sessionId: string): CheckpointMeta[] {
    return [...this.store.values()]
      .filter(cp => cp.sessionId === sessionId)
      .sort((a, b) => a.createdAt - b.createdAt)
      .map(cp => ({ id: cp.id, sessionId: cp.sessionId, agentName: cp.agentName, description: cp.description, createdAt: cp.createdAt }));
  }
}
```

- [ ] **Step 4: Implement sqlite-checkpoint.ts**

Create `typescript/src/coord/sqlite-checkpoint.ts`:

```typescript
import Database from 'better-sqlite3';
import { Checkpoint, CheckpointMeta, CheckpointStore } from './checkpoint.js';
import { AgentEnvelope, UUID } from '../types.js';
import { CheckpointNotFound } from '../errors.js';

export class SqliteCheckpointStore implements CheckpointStore {
  private db: Database.Database;

  constructor(path: string) {
    this.db = new Database(path);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS checkpoints (
        id            TEXT PRIMARY KEY,
        session_id    TEXT NOT NULL,
        agent_name    TEXT NOT NULL,
        description   TEXT NOT NULL,
        envelope_json TEXT NOT NULL,
        created_at    INTEGER NOT NULL
      )
    `);
  }

  save(cp: Checkpoint): UUID {
    this.db.prepare(
      `INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?)`
    ).run(cp.id, cp.sessionId, cp.agentName, cp.description, JSON.stringify(cp.envelopeSnapshot), cp.createdAt);
    return cp.id;
  }

  load(id: UUID): Checkpoint {
    const row = this.db.prepare(
      `SELECT id, session_id, agent_name, description, envelope_json, created_at FROM checkpoints WHERE id = ?`
    ).get(id) as any;
    if (!row) throw new CheckpointNotFound(id);
    return {
      id: row.id, sessionId: row.session_id, agentName: row.agent_name,
      description: row.description, envelopeSnapshot: JSON.parse(row.envelope_json),
      externalState: {}, createdAt: row.created_at,
    };
  }

  rollback(id: UUID): AgentEnvelope {
    return this.load(id).envelopeSnapshot;
  }

  list(sessionId: string): CheckpointMeta[] {
    const rows = this.db.prepare(
      `SELECT id, session_id, agent_name, description, created_at FROM checkpoints WHERE session_id = ? ORDER BY created_at`
    ).all(sessionId) as any[];
    return rows.map(r => ({ id: r.id, sessionId: r.session_id, agentName: r.agent_name, description: r.description, createdAt: r.created_at }));
  }
}
```

Update `typescript/src/coord/index.ts`:

```typescript
export { Checkpoint, CheckpointMeta, CheckpointStore, InMemoryCheckpointStore } from './checkpoint.js';
export { SqliteCheckpointStore } from './sqlite-checkpoint.js';
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd typescript && npm test tests/checkpoint.test.ts
```

Expected: `6 tests passed`

- [ ] **Step 6: Commit**

```bash
git add typescript/src/coord/ typescript/tests/checkpoint.test.ts
git commit -m "feat: add TypeScript checkpoint store with InMemory and SQLite"
```

---

## Task 9: Fence

**Files:**
- Create: `typescript/src/fence/memory-fence.ts`
- Modify: `typescript/src/fence/index.ts`
- Create: `typescript/tests/fence.test.ts`

- [ ] **Step 1: Write failing tests**

Create `typescript/tests/fence.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { InMemoryFence } from '../src/fence/memory-fence.js';
import { FenceLockConflict } from '../src/errors.js';

describe('InMemoryFence', () => {
  it('grants lock to first owner', () => {
    const fence = new InMemoryFence();
    expect(() => fence.acquire('doc-1', 'agent-a', 30_000, 0)).not.toThrow();
  });

  it('blocks second owner', () => {
    const fence = new InMemoryFence();
    fence.acquire('doc-1', 'agent-a', 30_000, 0);
    expect(() => fence.acquire('doc-1', 'agent-b', 30_000, 1000)).toThrow(FenceLockConflict);
  });

  it('expired lock can be reacquired', () => {
    const fence = new InMemoryFence();
    fence.acquire('doc-1', 'agent-a', 5_000, 0);
    expect(() => fence.acquire('doc-1', 'agent-b', 5_000, 30_000)).not.toThrow();
  });

  it('release frees lock', () => {
    const fence = new InMemoryFence();
    fence.acquire('doc-1', 'agent-a', 30_000, 0);
    fence.release('doc-1', 'agent-a');
    expect(() => fence.acquire('doc-1', 'agent-b', 30_000, 1000)).not.toThrow();
  });

  it('release by wrong owner is noop', () => {
    const fence = new InMemoryFence();
    fence.acquire('doc-1', 'agent-a', 30_000, 0);
    fence.release('doc-1', 'agent-b');
    expect(fence.isLocked('doc-1', 1000)).toBe(true);
  });

  it('isLocked returns false after expiry', () => {
    const fence = new InMemoryFence();
    fence.acquire('doc-1', 'agent-a', 5_000, 0);
    expect(fence.isLocked('doc-1', 6_000)).toBe(false);
  });

  it('owner can refresh own lock', () => {
    const fence = new InMemoryFence();
    fence.acquire('doc-1', 'agent-a', 30_000, 0);
    expect(() => fence.acquire('doc-1', 'agent-a', 30_000, 1000)).not.toThrow();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd typescript && npm test tests/fence.test.ts
```

Expected: `Cannot find module '../src/fence/memory-fence.js'`

- [ ] **Step 3: Implement memory-fence.ts**

Create `typescript/src/fence/memory-fence.ts`:

```typescript
import { FenceLockConflict } from '../errors.js';

export interface LockHandle {
  key: string;
  owner: string;
  acquiredAtMs: number;
  ttlMs: number;
}

function isExpired(handle: LockHandle, nowMs: number): boolean {
  return nowMs > handle.acquiredAtMs + handle.ttlMs;
}

export interface FenceStore {
  acquire(key: string, owner: string, ttlMs: number, nowMs: number): void;
  release(key: string, owner: string): void;
  isLocked(key: string, nowMs: number): boolean;
}

export class InMemoryFence implements FenceStore {
  private locks = new Map<string, LockHandle>();

  acquire(key: string, owner: string, ttlMs: number, nowMs: number): void {
    const handle = this.locks.get(key);
    if (handle && !isExpired(handle, nowMs) && handle.owner !== owner) {
      throw new FenceLockConflict(`${key} held by ${handle.owner}`);
    }
    this.locks.set(key, { key, owner, acquiredAtMs: nowMs, ttlMs });
  }

  release(key: string, owner: string): void {
    const handle = this.locks.get(key);
    if (handle?.owner === owner) this.locks.delete(key);
  }

  isLocked(key: string, nowMs: number): boolean {
    const handle = this.locks.get(key);
    return handle !== undefined && !isExpired(handle, nowMs);
  }
}
```

Update `typescript/src/fence/index.ts`:

```typescript
export { FenceStore, InMemoryFence, LockHandle } from './memory-fence.js';
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd typescript && npm test tests/fence.test.ts
```

Expected: `7 tests passed`

- [ ] **Step 5: Commit**

```bash
git add typescript/src/fence/ typescript/tests/fence.test.ts
git commit -m "feat: add TypeScript in-memory fence with TTL expiry"
```

---

## Task 10: Multi-LLM Router + MCP Interceptor

**Files:**
- Create: `typescript/src/router/router.ts`
- Modify: `typescript/src/router/index.ts`
- Create: `typescript/src/mcp/interceptor.ts`
- Modify: `typescript/src/mcp/index.ts`
- Create: `typescript/tests/router.test.ts`
- Create: `typescript/tests/mcp.test.ts`

- [ ] **Step 1: Write failing router tests**

Create `typescript/tests/router.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { ModelSpec, RouterConfig, RouterRule, route } from '../src/router/router.js';
import { ModelTier } from '../src/types.js';

const models: ModelSpec[] = [
  { name: 'claude-haiku-4-5', tier: ModelTier.Cheap, maxTokens: 8192, costPer1kInput: 0.001, costPer1kOutput: 0.005 },
  { name: 'claude-sonnet-4-6', tier: ModelTier.Standard, maxTokens: 16384, costPer1kInput: 0.003, costPer1kOutput: 0.015 },
  { name: 'claude-opus-4-8', tier: ModelTier.Premium, maxTokens: 32768, costPer1kInput: 0.015, costPer1kOutput: 0.075 },
];

describe('route', () => {
  it('applies matching keyword rule', () => {
    const config: RouterConfig = {
      models,
      rules: [{ keywords: ['summarise', 'summarize'], preferredTier: ModelTier.Cheap }],
      defaultTier: ModelTier.Standard,
    };
    expect(route('summarise this document', config).tier).toBe(ModelTier.Cheap);
  });

  it('falls back to default tier', () => {
    const config: RouterConfig = { models, rules: [], defaultTier: ModelTier.Standard };
    expect(route('analyse deeply', config).tier).toBe(ModelTier.Standard);
  });

  it('auto defaults to standard', () => {
    const config: RouterConfig = { models, rules: [], defaultTier: ModelTier.Auto };
    expect(route('any task', config).tier).toBe(ModelTier.Standard);
  });

  it('picks cheapest in tier', () => {
    const twoModels: ModelSpec[] = [
      { name: 'a', tier: ModelTier.Cheap, maxTokens: 4096, costPer1kInput: 0.002, costPer1kOutput: 0.010 },
      { name: 'b', tier: ModelTier.Cheap, maxTokens: 4096, costPer1kInput: 0.001, costPer1kOutput: 0.005 },
    ];
    const config: RouterConfig = { models: twoModels, rules: [], defaultTier: ModelTier.Cheap };
    expect(route('any', config).name).toBe('b');
  });
});
```

- [ ] **Step 2: Write failing MCP tests**

Create `typescript/tests/mcp.test.ts`:

```typescript
import { describe, it, expect, vi } from 'vitest';
import { McpManifest, McpInterceptor, McpCall } from '../src/mcp/interceptor.js';
import { ToolOutOfScope } from '../src/errors.js';

describe('McpInterceptor', () => {
  it('allows allowed tool', () => {
    const i = new McpInterceptor(new McpManifest(['read_file']));
    expect(() => i.check({ toolName: 'read_file', arguments: {} })).not.toThrow();
  });

  it('blocks denied tool', () => {
    const i = new McpInterceptor(new McpManifest(['read_file']));
    expect(() => i.check({ toolName: 'write_file', arguments: {} })).toThrow(ToolOutOfScope);
  });

  it('wrap calls fn on allowed', () => {
    const i = new McpInterceptor(new McpManifest(['tool_a']));
    const result = i.wrap({ toolName: 'tool_a', arguments: { x: 1 } }, c => (c.arguments as any).x * 2);
    expect(result).toBe(2);
  });

  it('wrap does not call fn on denied', () => {
    const fn = vi.fn();
    const i = new McpInterceptor(new McpManifest(['tool_a']));
    expect(() => i.wrap({ toolName: 'tool_b', arguments: {} }, fn)).toThrow(ToolOutOfScope);
    expect(fn).not.toHaveBeenCalled();
  });

  it('empty manifest denies all', () => {
    const i = new McpInterceptor(new McpManifest([]));
    expect(() => i.check({ toolName: 'anything', arguments: {} })).toThrow(ToolOutOfScope);
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd typescript && npm test tests/router.test.ts tests/mcp.test.ts
```

Expected: module not found errors for both.

- [ ] **Step 4: Implement router.ts**

Create `typescript/src/router/router.ts`:

```typescript
import { ModelTier } from '../types.js';

export interface ModelSpec {
  name: string;
  tier: ModelTier;
  maxTokens: number;
  costPer1kInput: number;
  costPer1kOutput: number;
}

export interface RouterRule {
  keywords: string[];
  preferredTier: ModelTier;
}

export interface RouterConfig {
  models: ModelSpec[];
  rules: RouterRule[];
  defaultTier: ModelTier;
}

export function route(task: string, config: RouterConfig): ModelSpec {
  const taskLower = task.toLowerCase();
  let tier = config.defaultTier;

  for (const rule of config.rules) {
    if (rule.keywords.some(kw => taskLower.includes(kw))) {
      tier = rule.preferredTier;
      break;
    }
  }

  if (tier === ModelTier.Auto) tier = ModelTier.Standard;

  const candidates = config.models.filter(m => m.tier === tier);
  const pool = candidates.length > 0 ? candidates : config.models;
  return pool.reduce((cheapest, m) =>
    m.costPer1kInput + m.costPer1kOutput < cheapest.costPer1kInput + cheapest.costPer1kOutput ? m : cheapest
  );
}
```

- [ ] **Step 5: Implement interceptor.ts**

Create `typescript/src/mcp/interceptor.ts`:

```typescript
import { ToolOutOfScope } from '../errors.js';

export interface McpCall {
  toolName: string;
  arguments: Record<string, unknown>;
}

export class McpManifest {
  constructor(public readonly allowedTools: string[]) {}
  isAllowed(toolName: string): boolean { return this.allowedTools.includes(toolName); }
}

export class McpInterceptor {
  constructor(private manifest: McpManifest) {}

  check(call: McpCall): void {
    if (!this.manifest.isAllowed(call.toolName)) {
      throw new ToolOutOfScope(`${call.toolName} denied by manifest`);
    }
  }

  wrap<T>(call: McpCall, fn: (call: McpCall) => T): T {
    this.check(call);
    return fn(call);
  }
}
```

Update barrel files:

`typescript/src/router/index.ts`:
```typescript
export { ModelSpec, RouterConfig, RouterRule, route } from './router.js';
```

`typescript/src/mcp/index.ts`:
```typescript
export { McpManifest, McpInterceptor, McpCall } from './interceptor.js';
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd typescript && npm test tests/router.test.ts tests/mcp.test.ts
```

Expected: `4 + 5 = 9 tests passed`

- [ ] **Step 7: Commit**

```bash
git add typescript/src/router/ typescript/src/mcp/ typescript/tests/router.test.ts typescript/tests/mcp.test.ts
git commit -m "feat: add TypeScript router and MCP interceptor"
```

---

## Task 11: Session + Public API + Hermes Example

**Files:**
- Create: `typescript/src/session.ts`
- Modify: `typescript/src/index.ts`
- Create: `typescript/tests/session.test.ts`
- Create: `typescript/examples/hermes.ts`

- [ ] **Step 1: Write failing session tests**

Create `typescript/tests/session.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { v4 as uuidv4 } from 'uuid';
import { makeContextBlock, ContextRole, ContextWeight } from '../src/types.js';
import { Session } from '../src/session.js';

function sampleBlocks(n = 10) {
  const weights = [ContextWeight.Critical, ContextWeight.High, ContextWeight.Normal, ContextWeight.Background];
  return Array.from({ length: n }, (_, i) =>
    makeContextBlock(ContextRole.Finding, weights[i % 4], `block ${i} ` + 'content '.repeat(20), `s${i}`)
  );
}

describe('Session', () => {
  it('compress returns result with tokensSaved >= 0', () => {
    const s = new Session({ targetTokens: 100, preserveRecent: 2 });
    s.compress(sampleBlocks(20));
    const report = s.report();
    expect(report.tokensSaved).toBeGreaterThanOrEqual(0);
    expect(report.tokensBefore).toBeGreaterThan(0);
  });

  it('tracks budget usage', () => {
    const s = new Session({ budgetUsd: 1.0 });
    s.recordUsage(500, 100, 0.05, 'test');
    const report = s.report();
    expect(report.budgetUsedUsd).toBeCloseTo(0.05, 3);
    expect(report.budgetLimitUsd).toBe(1.0);
  });

  it('checkpoint and rollback', () => {
    const s = new Session();
    s.envelope.task = 'original';
    const cpId = s.checkpoint('before change');
    s.envelope.task = 'modified';
    s.rollback(cpId);
    expect(s.envelope.task).toBe('original');
  });

  it('checkpoint count in report', () => {
    const s = new Session();
    s.checkpoint('cp1');
    s.checkpoint('cp2');
    expect(s.report().checkpointCount).toBe(2);
  });

  it('compress keeps critical blocks', () => {
    const critical = makeContextBlock(ContextRole.Task, ContextWeight.Critical, 'must keep this', 'agent');
    const bg = Array.from({ length: 20 }, () =>
      makeContextBlock(ContextRole.Background, ContextWeight.Background, 'noise '.repeat(50), 'x')
    );
    const s = new Session({ targetTokens: 50, preserveRecent: 0 });
    const result = s.compress([critical, ...bg]);
    expect(result.blocks.some(b => b.id === critical.id)).toBe(true);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd typescript && npm test tests/session.test.ts
```

Expected: `Cannot find module '../src/session.js'`

- [ ] **Step 3: Implement session.ts**

Create `typescript/src/session.ts`:

```typescript
import { v4 as uuidv4 } from 'uuid';
import { AgentEnvelope, ContextBlock, UUID, makeEnvelope } from './types.js';
import { BudgetWindow } from './budget/config.js';
import { LedgerEntry, LedgerStore, UsageReport } from './budget/ledger.js';
import { InMemoryStore } from './budget/memory-store.js';
import { SurgeonConfig, SurgeonResult, compress } from './context/surgeon.js';
import { Checkpoint, CheckpointStore, InMemoryCheckpointStore } from './coord/checkpoint.js';
import { FenceStore, InMemoryFence } from './fence/memory-fence.js';

const COST_PER_1K = 0.001;

export interface SessionReport {
  sessionId: string;
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  costSavedUsd: number;
  budgetUsedUsd: number;
  budgetLimitUsd?: number;
  checkpointCount: number;
}

export interface SessionOptions {
  envelope?: AgentEnvelope;
  budgetUsd?: number;
  targetTokens?: number;
  preserveRecent?: number;
  ledger?: LedgerStore;
  checkpointStore?: CheckpointStore;
  fence?: FenceStore;
}

export class Session {
  envelope: AgentEnvelope;
  private budgetUsd?: number;
  private tokensBefore = 0;
  private tokensAfter = 0;
  private ledger: LedgerStore;
  private checkpointStore: CheckpointStore;
  private fence: FenceStore;
  private surgeonConfig: SurgeonConfig;
  private cpCount = 0;

  constructor(opts: SessionOptions = {}) {
    this.envelope = opts.envelope ?? makeEnvelope('session');
    this.budgetUsd = opts.budgetUsd;
    this.ledger = opts.ledger ?? new InMemoryStore();
    this.checkpointStore = opts.checkpointStore ?? new InMemoryCheckpointStore();
    this.fence = opts.fence ?? new InMemoryFence();
    this.surgeonConfig = {
      strategy: 'hybrid' as any,
      targetTokens: opts.targetTokens ?? 8_000,
      preserveRecent: opts.preserveRecent ?? 5,
    };
  }

  get sessionId(): string { return this.envelope.id; }

  compress(blocks: ContextBlock[]): SurgeonResult {
    const result = compress(blocks, this.surgeonConfig);
    this.tokensBefore += result.tokensBefore;
    this.tokensAfter += result.tokensAfter;
    return result;
  }

  recordUsage(inputTokens: number, outputTokens: number, costUsd = 0, model = 'unknown'): void {
    const cost = costUsd || (inputTokens + outputTokens) / 1000 * COST_PER_1K;
    this.ledger.record({ id: uuidv4(), sessionId: this.sessionId, model, inputTokens, outputTokens, costUsd: cost, timestamp: 0, tags: {} });
  }

  checkpoint(description = ''): UUID {
    const cp: Checkpoint = {
      id: uuidv4(), sessionId: this.sessionId, agentName: 'session',
      envelopeSnapshot: JSON.parse(JSON.stringify(this.envelope)),
      externalState: {}, createdAt: 0, description,
    };
    this.cpCount += 1;
    return this.checkpointStore.save(cp);
  }

  rollback(checkpointId: UUID): void {
    this.envelope = this.checkpointStore.rollback(checkpointId);
  }

  report(): SessionReport {
    const tokensSaved = Math.max(0, this.tokensBefore - this.tokensAfter);
    const usage = this.ledger.usage(this.sessionId, BudgetWindow.PerSession);
    return {
      sessionId: this.sessionId,
      tokensBefore: this.tokensBefore,
      tokensAfter: this.tokensAfter,
      tokensSaved,
      costSavedUsd: tokensSaved / 1000 * COST_PER_1K,
      budgetUsedUsd: usage.totalCostUsd,
      budgetLimitUsd: this.budgetUsd,
      checkpointCount: this.cpCount,
    };
  }
}
```

- [ ] **Step 4: Run session tests to verify they pass**

```bash
cd typescript && npm test tests/session.test.ts
```

Expected: `5 tests passed`

- [ ] **Step 5: Write public index.ts**

Replace `typescript/src/index.ts`:

```typescript
export * from './errors.js';
export * from './types.js';
export { compress, SurgeonConfig, SurgeonResult, CompressionStrategy, scoreRelevance, detectContradiction } from './context/surgeon.js';
export { pack, unpack, BudgetCarve } from './handoff/envelope.js';
export { BudgetWindow, BudgetLimit, AlertConfig, BudgetConfig } from './budget/config.js';
export { LedgerEntry, UsageReport, LedgerStore } from './budget/ledger.js';
export { InMemoryStore } from './budget/memory-store.js';
export { SqliteLedgerStore } from './budget/sqlite-store.js';
export { CircuitBreaker, CircuitBreakerConfig, CircuitTrip } from './budget/circuit-breaker.js';
export { Checkpoint, CheckpointMeta, CheckpointStore, InMemoryCheckpointStore } from './coord/checkpoint.js';
export { SqliteCheckpointStore } from './coord/sqlite-checkpoint.js';
export { FenceStore, InMemoryFence, LockHandle } from './fence/memory-fence.js';
export { ModelSpec, RouterConfig, RouterRule, route } from './router/router.js';
export { McpManifest, McpInterceptor, McpCall } from './mcp/interceptor.js';
export { Session, SessionReport, SessionOptions } from './session.js';
```

- [ ] **Step 6: Run full test suite**

```bash
cd typescript && npm test
```

Expected: all tests pass (around 53 tests across all files).

- [ ] **Step 7: Write Hermes example**

Create `typescript/examples/hermes.ts`:

```typescript
/**
 * Hermes Agent — Truss TypeScript Phase 1 reference example.
 * Run: npx tsx examples/hermes.ts  (from typescript/)
 */
import {
  Session, makeContextBlock, makeEnvelope,
  ContextRole, ContextWeight, ModelTier,
  McpManifest, McpInterceptor,
  ModelSpec, RouterConfig, RouterRule, route,
  pack, BudgetCarve,
} from '../src/index.js';

const parent = makeEnvelope('Research cheapest S3-compatible cloud storage for 10TB');
parent.budgetUsdRemaining = 1.0;
parent.context = [
  makeContextBlock(ContextRole.Task, ContextWeight.Critical,
    'Find cheapest S3-compatible storage for 10TB dataset under $500/month.', 'user'),
  makeContextBlock(ContextRole.Constraint, ContextWeight.Critical,
    'Must be S3-compatible. Budget: $500/month max.', 'user'),
  makeContextBlock(ContextRole.Finding, ContextWeight.High,
    'Backblaze B2: $6/TB/month, S3-compatible.', 'search'),
  makeContextBlock(ContextRole.Finding, ContextWeight.High,
    'Cloudflare R2: $15/TB/month, zero egress fees.', 'search'),
  makeContextBlock(ContextRole.Finding, ContextWeight.Normal,
    'AWS S3 Standard: $23/TB/month, widest ecosystem.', 'search'),
  makeContextBlock(ContextRole.Background, ContextWeight.Background,
    'Cloud storage history dates to the 1960s mainframe era.', 'wikipedia'),
  makeContextBlock(ContextRole.Background, ContextWeight.Background,
    'IBM 305 RAMAC was the first commercial hard disk drive, 1956.', 'wikipedia'),
];

// Route to cheapest model for summarise task
const models: ModelSpec[] = [
  { name: 'claude-haiku-4-5', tier: ModelTier.Cheap, maxTokens: 8192, costPer1kInput: 0.001, costPer1kOutput: 0.005 },
  { name: 'claude-sonnet-4-6', tier: ModelTier.Standard, maxTokens: 16384, costPer1kInput: 0.003, costPer1kOutput: 0.015 },
];
const routerConfig: RouterConfig = {
  models,
  rules: [{ keywords: ['cheapest', 'summarise'], preferredTier: ModelTier.Cheap }],
  defaultTier: ModelTier.Standard,
};
const selectedModel = route('Find cheapest storage option', routerConfig);

// MCP interceptor
const interceptor = new McpInterceptor(new McpManifest(['search_web', 'read_url']));

// Pack child envelope
const child = pack(parent, 'Rank options by price', [ContextWeight.Critical, ContextWeight.High], BudgetCarve.percent(0.3));

// Run session
const session = new Session({ envelope: parent, budgetUsd: 1.0, targetTokens: 150, preserveRecent: 2 });
const result = session.compress(parent.context);
const cpId = session.checkpoint('after compression');
session.recordUsage(result.tokensAfter, 80, 0.005, selectedModel.name);

// MCP checks
try {
  interceptor.check({ toolName: 'search_web', arguments: { q: 'cheapest cloud storage' } });
  console.log('MCP: search_web allowed ✓');
} catch (e) { console.log(`MCP blocked: ${e}`); }

try {
  interceptor.check({ toolName: 'delete_file', arguments: { path: '/important.db' } });
} catch (e) { console.log(`MCP blocked delete_file ✓ (${e})`); }

console.log('\n=== Compression ===');
console.log(`Blocks: ${parent.context.length} → ${result.blocks.length} (saved ${result.tokensSaved} tokens)`);

console.log('\n=== Child Envelope ===');
console.log(`Task: ${child.task}`);
console.log(`Context blocks carried: ${child.context.length}`);
console.log(`Budget carved: $${child.budgetUsdRemaining.toFixed(2)}`);

const report = session.report();
console.log('\n=== Session Report ===');
console.log(`Session: ${report.sessionId.slice(0, 8)}`);
console.log(`Context: ${report.tokensBefore} → ${report.tokensAfter} tokens (saved ${report.tokensSaved})`);
console.log(`Budget: $${report.budgetUsedUsd.toFixed(4)} of $${report.budgetLimitUsd?.toFixed(2)} used`);
console.log(`Checkpoints: ${report.checkpointCount}`);
```

- [ ] **Step 8: Run example**

```bash
cd typescript && npx tsx examples/hermes.ts
```

Expected output:
```
MCP: search_web allowed ✓
MCP blocked delete_file ✓ (delete_file denied by manifest)

=== Compression ===
Blocks: 7 → N (saved M tokens)

=== Child Envelope ===
Task: Rank options by price
Context blocks carried: 4
Budget carved: $0.30

=== Session Report ===
Session: xxxxxxxx
Context: N → M tokens (saved K)
Budget: $0.0050 of $1.00 used
Checkpoints: 1
```

- [ ] **Step 9: Commit**

```bash
git add typescript/src/session.ts typescript/src/index.ts typescript/tests/session.test.ts typescript/examples/hermes.ts
git commit -m "feat: wire TypeScript public API, Session, and Hermes example"
```

---

## Self-Review Against Spec

| Spec requirement | Task |
|---|---|
| ContextBlock, AgentEnvelope, shared enums + makeContextBlock | Task 2 |
| estimateTokens = ceil(len/4) | Task 2 |
| Context surgeon: SlidingWindow, WeightedPrune, Hybrid | Task 3 |
| scoreRelevance / detectContradiction (keyword heuristics) | Task 3 |
| BudgetWindow, BudgetConfig, AlertConfig | Task 4 |
| LedgerStore interface + InMemoryStore | Task 4 |
| SqliteLedgerStore with better-sqlite3 | Task 5 |
| Agent handoff pack/unpack with BudgetCarve | Task 6 |
| Circuit breaker: rate, cost velocity, repeat, retry depth | Task 7 |
| InMemoryCheckpointStore + SqliteCheckpointStore | Task 8 |
| InMemoryFence with TTL expiry | Task 9 |
| Multi-LLM Router with keyword rules | Task 10 |
| MCP Interceptor with manifest allowlist | Task 10 |
| Session with compress/recordUsage/checkpoint/rollback/report | Task 11 |
| Hermes reference example | Task 11 |
| TrussError hierarchy | Task 2 |

**Gaps not in TypeScript Phase 1:**
- LangChain adapter (Python-only)
- SemanticDedup and Summarise strategies (Phase 2)
- Slack webhook alerting (stub field present, not wired)
- napi-rs native addon (was Rust Phase 2 plan; not applicable here)
