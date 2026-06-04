import { v4 as uuidv4 } from 'uuid';
import { AgentEnvelope, ContextBlock, UUID, makeEnvelope } from './types.js';
import { BudgetWindow } from './budget/config.js';
import { LedgerEntry, LedgerStore } from './budget/ledger.js';
import { InMemoryStore } from './budget/memory-store.js';
import { SurgeonConfig, SurgeonResult, compress, CompressionStrategy } from './context/surgeon.js';
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
      strategy: CompressionStrategy.Hybrid,
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
    this.ledger.record({
      id: uuidv4(), sessionId: this.sessionId, model, inputTokens, outputTokens,
      costUsd: cost, timestamp: 0, tags: {},
    });
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
