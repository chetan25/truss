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
