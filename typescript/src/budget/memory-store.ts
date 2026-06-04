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
