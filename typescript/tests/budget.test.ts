import { describe, it, expect, beforeAll } from 'vitest';
import { v4 as uuidv4 } from 'uuid';
import { BudgetWindow, BudgetConfig } from '../src/budget/config.js';
import { LedgerEntry } from '../src/budget/ledger.js';
import { InMemoryStore } from '../src/budget/memory-store.js';
import { SqliteLedgerStore } from '../src/budget/sqlite-store.js';

function entry(sessionId: string, tokens = 100, cost = 0.01): LedgerEntry {
  return { id: uuidv4(), sessionId, model: 'test', inputTokens: tokens, outputTokens: 0, costUsd: cost, timestamp: 0, tags: {} };
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

describe('LedgerEntry', () => {
  it('totalTokens sums input and output', () => {
    const e = entry('s', 100, 0.01);
    e.outputTokens = 50;
    expect(e.inputTokens + e.outputTokens).toBe(150);
  });
});

describe('SqliteLedgerStore', () => {
  beforeAll(async () => {
    await SqliteLedgerStore.init();
  });

  it('records and queries in-memory', () => {
    const store = new SqliteLedgerStore();
    store.record(entry('sess-a', 100, 0.10));
    store.record(entry('sess-a', 200, 0.20));
    const report = store.usage('sess-a', BudgetWindow.PerSession);
    expect(report.totalTokens).toBe(300);
    expect(report.totalCostUsd).toBeCloseTo(0.30, 3);
  });

  it('returns zero for unknown key', () => {
    const store = new SqliteLedgerStore();
    const report = store.usage('nobody', BudgetWindow.PerSession);
    expect(report.totalTokens).toBe(0);
  });

  it('can export and reimport data', () => {
    const store1 = new SqliteLedgerStore();
    store1.record(entry('persistent', 500, 0.50));
    const data = store1.export();

    const store2 = new SqliteLedgerStore(data);
    const report = store2.usage('persistent', BudgetWindow.PerSession);
    expect(report.totalTokens).toBe(500);
  });
});
