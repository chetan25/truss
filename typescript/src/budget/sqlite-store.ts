import initSqlJs, { Database } from 'sql.js';
import { readFileSync } from 'fs';
import { createRequire } from 'module';
import { dirname, join } from 'path';
import { LedgerEntry, LedgerStore, UsageReport } from './ledger.js';
import { BudgetWindow } from './config.js';

// Resolve the sql-wasm.wasm binary path at module load time.
// createRequire lets us use Node's require.resolve in an ESM context.
const _require = createRequire(import.meta.url);
const _sqlJsPath = _require.resolve('sql.js');
const _wasmPath = join(dirname(_sqlJsPath), 'sql-wasm.wasm');

let SQL: Awaited<ReturnType<typeof initSqlJs>> | null = null;

function getSQL() {
  if (!SQL) {
    throw new Error('SqliteLedgerStore: call await SqliteLedgerStore.init() before use');
  }
  return SQL;
}

export class SqliteLedgerStore implements LedgerStore {
  private db: Database;

  /**
   * Initialise the sql.js WASM engine. Must be awaited once before
   * constructing any SqliteLedgerStore instances.
   */
  static async init(): Promise<void> {
    if (!SQL) {
      const wasmBinary = readFileSync(_wasmPath);
      SQL = await initSqlJs({ wasmBinary });
    }
  }

  constructor(data?: Uint8Array) {
    const sqlLib = getSQL();
    this.db = data ? new sqlLib.Database(data) : new sqlLib.Database();
    this.db.run(`
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
    this.db.run(
      `INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?,?,?)`,
      [
        entry.id,
        entry.sessionId,
        entry.userId ?? null,
        entry.agentName ?? null,
        entry.model,
        entry.inputTokens,
        entry.outputTokens,
        entry.costUsd,
        entry.timestamp,
        JSON.stringify(entry.tags),
      ],
    );
  }

  usage(key: string, window: BudgetWindow): UsageReport {
    const stmt = this.db.prepare(
      `SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens,
              COALESCE(SUM(cost_usd), 0.0) AS cost
       FROM ledger
       WHERE session_id = $key OR user_id = $key OR agent_name = $key`,
    );
    stmt.bind({ $key: key });
    stmt.step();
    const row = stmt.getAsObject() as { tokens: number; cost: number };
    stmt.free();
    return {
      key,
      totalTokens: Number(row.tokens),
      totalCostUsd: Number(row.cost),
      window,
      pctUsed: 0,
    };
  }

  /** Export the in-memory database as a Uint8Array for persistence. */
  export(): Uint8Array {
    return this.db.export();
  }

  flush(): void {}
}
