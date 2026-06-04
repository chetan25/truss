import initSqlJs, { Database } from 'sql.js';
import { createRequire } from 'module';
import { readFileSync } from 'fs';
import { Checkpoint, CheckpointMeta, CheckpointStore } from './checkpoint.js';
import { AgentEnvelope, UUID } from '../types.js';
import { CheckpointNotFound } from '../errors.js';

let SQL: Awaited<ReturnType<typeof initSqlJs>> | null = null;

function getSQL() {
  if (!SQL) throw new Error('SqliteCheckpointStore: call await SqliteCheckpointStore.init() before use');
  return SQL;
}

export class SqliteCheckpointStore implements CheckpointStore {
  private db: Database;

  static async init(): Promise<void> {
    if (!SQL) {
      const require = createRequire(import.meta.url);
      const wasmPath = require.resolve('sql.js/dist/sql-wasm.wasm');
      const wasmBinary = readFileSync(wasmPath);
      SQL = await initSqlJs({ wasmBinary });
    }
  }

  constructor(data?: Uint8Array) {
    const sqlLib = getSQL();
    this.db = data ? new sqlLib.Database(data) : new sqlLib.Database();
    this.db.run(`
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
    this.db.run(
      `INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?)`,
      [cp.id, cp.sessionId, cp.agentName, cp.description, JSON.stringify(cp.envelopeSnapshot), cp.createdAt],
    );
    return cp.id;
  }

  load(id: UUID): Checkpoint {
    const stmt = this.db.prepare(`SELECT * FROM checkpoints WHERE id = ?`);
    stmt.bind([id]);
    const found = stmt.step();
    const row = found ? stmt.getAsObject() as any : null;
    stmt.free();
    if (!row) throw new CheckpointNotFound(id);
    return {
      id: row.id, sessionId: row.session_id, agentName: row.agent_name,
      description: row.description, envelopeSnapshot: JSON.parse(row.envelope_json),
      externalState: {}, createdAt: Number(row.created_at),
    };
  }

  rollback(id: UUID): AgentEnvelope {
    return this.load(id).envelopeSnapshot;
  }

  list(sessionId: string): CheckpointMeta[] {
    const stmt = this.db.prepare(
      `SELECT id, session_id, agent_name, description, created_at FROM checkpoints WHERE session_id = ? ORDER BY created_at`
    );
    stmt.bind([sessionId]);
    const results: CheckpointMeta[] = [];
    while (stmt.step()) {
      const r = stmt.getAsObject() as any;
      results.push({ id: r.id, sessionId: r.session_id, agentName: r.agent_name, description: r.description, createdAt: Number(r.created_at) });
    }
    stmt.free();
    return results;
  }

  export(): Uint8Array { return this.db.export(); }
}
