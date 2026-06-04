import { describe, it, expect, beforeAll } from 'vitest';
import { v4 as uuidv4 } from 'uuid';
import { makeEnvelope } from '../src/types.js';
import { Checkpoint, InMemoryCheckpointStore } from '../src/coord/checkpoint.js';
import { SqliteCheckpointStore } from '../src/coord/sqlite-checkpoint.js';
import { CheckpointNotFound } from '../src/errors.js';

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
  beforeAll(async () => {
    await SqliteCheckpointStore.init();
  });

  it('save and load in-memory', () => {
    const store = new SqliteCheckpointStore();
    const cp = makeCheckpoint('sess-1', 'sqlite test');
    const id = store.save(cp);
    const loaded = store.load(id);
    expect(loaded.description).toBe('sqlite test');
    expect(loaded.envelopeSnapshot.task).toBe('test task');
  });

  it('export and reimport', () => {
    const store1 = new SqliteCheckpointStore();
    const cp = makeCheckpoint('sess-p', 'persistent');
    const id = store1.save(cp);
    const data = store1.export();
    const store2 = new SqliteCheckpointStore(data);
    expect(store2.load(id).description).toBe('persistent');
  });
});
