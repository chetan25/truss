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
