import { FenceLockConflict } from '../errors.js';

export interface LockHandle {
  key: string;
  owner: string;
  acquiredAtMs: number;
  ttlMs: number;
}

function isExpired(h: LockHandle, nowMs: number): boolean {
  return nowMs > h.acquiredAtMs + h.ttlMs;
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
