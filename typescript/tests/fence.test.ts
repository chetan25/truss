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
