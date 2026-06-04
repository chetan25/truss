import { describe, it, expect } from 'vitest';
import { v4 as uuidv4 } from 'uuid';
import { makeContextBlock, ContextRole, ContextWeight } from '../src/types.js';
import { Session } from '../src/session.js';

function sampleBlocks(n = 10) {
  const weights = [ContextWeight.Critical, ContextWeight.High, ContextWeight.Normal, ContextWeight.Background];
  return Array.from({ length: n }, (_, i) =>
    makeContextBlock(ContextRole.Finding, weights[i % 4], `block ${i} ` + 'content '.repeat(20), `s${i}`)
  );
}

describe('Session', () => {
  it('compress returns result', () => {
    const s = new Session({ targetTokens: 100, preserveRecent: 2 });
    s.compress(sampleBlocks(20));
    const report = s.report();
    expect(report.tokensBefore).toBeGreaterThan(0);
  });

  it('tracks budget usage', () => {
    const s = new Session({ budgetUsd: 1.0 });
    s.recordUsage(500, 100, 0.05, 'test');
    expect(s.report().budgetUsedUsd).toBeCloseTo(0.05, 3);
    expect(s.report().budgetLimitUsd).toBe(1.0);
  });

  it('checkpoint and rollback', () => {
    const s = new Session();
    s.envelope.task = 'original';
    const cpId = s.checkpoint('before change');
    s.envelope.task = 'modified';
    s.rollback(cpId);
    expect(s.envelope.task).toBe('original');
  });

  it('checkpoint count in report', () => {
    const s = new Session();
    s.checkpoint('cp1');
    s.checkpoint('cp2');
    expect(s.report().checkpointCount).toBe(2);
  });

  it('compress keeps critical blocks', () => {
    const critical = makeContextBlock(ContextRole.Task, ContextWeight.Critical, 'must keep', 'agent');
    const bg = Array.from({ length: 20 }, () =>
      makeContextBlock(ContextRole.Background, ContextWeight.Background, 'noise '.repeat(50), 'x')
    );
    const s = new Session({ targetTokens: 50, preserveRecent: 0 });
    const result = s.compress([critical, ...bg]);
    expect(result.blocks.some(b => b.id === critical.id)).toBe(true);
  });
});
