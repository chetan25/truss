import { describe, it, expect } from 'vitest';
import { BudgetExceeded, ToolOutOfScope, CheckpointNotFound, FenceLockConflict } from '../src/errors.js';
import {
  estimateTokens, makeContextBlock, makeEnvelope,
  ContextRole, ContextWeight, ModelTier,
} from '../src/types.js';

describe('estimateTokens', () => {
  it('ceiling-divides by 4', () => {
    expect(estimateTokens('Hello')).toBe(2);
    expect(estimateTokens('Hello world')).toBe(3);
    expect(estimateTokens('')).toBe(0);
  });
});

describe('makeContextBlock', () => {
  it('auto-estimates token count', () => {
    const b = makeContextBlock(ContextRole.Task, ContextWeight.Critical, 'Hello world', 'test');
    expect(b.tokenCount).toBe(3);
  });

  it('has unique id', () => {
    const a = makeContextBlock(ContextRole.Task, ContextWeight.Normal, 'a', 'test');
    const b = makeContextBlock(ContextRole.Task, ContextWeight.Normal, 'a', 'test');
    expect(a.id).not.toBe(b.id);
  });
});

describe('ContextWeight', () => {
  it('is numerically comparable', () => {
    expect(ContextWeight.Critical).toBeGreaterThan(ContextWeight.Normal);
    expect(ContextWeight.Background).toBeLessThan(ContextWeight.High);
  });
});

describe('makeEnvelope', () => {
  it('starts with no checkpoint', () => {
    const env = makeEnvelope('test task');
    expect(env.checkpointId).toBeUndefined();
  });

  it('is JSON-serializable', () => {
    const env = makeEnvelope('analyse pricing');
    const json = JSON.stringify(env);
    const back = JSON.parse(json);
    expect(back.task).toBe(env.task);
    expect(back.id).toBe(env.id);
  });
});

describe('errors', () => {
  it('BudgetExceeded is instanceof Error', () => {
    const e = new BudgetExceeded('wallet exceeded $5');
    expect(e).toBeInstanceOf(Error);
    expect(e.message).toContain('wallet');
  });

  it('ToolOutOfScope includes tool name', () => {
    const e = new ToolOutOfScope('readFile denied');
    expect(e.message).toContain('readFile');
  });
});
