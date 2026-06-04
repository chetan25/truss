import { describe, it, expect } from 'vitest';
import { makeEnvelope, makeContextBlock, ContextRole, ContextWeight, ModelTier } from '../src/types.js';
import { pack, unpack, BudgetCarve } from '../src/handoff/envelope.js';

function parentEnvelope() {
  const env = makeEnvelope('parent task');
  env.budgetUsdRemaining = 1.0;
  env.context.push(makeContextBlock(ContextRole.Task, ContextWeight.Critical, 'critical info', 'planner'));
  env.context.push(makeContextBlock(ContextRole.Background, ContextWeight.Background, 'background fluff', 'loader'));
  return env;
}

describe('pack', () => {
  it('filters context by weight', () => {
    const parent = parentEnvelope();
    const child = pack(parent, 'child task', [ContextWeight.Critical], BudgetCarve.fixedUsd(0.20));
    expect(child.context).toHaveLength(1);
    expect(child.context[0].weight).toBe(ContextWeight.Critical);
  });

  it('sets parentAgent to parent id', () => {
    const parent = parentEnvelope();
    const child = pack(parent, 'child', [ContextWeight.Critical], BudgetCarve.fixedUsd(0.1));
    expect(child.parentAgent).toBe(parent.id);
  });

  it('does not exceed parent budget', () => {
    const parent = parentEnvelope();
    const child = pack(parent, 'child', [], BudgetCarve.fixedUsd(999));
    expect(child.budgetUsdRemaining).not.toBeNull();
    expect(child.budgetUsdRemaining!).toBeLessThanOrEqual(1.0);
  });

  it('percent carve works', () => {
    const parent = parentEnvelope();
    const child = pack(parent, 'child', [], BudgetCarve.percent(0.5));
    expect(child.budgetUsdRemaining).toBeCloseTo(0.5, 3);
  });

  it('inherits modelHint', () => {
    const parent = parentEnvelope();
    parent.modelHint = ModelTier.Premium;
    const child = pack(parent, 'child', [], BudgetCarve.fixedUsd(0.1));
    expect(child.modelHint).toBe(ModelTier.Premium);
  });
});

describe('unpack', () => {
  it('returns all context blocks', () => {
    const parent = parentEnvelope();
    expect(unpack(parent)).toHaveLength(parent.context.length);
  });
});
