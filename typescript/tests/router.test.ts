import { describe, it, expect } from 'vitest';
import { ModelSpec, RouterConfig, RouterRule, route } from '../src/router/router.js';
import { ModelTier } from '../src/types.js';

const models: ModelSpec[] = [
  { name: 'claude-haiku-4-5', tier: ModelTier.Cheap, maxTokens: 8192, costPer1kInput: 0.001, costPer1kOutput: 0.005 },
  { name: 'claude-sonnet-4-6', tier: ModelTier.Standard, maxTokens: 16384, costPer1kInput: 0.003, costPer1kOutput: 0.015 },
  { name: 'claude-opus-4-8', tier: ModelTier.Premium, maxTokens: 32768, costPer1kInput: 0.015, costPer1kOutput: 0.075 },
];

describe('route', () => {
  it('applies matching keyword rule', () => {
    const config: RouterConfig = { models, rules: [{ keywords: ['summarise'], preferredTier: ModelTier.Cheap }], defaultTier: ModelTier.Standard };
    expect(route('summarise this document', config).tier).toBe(ModelTier.Cheap);
  });

  it('falls back to default tier', () => {
    const config: RouterConfig = { models, rules: [], defaultTier: ModelTier.Standard };
    expect(route('analyse', config).tier).toBe(ModelTier.Standard);
  });

  it('auto defaults to standard', () => {
    const config: RouterConfig = { models, rules: [], defaultTier: ModelTier.Auto };
    expect(route('task', config).tier).toBe(ModelTier.Standard);
  });

  it('picks cheapest in tier', () => {
    const twoModels: ModelSpec[] = [
      { name: 'a', tier: ModelTier.Cheap, maxTokens: 4096, costPer1kInput: 0.002, costPer1kOutput: 0.010 },
      { name: 'b', tier: ModelTier.Cheap, maxTokens: 4096, costPer1kInput: 0.001, costPer1kOutput: 0.005 },
    ];
    const config: RouterConfig = { models: twoModels, rules: [], defaultTier: ModelTier.Cheap };
    expect(route('task', config).name).toBe('b');
  });
});
