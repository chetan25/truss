import { ModelTier } from '../types.js';

export interface ModelSpec {
  name: string;
  tier: ModelTier;
  maxTokens: number;
  costPer1kInput: number;
  costPer1kOutput: number;
}

export interface RouterRule {
  keywords: string[];
  preferredTier: ModelTier;
}

export interface RouterConfig {
  models: ModelSpec[];
  rules: RouterRule[];
  defaultTier: ModelTier;
}

export function route(task: string, config: RouterConfig): ModelSpec {
  const taskLower = task.toLowerCase();
  let tier = config.defaultTier;

  for (const rule of config.rules) {
    if (rule.keywords.some(kw => taskLower.includes(kw))) {
      tier = rule.preferredTier;
      break;
    }
  }

  if (tier === ModelTier.Auto) tier = ModelTier.Standard;

  const candidates = config.models.filter(m => m.tier === tier);
  const pool = candidates.length > 0 ? candidates : config.models;
  return pool.reduce((cheapest, m) =>
    m.costPer1kInput + m.costPer1kOutput < cheapest.costPer1kInput + cheapest.costPer1kOutput ? m : cheapest
  );
}
