import { AgentEnvelope, ContextBlock, ContextWeight, makeEnvelope } from '../types.js';

export class BudgetCarve {
  private constructor(private kind: 'usd' | 'pct' | 'tokens', private value: number) {}

  static fixedUsd(amount: number): BudgetCarve { return new BudgetCarve('usd', amount); }
  static percent(fraction: number): BudgetCarve { return new BudgetCarve('pct', fraction); }
  static fixedTokens(tokens: number): BudgetCarve { return new BudgetCarve('tokens', tokens); }

  apply(parentBudget: number | null): number | null {
    if (parentBudget === null) {
      return this.kind === 'usd' ? this.value : null;
    }
    if (this.kind === 'usd') return Math.min(this.value, parentBudget);
    if (this.kind === 'pct') return parentBudget * this.value;
    return parentBudget * 0.5;
  }
}

export function pack(
  parent: AgentEnvelope,
  task: string,
  carryWeights: ContextWeight[],
  budgetCarve: BudgetCarve,
): AgentEnvelope {
  const child = makeEnvelope(task);
  child.context = parent.context.filter(b => carryWeights.includes(b.weight));
  child.budgetUsdRemaining = budgetCarve.apply(parent.budgetUsdRemaining);
  child.parentAgent = parent.id;
  child.modelHint = parent.modelHint;
  return child;
}

export function unpack(envelope: AgentEnvelope): ContextBlock[] {
  return [...envelope.context];
}
