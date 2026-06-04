import { ContextBlock, ContextWeight } from '../types.js';

export enum CompressionStrategy {
  SlidingWindow = 'sliding_window',
  WeightedPrune = 'weighted_prune',
  Hybrid = 'hybrid',
}

export interface SurgeonConfig {
  strategy: CompressionStrategy;
  targetTokens: number;
  preserveRecent: number;
  keepRecent?: number;
}

export interface SurgeonResult {
  blocks: ContextBlock[];
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  strategyApplied: string;
}

export function compress(blocks: ContextBlock[], config: SurgeonConfig): SurgeonResult {
  const tokensBefore = blocks.reduce((s, b) => s + b.tokenCount, 0);
  let kept: ContextBlock[];

  if (config.strategy === CompressionStrategy.SlidingWindow) {
    kept = slidingWindow(blocks, config.keepRecent ?? config.preserveRecent, config.preserveRecent);
  } else if (config.strategy === CompressionStrategy.WeightedPrune) {
    kept = weightedPrune(blocks, config.targetTokens, config.preserveRecent);
  } else {
    const afterPrune = weightedPrune(blocks, config.targetTokens, config.preserveRecent);
    const total = afterPrune.reduce((s, b) => s + b.tokenCount, 0);
    kept = total > config.targetTokens
      ? slidingWindow(afterPrune, config.keepRecent ?? config.preserveRecent, config.preserveRecent)
      : afterPrune;
  }

  const tokensAfter = kept.reduce((s, b) => s + b.tokenCount, 0);
  return {
    blocks: kept,
    tokensBefore,
    tokensAfter,
    tokensSaved: tokensBefore - tokensAfter,
    strategyApplied: config.strategy,
  };
}

function slidingWindow(blocks: ContextBlock[], keepRecent: number, preserveRecent: number): ContextBlock[] {
  if (keepRecent === 0) return [...blocks];
  const alwaysKeep = Math.max(preserveRecent, keepRecent);
  if (blocks.length <= alwaysKeep) return [...blocks];

  const pinned = blocks.filter(b => b.weight >= ContextWeight.High);
  const pinnedIds = new Set(pinned.map(b => b.id));
  const recentStart = Math.max(0, blocks.length - keepRecent);
  const result = [...pinned];
  for (const b of blocks.slice(recentStart)) {
    if (!pinnedIds.has(b.id)) result.push(b);
  }
  return result;
}

function weightedPrune(blocks: ContextBlock[], targetTokens: number, preserveRecent: number): ContextBlock[] {
  const total = blocks.reduce((s, b) => s + b.tokenCount, 0);
  if (total <= targetTokens) return [...blocks];

  const recentSlice = preserveRecent > 0 ? blocks.slice(-preserveRecent) : [];
  const preserveIds = new Set(recentSlice.map(b => b.id));
  const removable = blocks
    .filter(b => !preserveIds.has(b.id) && b.weight < ContextWeight.High)
    .sort((a, b) => a.weight !== b.weight ? a.weight - b.weight : a.createdAt - b.createdAt);

  const toRemove = new Set<string>();
  let running = total;
  for (const b of removable) {
    if (running <= targetTokens) break;
    running -= b.tokenCount;
    toRemove.add(b.id);
  }
  return blocks.filter(b => !toRemove.has(b.id));
}

export function scoreRelevance(block: ContextBlock, task: string): number {
  const taskWords = new Set(task.split(/\s+/).filter(Boolean));
  if (taskWords.size === 0) return 0;
  const matches = block.content.split(/\s+/).filter(w => taskWords.has(w)).length;
  return Math.min(matches / taskWords.size, 1.0);
}

export function detectContradiction(a: ContextBlock, b: ContextBlock): boolean {
  const aLow = a.content.toLowerCase();
  const bLow = b.content.toLowerCase();
  for (const word of aLow.split(/\s+/)) {
    if (word.length > 4 && bLow.includes(`not ${word}`)) return true;
  }
  return false;
}
