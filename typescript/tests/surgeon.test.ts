import { describe, it, expect } from 'vitest';
import { makeContextBlock, ContextRole, ContextWeight } from '../src/types.js';
import { compress, SurgeonConfig, CompressionStrategy, scoreRelevance, detectContradiction } from '../src/context/surgeon.js';

function makeBlock(weight: ContextWeight, tokens: number, content = ''): ReturnType<typeof makeContextBlock> {
  const b = makeContextBlock(ContextRole.Finding, weight, content || 'x '.repeat(tokens), 'test');
  b.tokenCount = tokens;
  return b;
}

describe('sliding window', () => {
  it('keeps recent N blocks within token budget', () => {
    const blocks = Array.from({ length: 10 }, (_, i) => makeBlock(ContextWeight.Normal, 100, `block ${i}`));
    const config: SurgeonConfig = { strategy: CompressionStrategy.SlidingWindow, targetTokens: 300, preserveRecent: 3, keepRecent: 3 };
    const result = compress(blocks, config);
    expect(result.tokensAfter).toBeLessThanOrEqual(300);
  });

  it('keep_recent=0 returns all blocks', () => {
    const blocks = Array.from({ length: 5 }, (_, i) => makeBlock(ContextWeight.Normal, 100, `b${i}`));
    const config: SurgeonConfig = { strategy: CompressionStrategy.SlidingWindow, targetTokens: 300, preserveRecent: 0, keepRecent: 0 };
    const result = compress(blocks, config);
    expect(result.blocks).toHaveLength(5);
  });
});

describe('weighted prune', () => {
  it('drops background before normal', () => {
    const bg = makeBlock(ContextWeight.Background, 500, 'background');
    const normal = makeBlock(ContextWeight.Normal, 500, 'normal');
    const critical = makeBlock(ContextWeight.Critical, 100, 'critical');
    const config: SurgeonConfig = { strategy: CompressionStrategy.WeightedPrune, targetTokens: 700, preserveRecent: 0 };
    const result = compress([bg, normal, critical], config);
    const ids = new Set(result.blocks.map(b => b.id));
    expect(ids.has(bg.id)).toBe(false);
    expect(ids.has(critical.id)).toBe(true);
  });

  it('never removes critical blocks', () => {
    const critical = makeBlock(ContextWeight.Critical, 9000, 'must keep');
    const bg = makeBlock(ContextWeight.Background, 100, 'droppable');
    const config: SurgeonConfig = { strategy: CompressionStrategy.Hybrid, targetTokens: 500, preserveRecent: 0 };
    const result = compress([critical, bg], config);
    expect(result.blocks.some(b => b.id === critical.id)).toBe(true);
  });
});

describe('SurgeonResult', () => {
  it('tokensSaved equals before minus after', () => {
    const blocks = Array.from({ length: 4 }, () => makeBlock(ContextWeight.Background, 500));
    const config: SurgeonConfig = { strategy: CompressionStrategy.WeightedPrune, targetTokens: 500, preserveRecent: 0 };
    const result = compress(blocks, config);
    expect(result.tokensSaved).toBe(result.tokensBefore - result.tokensAfter);
  });
});

describe('scoreRelevance', () => {
  it('returns 0 for empty task', () => {
    expect(scoreRelevance(makeBlock(ContextWeight.Normal, 10, 'content'), '')).toBe(0);
  });

  it('returns 1 for exact match', () => {
    const b = makeBlock(ContextWeight.Normal, 10, 'pricing cloud storage');
    expect(scoreRelevance(b, 'pricing cloud storage')).toBe(1.0);
  });
});

describe('detectContradiction', () => {
  it('catches not-X pattern', () => {
    const a = makeBlock(ContextWeight.Normal, 10, 'the service is available');
    const b = makeBlock(ContextWeight.Normal, 10, 'the service is not available');
    expect(detectContradiction(a, b)).toBe(true);
  });

  it('no false positive for unrelated blocks', () => {
    const a = makeBlock(ContextWeight.Normal, 10, 'the weather is sunny today');
    const b = makeBlock(ContextWeight.Normal, 10, 'the price is twenty dollars');
    expect(detectContradiction(a, b)).toBe(false);
  });

  it('no false positive on shared vocab with incidental negation', () => {
    const a = makeBlock(ContextWeight.Normal, 10, 'do not press the button');
    const b = makeBlock(ContextWeight.Normal, 10, 'press the lever to open');
    expect(detectContradiction(a, b)).toBe(false);
  });
});
