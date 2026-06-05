import { describe, it, expect, vi } from 'vitest';
import {
  COST_TABLE, computeCost,
  LLMMessage, LLMUsage, LLMResponse,
} from '../src/providers/base.js';

describe('COST_TABLE', () => {
  it('has Anthropic models', () => {
    expect(COST_TABLE['claude-haiku-4-5']).toBeDefined();
    expect(COST_TABLE['claude-sonnet-4-6']).toBeDefined();
    expect(COST_TABLE['claude-opus-4-8']).toBeDefined();
  });

  it('has OpenAI models', () => {
    expect(COST_TABLE['gpt-4o']).toBeDefined();
    expect(COST_TABLE['gpt-4o-mini']).toBeDefined();
  });
});

describe('computeCost', () => {
  it('calculates cost for known model', () => {
    // claude-haiku-4-5: $0.001/1k in, $0.005/1k out
    const cost = computeCost('claude-haiku-4-5', 1000, 1000);
    expect(Math.abs(cost - 0.006)).toBeLessThan(0.0001);
  });

  it('uses default rates for unknown model', () => {
    const cost = computeCost('unknown-model-xyz', 1000, 0);
    expect(cost).toBeCloseTo(0.001, 5);
  });
});

describe('base types', () => {
  it('LLMMessage has role and content', () => {
    const msg: LLMMessage = { role: 'user', content: 'hello' };
    expect(msg.role).toBe('user');
  });

  it('LLMUsage sums tokens', () => {
    const usage: LLMUsage = { inputTokens: 100, outputTokens: 50, costUsd: 0.01 };
    expect(usage.inputTokens + usage.outputTokens).toBe(150);
  });

  it('LLMResponse has optional raw', () => {
    const resp: LLMResponse = { text: 'hi', model: 'test', usage: { inputTokens: 1, outputTokens: 1, costUsd: 0 } };
    expect(resp.raw).toBeUndefined();
  });
});

import { AnthropicProvider } from '../src/providers/anthropic.js';
import { Session } from '../src/session.js';

function makeAnthropicMock(text = 'Hello!', inputTokens = 10, outputTokens = 5) {
  return {
    messages: {
      create: vi.fn().mockResolvedValue({
        content: [{ text }],
        usage: { input_tokens: inputTokens, output_tokens: outputTokens },
      }),
    },
  };
}

describe('AnthropicProvider', () => {
  it('complete returns LLMResponse', async () => {
    const provider = new AnthropicProvider({ apiKey: 'test-key' });
    (provider as any)._client = makeAnthropicMock('Test reply', 100, 50);

    const result = await provider.complete(
      [{ role: 'user', content: 'hello' }],
      'claude-haiku-4-5',
    );
    expect(result.text).toBe('Test reply');
    expect(result.usage.inputTokens).toBe(100);
    expect(result.usage.outputTokens).toBe(50);
    expect(result.usage.costUsd).toBeGreaterThan(0);
  });

  it('records usage to session', async () => {
    const session = new Session();
    const provider = new AnthropicProvider({ apiKey: 'test-key', session });
    (provider as any)._client = makeAnthropicMock('reply', 200, 100);

    await provider.complete([{ role: 'user', content: 'hi' }], 'claude-haiku-4-5');

    const report = session.report();
    expect(report.budgetUsedUsd).toBeGreaterThan(0);
  });

  it('throws BudgetExceeded when circuit breaker trips', async () => {
    const { CircuitBreaker, CircuitBreakerConfig } = await import('../src/budget/circuit-breaker.js');
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxRequestsPerMinute: 0 }));
    const provider = new AnthropicProvider({ apiKey: 'test-key', circuitBreaker: cb });
    (provider as any)._client = makeAnthropicMock();

    const { BudgetExceeded } = await import('../src/errors.js');
    await expect(
      provider.complete([{ role: 'user', content: 'hi' }], 'claude-haiku-4-5'),
    ).rejects.toThrow(BudgetExceeded);
  });

  it('throws if API key missing', () => {
    const origEnv = process.env.ANTHROPIC_API_KEY;
    delete process.env.ANTHROPIC_API_KEY;
    try {
      expect(() => new AnthropicProvider({})).toThrow('ANTHROPIC_API_KEY');
    } finally {
      if (origEnv) process.env.ANTHROPIC_API_KEY = origEnv;
    }
  });
});

import { OpenAIProvider } from '../src/providers/openai.js';

function makeOpenAIMock(text = 'OpenAI reply', promptTokens = 10, completionTokens = 5) {
  return {
    chat: {
      completions: {
        create: vi.fn().mockResolvedValue({
          choices: [{ message: { content: text } }],
          usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens },
          model: 'gpt-4o-mini',
        }),
      },
    },
  };
}

describe('OpenAIProvider', () => {
  it('complete returns LLMResponse', async () => {
    const provider = new OpenAIProvider({ apiKey: 'test-key' });
    (provider as any)._client = makeOpenAIMock('OpenAI response', 80, 40);

    const result = await provider.complete([{ role: 'user', content: 'hello' }], 'gpt-4o-mini');
    expect(result.text).toBe('OpenAI response');
    expect(result.usage.inputTokens).toBe(80);
    expect(result.usage.outputTokens).toBe(40);
    expect(result.usage.costUsd).toBeGreaterThan(0);
  });

  it('records usage to session', async () => {
    const session = new Session();
    const provider = new OpenAIProvider({ apiKey: 'test-key', session });
    (provider as any)._client = makeOpenAIMock('reply', 150, 75);

    await provider.complete([{ role: 'user', content: 'hi' }], 'gpt-4o-mini');
    expect(session.report().budgetUsedUsd).toBeGreaterThan(0);
  });

  it('throws if API key missing', () => {
    const origEnv = process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_KEY;
    try {
      expect(() => new OpenAIProvider({})).toThrow('OPENAI_API_KEY');
    } finally {
      if (origEnv) process.env.OPENAI_API_KEY = origEnv;
    }
  });
});

import { GoogleProvider } from '../src/providers/google.js';
import { OllamaProvider } from '../src/providers/ollama.js';

describe('GoogleProvider stub', () => {
  it('throws not yet implemented', async () => {
    const p = new GoogleProvider();
    await expect(p.complete([{ role: 'user', content: 'hi' }], 'gemini-pro'))
      .rejects.toThrow('not yet implemented');
  });
});

describe('OllamaProvider stub', () => {
  it('throws not yet implemented', async () => {
    const p = new OllamaProvider();
    await expect(p.complete([{ role: 'user', content: 'hi' }], 'llama3'))
      .rejects.toThrow('not yet implemented');
  });
});
