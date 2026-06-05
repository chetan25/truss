import { describe, it, expect, vi } from 'vitest';
import { Session } from '../src/session.js';
import { wrapModel, createTrussMiddleware } from '../src/adapters/vercel.js';

function makeMockModel(text = 'response', promptTokens = 10, completionTokens = 5) {
  return {
    specificationVersion: 'v1' as const,
    provider: 'test',
    modelId: 'test-model',
    defaultObjectGenerationMode: undefined,
    doGenerate: vi.fn().mockResolvedValue({
      text,
      usage: { promptTokens, completionTokens },
      finishReason: 'stop',
      rawCall: { rawPrompt: '', rawSettings: {} },
    }),
    doStream: vi.fn().mockResolvedValue({
      stream: new ReadableStream({
        start(controller) {
          controller.enqueue({ type: 'text-delta', textDelta: text });
          controller.enqueue({ type: 'finish', finishReason: 'stop', usage: { promptTokens, completionTokens } });
          controller.close();
        },
      }),
      rawCall: { rawPrompt: '', rawSettings: {} },
    }),
  };
}

describe('wrapModel', () => {
  it('returns a model with same interface', () => {
    const session = new Session();
    const model = makeMockModel();
    const wrapped = wrapModel(model as any, session);
    expect(typeof wrapped.doGenerate).toBe('function');
    expect(typeof wrapped.doStream).toBe('function');
  });

  it('records usage after doGenerate', async () => {
    const session = new Session();
    const model = makeMockModel('hello', 100, 50);
    const wrapped = wrapModel(model as any, session);

    await wrapped.doGenerate({ inputFormat: 'messages', mode: { type: 'regular' }, prompt: [] });

    expect(session.report().budgetUsedUsd).toBeGreaterThan(0);
  });

  it('passes through the generate result unchanged', async () => {
    const session = new Session();
    const model = makeMockModel('exact text', 10, 5);
    const wrapped = wrapModel(model as any, session);

    const result = await wrapped.doGenerate({ inputFormat: 'messages', mode: { type: 'regular' }, prompt: [] });
    expect(result.text).toBe('exact text');
  });

  it('checks circuit breaker before generating', async () => {
    const { CircuitBreaker, CircuitBreakerConfig } = await import('../src/budget/circuit-breaker.js');
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxRequestsPerMinute: 0 }));
    const session = new Session();
    const model = makeMockModel();
    const wrapped = wrapModel(model as any, session, { circuitBreaker: cb });

    const { BudgetExceeded } = await import('../src/errors.js');
    await expect(
      wrapped.doGenerate({ inputFormat: 'messages', mode: { type: 'regular' }, prompt: [] }),
    ).rejects.toThrow(BudgetExceeded);
  });
});

describe('createTrussMiddleware', () => {
  it('wrapGenerate records usage', async () => {
    const session = new Session();
    const middleware = createTrussMiddleware(session);

    const mockDoGenerate = vi.fn().mockResolvedValue({
      text: 'test',
      usage: { promptTokens: 50, completionTokens: 25 },
      finishReason: 'stop',
      rawCall: { rawPrompt: '', rawSettings: {} },
    });

    await middleware.wrapGenerate!({
      doGenerate: mockDoGenerate,
      params: { inputFormat: 'messages', mode: { type: 'regular' }, prompt: [] } as any,
      model: { modelId: 'test-model' } as any,
    });

    expect(session.report().budgetUsedUsd).toBeGreaterThan(0);
  });
});
