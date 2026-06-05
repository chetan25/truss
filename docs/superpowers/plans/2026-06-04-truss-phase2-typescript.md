# Truss Phase 2 TypeScript — Provider + Framework Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provider adapters (Anthropic full, OpenAI full, Google/Ollama stubs) and a Vercel AI SDK adapter (`wrapModel` + `createTrussMiddleware`) to the existing `truss-ai` TypeScript package.

**Architecture:** Option C (opt-in full stack) — providers work standalone or with a `Session` for auto-recording. The Vercel AI SDK adapter wraps any SDK model as a transparent proxy, recording usage and checking the circuit breaker without requiring a Truss provider. SDK imports are lazy (dynamic import in constructor) so missing packages give a helpful error at construction time.

**Tech Stack:** Node.js 18+ · TypeScript 5 · vitest · @anthropic-ai/sdk · openai · ai (Vercel AI SDK)

---

## File Structure

```
typescript/src/
├── providers/
│   ├── index.ts      # re-exports
│   ├── base.ts       # LLMMessage, LLMUsage, LLMResponse, LLMProvider, COST_TABLE, computeCost
│   ├── anthropic.ts  # AnthropicProvider (full)
│   ├── openai.ts     # OpenAIProvider (full)
│   ├── google.ts     # GoogleProvider (stub)
│   └── ollama.ts     # OllamaProvider (stub)
├── adapters/
│   └── vercel.ts     # wrapModel, createTrussMiddleware
└── index.ts          # add new exports

typescript/tests/
├── providers.test.ts  # all provider tests
└── vercel.test.ts     # Vercel AI SDK adapter tests
```

---

## Task 1: TypeScript Provider Base Types

**Files:**
- Create: `typescript/src/providers/base.ts`
- Create: `typescript/src/providers/index.ts`
- Create: `typescript/tests/providers.test.ts`

- [ ] **Step 1: Write failing tests**

Create `typescript/tests/providers.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd typescript && npm test tests/providers.test.ts
```

Expected: `Cannot find module '../src/providers/base.js'`

- [ ] **Step 3: Implement base.ts**

Create `typescript/src/providers/base.ts`:

```typescript
import type { Session } from '../session.js';
import type { CircuitBreaker } from '../budget/circuit-breaker.js';

export interface LLMMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface LLMUsage {
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
}

export interface LLMResponse {
  text: string;
  model: string;
  usage: LLMUsage;
  raw?: unknown;
}

export interface LLMProvider {
  complete(messages: LLMMessage[], model: string, opts?: Record<string, unknown>): Promise<LLMResponse>;
}

export interface ProviderOptions {
  apiKey?: string;
  session?: Session;
  circuitBreaker?: CircuitBreaker;
  defaultModel?: string;
}

export const COST_TABLE: Record<string, [number, number]> = {
  // [input_$/1k, output_$/1k]
  'claude-haiku-4-5':  [0.001,   0.005],
  'claude-sonnet-4-6': [0.003,   0.015],
  'claude-opus-4-8':   [0.015,   0.075],
  'gpt-4o-mini':       [0.00015, 0.0006],
  'gpt-4o':            [0.005,   0.015],
  'gpt-4-turbo':       [0.010,   0.030],
  'o1':                [0.015,   0.060],
  'o1-mini':           [0.003,   0.012],
};

const DEFAULT_RATES: [number, number] = [0.001, 0.005];

export function computeCost(model: string, inputTokens: number, outputTokens: number): number {
  const [inRate, outRate] = COST_TABLE[model] ?? DEFAULT_RATES;
  return (inputTokens / 1000) * inRate + (outputTokens / 1000) * outRate;
}
```

Create `typescript/src/providers/index.ts`:

```typescript
export { LLMMessage, LLMUsage, LLMResponse, LLMProvider, ProviderOptions, COST_TABLE, computeCost } from './base.js';
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd typescript && npm test tests/providers.test.ts
```

Expected: `8 tests passed`

- [ ] **Step 5: Commit**

```bash
git add typescript/src/providers/base.ts typescript/src/providers/index.ts typescript/tests/providers.test.ts
git commit -m "feat: add TypeScript provider base types — LLMMessage, COST_TABLE, computeCost"
```

---

## Task 2: AnthropicProvider (TypeScript)

**Files:**
- Create: `typescript/src/providers/anthropic.ts`
- Modify: `typescript/src/providers/index.ts`
- Modify: `typescript/tests/providers.test.ts`

- [ ] **Step 1: Install @anthropic-ai/sdk**

```bash
cd typescript && npm install @anthropic-ai/sdk
```

- [ ] **Step 2: Append AnthropicProvider tests**

Append to `typescript/tests/providers.test.ts`:

```typescript
import { vi } from 'vitest';
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd typescript && npm test tests/providers.test.ts
```

Expected: `Cannot find module '../src/providers/anthropic.js'`

- [ ] **Step 4: Implement anthropic.ts**

Create `typescript/src/providers/anthropic.ts`:

```typescript
import { BudgetExceeded } from '../errors.js';
import { LLMMessage, LLMProvider, LLMResponse, LLMUsage, ProviderOptions, computeCost } from './base.js';

export class AnthropicProvider implements LLMProvider {
  private _client: any;
  private session?: ProviderOptions['session'];
  private circuitBreaker?: ProviderOptions['circuitBreaker'];
  private defaultModel: string;

  constructor(opts: ProviderOptions = {}) {
    const apiKey = opts.apiKey ?? process.env['ANTHROPIC_API_KEY'];
    if (!apiKey) throw new Error('ANTHROPIC_API_KEY not set and apiKey not provided');

    // Lazy import — throws helpful error if package missing
    let Anthropic: any;
    try {
      // Use require for sync init; ESM consumers get the same package via cjs interop
      const mod = require('@anthropic-ai/sdk');
      Anthropic = mod.default ?? mod.Anthropic ?? mod;
    } catch {
      throw new Error('anthropic package required: npm install @anthropic-ai/sdk');
    }

    this._client = new Anthropic({ apiKey });
    this.session = opts.session;
    this.circuitBreaker = opts.circuitBreaker;
    this.defaultModel = opts.defaultModel ?? 'claude-haiku-4-5';
  }

  async complete(
    messages: LLMMessage[],
    model?: string,
    opts: Record<string, unknown> = {},
  ): Promise<LLMResponse> {
    const modelId = model ?? this.defaultModel;
    const nowMs = Date.now();

    if (this.circuitBreaker) {
      const trip = this.circuitBreaker.checkAndRecord('', 0, nowMs);
      if (trip !== null) throw new BudgetExceeded(`Circuit breaker tripped: ${trip}`);
    }

    const response = await this._client.messages.create({
      model: modelId,
      max_tokens: (opts['maxTokens'] as number) ?? 1024,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
    });

    const inputTokens: number = response.usage.input_tokens;
    const outputTokens: number = response.usage.output_tokens;
    const costUsd = computeCost(modelId, inputTokens, outputTokens);

    if (this.session) {
      this.session.recordUsage(inputTokens, outputTokens, costUsd, modelId);
    }

    const text: string = response.content?.[0]?.text ?? '';
    return { text, model: modelId, usage: { inputTokens, outputTokens, costUsd }, raw: response };
  }
}
```

Update `typescript/src/providers/index.ts`:

```typescript
export { LLMMessage, LLMUsage, LLMResponse, LLMProvider, ProviderOptions, COST_TABLE, computeCost } from './base.js';
export { AnthropicProvider } from './anthropic.js';
```

- [ ] **Step 5: Run all provider tests**

```bash
cd typescript && npm test tests/providers.test.ts
```

Expected: `12 tests passed`

- [ ] **Step 6: Commit**

```bash
git add typescript/src/providers/anthropic.ts typescript/src/providers/index.ts typescript/tests/providers.test.ts
git commit -m "feat: add TypeScript AnthropicProvider"
```

---

## Task 3: OpenAIProvider (TypeScript)

**Files:**
- Create: `typescript/src/providers/openai.ts`
- Modify: `typescript/src/providers/index.ts`
- Modify: `typescript/tests/providers.test.ts`

- [ ] **Step 1: Install openai**

```bash
cd typescript && npm install openai
```

- [ ] **Step 2: Append OpenAIProvider tests**

Append to `typescript/tests/providers.test.ts`:

```typescript
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd typescript && npm test tests/providers.test.ts
```

Expected: `Cannot find module '../src/providers/openai.js'`

- [ ] **Step 4: Implement openai.ts**

Create `typescript/src/providers/openai.ts`:

```typescript
import { BudgetExceeded } from '../errors.js';
import { LLMMessage, LLMProvider, LLMResponse, LLMUsage, ProviderOptions, computeCost } from './base.js';

export class OpenAIProvider implements LLMProvider {
  private _client: any;
  private session?: ProviderOptions['session'];
  private circuitBreaker?: ProviderOptions['circuitBreaker'];
  private defaultModel: string;

  constructor(opts: ProviderOptions = {}) {
    const apiKey = opts.apiKey ?? process.env['OPENAI_API_KEY'];
    if (!apiKey) throw new Error('OPENAI_API_KEY not set and apiKey not provided');

    let OpenAI: any;
    try {
      const mod = require('openai');
      OpenAI = mod.default ?? mod.OpenAI ?? mod;
    } catch {
      throw new Error('openai package required: npm install openai');
    }

    this._client = new OpenAI({ apiKey });
    this.session = opts.session;
    this.circuitBreaker = opts.circuitBreaker;
    this.defaultModel = opts.defaultModel ?? 'gpt-4o-mini';
  }

  async complete(
    messages: LLMMessage[],
    model?: string,
    opts: Record<string, unknown> = {},
  ): Promise<LLMResponse> {
    const modelId = model ?? this.defaultModel;
    const nowMs = Date.now();

    if (this.circuitBreaker) {
      const trip = this.circuitBreaker.checkAndRecord('', 0, nowMs);
      if (trip !== null) throw new BudgetExceeded(`Circuit breaker tripped: ${trip}`);
    }

    const response = await this._client.chat.completions.create({
      model: modelId,
      max_tokens: (opts['maxTokens'] as number) ?? 1024,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
    });

    const inputTokens: number = response.usage.prompt_tokens;
    const outputTokens: number = response.usage.completion_tokens;
    const costUsd = computeCost(modelId, inputTokens, outputTokens);

    if (this.session) {
      this.session.recordUsage(inputTokens, outputTokens, costUsd, modelId);
    }

    const text: string = response.choices?.[0]?.message?.content ?? '';
    return { text, model: modelId, usage: { inputTokens, outputTokens, costUsd }, raw: response };
  }
}
```

Update `typescript/src/providers/index.ts`:

```typescript
export { LLMMessage, LLMUsage, LLMResponse, LLMProvider, ProviderOptions, COST_TABLE, computeCost } from './base.js';
export { AnthropicProvider } from './anthropic.js';
export { OpenAIProvider } from './openai.js';
```

- [ ] **Step 5: Run all provider tests**

```bash
cd typescript && npm test tests/providers.test.ts
```

Expected: `15 tests passed`

- [ ] **Step 6: Commit**

```bash
git add typescript/src/providers/openai.ts typescript/src/providers/index.ts typescript/tests/providers.test.ts
git commit -m "feat: add TypeScript OpenAIProvider"
```

---

## Task 4: Provider Stubs (Google + Ollama) + TypeScript

**Files:**
- Create: `typescript/src/providers/google.ts`
- Create: `typescript/src/providers/ollama.ts`
- Modify: `typescript/src/providers/index.ts`
- Modify: `typescript/tests/providers.test.ts`

- [ ] **Step 1: Append stub tests**

Append to `typescript/tests/providers.test.ts`:

```typescript
import { GoogleProvider } from '../src/providers/google.js';
import { OllamaProvider } from '../src/providers/ollama.js';

describe('GoogleProvider stub', () => {
  it('throws NotImplementedError', async () => {
    const p = new GoogleProvider();
    await expect(p.complete([{ role: 'user', content: 'hi' }], 'gemini-pro'))
      .rejects.toThrow('not yet implemented');
  });
});

describe('OllamaProvider stub', () => {
  it('throws NotImplementedError', async () => {
    const p = new OllamaProvider();
    await expect(p.complete([{ role: 'user', content: 'hi' }], 'llama3'))
      .rejects.toThrow('not yet implemented');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd typescript && npm test tests/providers.test.ts
```

Expected: `Cannot find module '../src/providers/google.js'`

- [ ] **Step 3: Implement stubs**

Create `typescript/src/providers/google.ts`:

```typescript
import type { LLMMessage, LLMProvider, LLMResponse } from './base.js';

export class GoogleProvider implements LLMProvider {
  async complete(_messages: LLMMessage[], _model = 'gemini-pro'): Promise<LLMResponse> {
    throw new Error(
      'GoogleProvider is not yet implemented. ' +
      'Track progress at github.com/your-org/truss — planned for Phase 3.',
    );
  }
}
```

Create `typescript/src/providers/ollama.ts`:

```typescript
import type { LLMMessage, LLMProvider, LLMResponse } from './base.js';

export class OllamaProvider implements LLMProvider {
  async complete(_messages: LLMMessage[], _model = 'llama3'): Promise<LLMResponse> {
    throw new Error(
      'OllamaProvider is not yet implemented. ' +
      'Track progress at github.com/your-org/truss — planned for Phase 3.',
    );
  }
}
```

Update `typescript/src/providers/index.ts`:

```typescript
export { LLMMessage, LLMUsage, LLMResponse, LLMProvider, ProviderOptions, COST_TABLE, computeCost } from './base.js';
export { AnthropicProvider } from './anthropic.js';
export { OpenAIProvider } from './openai.js';
export { GoogleProvider } from './google.ts';
export { OllamaProvider } from './ollama.ts';
```

Wait — use `.js` extension for TypeScript ESM:

Update `typescript/src/providers/index.ts`:

```typescript
export { LLMMessage, LLMUsage, LLMResponse, LLMProvider, ProviderOptions, COST_TABLE, computeCost } from './base.js';
export { AnthropicProvider } from './anthropic.js';
export { OpenAIProvider } from './openai.js';
export { GoogleProvider } from './google.js';
export { OllamaProvider } from './ollama.js';
```

- [ ] **Step 4: Run all provider tests**

```bash
cd typescript && npm test tests/providers.test.ts
```

Expected: `17 tests passed`

- [ ] **Step 5: Commit**

```bash
git add typescript/src/providers/ typescript/tests/providers.test.ts
git commit -m "feat: add TypeScript GoogleProvider and OllamaProvider stubs"
```

---

## Task 5: Vercel AI SDK Adapter

**Files:**
- Create: `typescript/src/adapters/vercel.ts`
- Create: `typescript/tests/vercel.test.ts`

- [ ] **Step 1: Install Vercel AI SDK**

```bash
cd typescript && npm install ai
```

- [ ] **Step 2: Write failing tests**

Create `typescript/tests/vercel.test.ts`:

```typescript
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd typescript && npm test tests/vercel.test.ts
```

Expected: `Cannot find module '../src/adapters/vercel.js'`

- [ ] **Step 4: Implement vercel.ts**

Create `typescript/src/adapters/vercel.ts`:

```typescript
import { BudgetExceeded } from '../errors.js';
import type { Session } from '../session.js';
import type { CircuitBreaker } from '../budget/circuit-breaker.js';
import { computeCost } from '../providers/base.js';

export interface WrapModelOptions {
  circuitBreaker?: CircuitBreaker;
}

/**
 * Wraps any Vercel AI SDK LanguageModelV1 model.
 * Records token usage to the session after every generate/stream call.
 * Optionally checks the circuit breaker before each call.
 *
 * Usage:
 *   const model = wrapModel(anthropic('claude-haiku-4-5'), session);
 *   const { text } = await generateText({ model, prompt: '...' });
 */
export function wrapModel(model: any, session: Session, opts: WrapModelOptions = {}): any {
  const { circuitBreaker } = opts;

  return {
    ...model,

    async doGenerate(params: any) {
      if (circuitBreaker) {
        const trip = circuitBreaker.checkAndRecord('', 0, Date.now());
        if (trip !== null) throw new BudgetExceeded(`Circuit breaker tripped: ${trip}`);
      }

      const result = await model.doGenerate(params);

      if (result.usage) {
        const { promptTokens = 0, completionTokens = 0 } = result.usage;
        const modelId: string = model.modelId ?? 'unknown';
        session.recordUsage(promptTokens, completionTokens, computeCost(modelId, promptTokens, completionTokens), modelId);
      }

      return result;
    },

    async doStream(params: any) {
      if (circuitBreaker) {
        const trip = circuitBreaker.checkAndRecord('', 0, Date.now());
        if (trip !== null) throw new BudgetExceeded(`Circuit breaker tripped: ${trip}`);
      }

      const { stream, ...rest } = await model.doStream(params);
      const modelId: string = model.modelId ?? 'unknown';

      const recordingStream = new ReadableStream({
        async start(controller) {
          const reader = (stream as ReadableStream).getReader();
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (value?.type === 'finish' && value.usage) {
              const { promptTokens = 0, completionTokens = 0 } = value.usage;
              session.recordUsage(promptTokens, completionTokens, computeCost(modelId, promptTokens, completionTokens), modelId);
            }
            controller.enqueue(value);
          }
          controller.close();
        },
      });

      return { stream: recordingStream, ...rest };
    },
  };
}

/**
 * Creates a Vercel AI SDK LanguageModelV1Middleware that records usage to a session.
 * Use with experimental_wrapLanguageModel from the 'ai' package.
 *
 * Usage:
 *   import { experimental_wrapLanguageModel as wrap } from 'ai';
 *   const model = wrap({ model: anthropic('claude-haiku-4-5'), middleware: createTrussMiddleware(session) });
 */
export function createTrussMiddleware(session: Session, opts: WrapModelOptions = {}) {
  const { circuitBreaker } = opts;

  return {
    async wrapGenerate({ doGenerate, params, model }: any) {
      if (circuitBreaker) {
        const trip = circuitBreaker.checkAndRecord('', 0, Date.now());
        if (trip !== null) throw new BudgetExceeded(`Circuit breaker tripped: ${trip}`);
      }

      const result = await doGenerate();

      if (result.usage) {
        const { promptTokens = 0, completionTokens = 0 } = result.usage;
        const modelId: string = model?.modelId ?? 'unknown';
        session.recordUsage(promptTokens, completionTokens, computeCost(modelId, promptTokens, completionTokens), modelId);
      }

      return result;
    },

    async wrapStream({ doStream, params, model }: any) {
      const { stream, ...rest } = await doStream();
      const modelId: string = model?.modelId ?? 'unknown';

      const recordingStream = new ReadableStream({
        async start(controller) {
          const reader = (stream as ReadableStream).getReader();
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (value?.type === 'finish' && value.usage) {
              const { promptTokens = 0, completionTokens = 0 } = value.usage;
              session.recordUsage(promptTokens, completionTokens, computeCost(modelId, promptTokens, completionTokens), modelId);
            }
            controller.enqueue(value);
          }
          controller.close();
        },
      });

      return { stream: recordingStream, ...rest };
    },
  };
}
```

- [ ] **Step 5: Run all Vercel adapter tests**

```bash
cd typescript && npm test tests/vercel.test.ts
```

Expected: `5 tests passed`

- [ ] **Step 6: Commit**

```bash
git add typescript/src/adapters/vercel.ts typescript/tests/vercel.test.ts
git commit -m "feat: add Vercel AI SDK wrapModel and createTrussMiddleware"
```

---

## Task 6: TypeScript Public API

**Files:**
- Modify: `typescript/src/index.ts`

- [ ] **Step 1: Add new exports to index.ts**

Append to `typescript/src/index.ts`:

```typescript
// Phase 2 — Providers
export { LLMMessage, LLMUsage, LLMResponse, LLMProvider, ProviderOptions, COST_TABLE, computeCost } from './providers/base.js';
export { AnthropicProvider } from './providers/anthropic.js';
export { OpenAIProvider } from './providers/openai.js';
export { GoogleProvider } from './providers/google.js';
export { OllamaProvider } from './providers/ollama.js';
// Phase 2 — Adapters
export { wrapModel, createTrussMiddleware, WrapModelOptions } from './adapters/vercel.js';
```

- [ ] **Step 2: Verify full import**

```bash
cd typescript && node --input-type=module <<< "import * as truss from './src/index.js'; console.log('OK')"
```

Expected: `OK` (or use tsx/vitest to verify)

- [ ] **Step 3: Run the full test suite**

```bash
cd typescript && npm test 2>&1 | tail -8
```

Expected: all tests pass (around 82 total).

- [ ] **Step 4: Commit**

```bash
git add typescript/src/index.ts
git commit -m "feat: export Phase 2 providers and Vercel adapter from public API"
```

---

## Self-Review Against Spec

| Spec requirement | Task |
|---|---|
| `LLMMessage`, `LLMUsage`, `LLMResponse`, `LLMProvider` interface | Task 1 |
| `COST_TABLE` with Anthropic + OpenAI models, `computeCost()` | Task 1 |
| `AnthropicProvider` — full, records to session, checks circuit breaker | Task 2 |
| `OpenAIProvider` — full, records to session, checks circuit breaker | Task 3 |
| `GoogleProvider`, `OllamaProvider` — stubs with error messages | Task 4 |
| `wrapModel(model, session, opts?)` — proxies doGenerate + doStream | Task 5 |
| `createTrussMiddleware(session)` — LanguageModelV1Middleware | Task 5 |
| Circuit breaker check before calls in providers and wrapModel | Tasks 2, 3, 5 |
| All symbols exported from `src/index.ts` | Task 6 |

**Gaps not in this plan (out of scope per spec):**
- Streaming in providers (Phase 3)
- Async provider construction (Phase 3)
- `@ai-sdk/*` provider wrappers (Phase 3 — we wrap at the model level instead)
