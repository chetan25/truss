# Truss Phase 3 TypeScript — Streaming, Full Google + Ollama Providers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add async streaming and full Google/Ollama provider implementations to the existing `truss-ai` TypeScript package.

**Architecture:** All provider methods are already `async` in TypeScript, so there is no `asyncComplete()` concept here — it already exists. Streaming adds `stream()` returning `AsyncIterable<StreamChunk>`, which is a native async generator. `GoogleProvider` wraps `@google/generative-ai` via `createRequire` (same ESM-compat pattern as other providers). `OllamaProvider` uses the built-in `fetch` (available in Node 18+) — no extra dependency. COST_TABLE gains Gemini and Ollama zero-cost entries.

**Tech Stack:** TypeScript 5 · ESM · Vitest · Node 18+ · @google/generative-ai (optional) · built-in fetch (Ollama)

---

## File Structure

```
typescript/src/
├── providers/
│   ├── base.ts        # add StreamChunk + Gemini/Ollama COST_TABLE entries
│   ├── anthropic.ts   # add stream() async generator
│   ├── openai.ts      # add stream() async generator
│   ├── google.ts      # REPLACE stub with full implementation
│   ├── ollama.ts      # REPLACE stub with full implementation
│   └── index.ts       # export StreamChunk

typescript/tests/
└── providers-p3.test.ts   # all Phase 3 tests
```

---

## Task 1: StreamChunk type + COST_TABLE additions

**Files:**
- Modify: `typescript/src/providers/base.ts`
- Create: `typescript/tests/providers-p3.test.ts`

- [ ] **Step 1: Write failing tests**

Create `typescript/tests/providers-p3.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { COST_TABLE, computeCost, StreamChunk, LLMUsage } from '../src/providers/base.js';

describe('StreamChunk', () => {
  it('non-final chunk has no usage', () => {
    const chunk: StreamChunk = { text: 'hello', isFinal: false };
    expect(chunk.usage).toBeUndefined();
  });

  it('final chunk carries usage', () => {
    const usage: LLMUsage = { inputTokens: 10, outputTokens: 5, costUsd: 0.001 };
    const chunk: StreamChunk = { text: '', isFinal: true, usage };
    expect(chunk.usage?.inputTokens).toBe(10);
  });
});

describe('COST_TABLE phase 3 additions', () => {
  it('has Gemini models', () => {
    expect(COST_TABLE['gemini-1.5-flash']).toBeDefined();
    expect(COST_TABLE['gemini-1.5-pro']).toBeDefined();
    expect(COST_TABLE['gemini-2.0-flash']).toBeDefined();
  });

  it('Ollama cost is zero', () => {
    expect(computeCost('llama3', 1000, 1000)).toBe(0);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd typescript && npx vitest run tests/providers-p3.test.ts
```

Expected: `Cannot find module` or `StreamChunk is not exported`

- [ ] **Step 3: Update base.ts**

In `typescript/src/providers/base.ts`, add `StreamChunk` interface and update `COST_TABLE`:

```typescript
export interface LLMMessage {
  role: string;   // 'user' | 'assistant' | 'system'
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

export interface StreamChunk {
  text: string;
  isFinal: boolean;
  usage?: LLMUsage;
}

export interface ProviderOptions {
  apiKey?: string;
  session?: unknown;
  circuitBreaker?: unknown;
  defaultModel?: string;
}

export interface LLMProvider {
  complete(messages: LLMMessage[], model: string, opts?: Record<string, unknown>): Promise<LLMResponse>;
}

export interface LLMStreamProvider extends LLMProvider {
  stream(messages: LLMMessage[], model: string, opts?: Record<string, unknown>): AsyncIterable<StreamChunk>;
}

export const COST_TABLE: Record<string, [number, number]> = {
  // [input_$/1k, output_$/1k]
  'claude-haiku-4-5':  [0.001,    0.005],
  'claude-sonnet-4-6': [0.003,    0.015],
  'claude-opus-4-8':   [0.015,    0.075],
  'gpt-4o-mini':       [0.00015,  0.0006],
  'gpt-4o':            [0.005,    0.015],
  'gpt-4-turbo':       [0.010,    0.030],
  'o1':                [0.015,    0.060],
  'o1-mini':           [0.003,    0.012],
  // Gemini models
  'gemini-1.5-flash':  [0.000075, 0.0003],
  'gemini-1.5-pro':    [0.00125,  0.005],
  'gemini-2.0-flash':  [0.0001,   0.0004],
  // Ollama (local — always free)
  'llama3':            [0.0,      0.0],
  'llama3.1':          [0.0,      0.0],
  'mistral':           [0.0,      0.0],
  'qwen2':             [0.0,      0.0],
};

const DEFAULT_RATES: [number, number] = [0.001, 0.005];

export function computeCost(model: string, inputTokens: number, outputTokens: number): number {
  const rates = COST_TABLE[model] ?? DEFAULT_RATES;
  return (inputTokens / 1000) * rates[0] + (outputTokens / 1000) * rates[1];
}
```

Also update `typescript/src/providers/index.ts` to export `StreamChunk`:

```typescript
export type { LLMMessage, LLMUsage, LLMResponse, StreamChunk, LLMProvider, LLMStreamProvider, ProviderOptions } from './base.js';
export { COST_TABLE, computeCost } from './base.js';
export { AnthropicProvider } from './anthropic.js';
export { OpenAIProvider } from './openai.js';
export { GoogleProvider } from './google.js';
export { OllamaProvider } from './ollama.js';
```

- [ ] **Step 4: Run tests**

```
cd typescript && npx vitest run tests/providers-p3.test.ts
```

Expected: `4 passed`

- [ ] **Step 5: Verify existing tests still pass**

```
cd typescript && npx vitest run
```

Expected: all previous tests pass.

- [ ] **Step 6: Commit**

```bash
git add typescript/src/providers/base.ts typescript/src/providers/index.ts typescript/tests/providers-p3.test.ts
git commit -m "feat: add StreamChunk type and Gemini/Ollama COST_TABLE entries (TypeScript)"
```

---

## Task 2: AnthropicProvider — stream()

**Files:**
- Modify: `typescript/src/providers/anthropic.ts`
- Modify: `typescript/tests/providers-p3.test.ts`

- [ ] **Step 1: Append streaming tests**

Append to `typescript/tests/providers-p3.test.ts`:

```typescript
import { describe, it, expect, vi } from 'vitest';
import { AnthropicProvider } from '../src/providers/anthropic.js';
import { Session } from '../src/session.js';
import type { LLMMessage } from '../src/providers/base.js';

function makeAnthropicStreamMock(chunks = ['Hello', ' world'], inputTokens = 10, outputTokens = 5) {
  const events = [
    ...chunks.map(text => ({
      type: 'content_block_delta',
      delta: { type: 'text_delta', text },
    })),
  ];
  let i = 0;
  const asyncIter = {
    [Symbol.asyncIterator]() {
      return {
        next: vi.fn().mockImplementation(async () => {
          if (i < events.length) return { done: false, value: events[i++] };
          return { done: true, value: undefined };
        }),
      };
    },
    finalMessage: vi.fn().mockResolvedValue({
      usage: { input_tokens: inputTokens, output_tokens: outputTokens },
    }),
  };
  return asyncIter;
}

describe('AnthropicProvider.stream()', () => {
  it('yields text chunks before final', async () => {
    const provider = new AnthropicProvider({ apiKey: 'test-key' });
    (provider as any)._client = {
      messages: { stream: vi.fn().mockReturnValue(makeAnthropicStreamMock(['Hi', '!'])) },
    };

    const chunks = [];
    for await (const chunk of provider.stream(
      [{ role: 'user', content: 'hello' }],
      'claude-haiku-4-5',
    )) {
      chunks.push(chunk);
    }

    const nonFinal = chunks.filter(c => !c.isFinal);
    expect(nonFinal.map(c => c.text)).toEqual(['Hi', '!']);
  });

  it('emits final chunk with usage', async () => {
    const provider = new AnthropicProvider({ apiKey: 'test-key' });
    (provider as any)._client = {
      messages: { stream: vi.fn().mockReturnValue(makeAnthropicStreamMock(['x'], 100, 50)) },
    };

    const chunks = [];
    for await (const chunk of provider.stream(
      [{ role: 'user', content: 'hi' }],
      'claude-haiku-4-5',
    )) {
      chunks.push(chunk);
    }

    const final = chunks.find(c => c.isFinal)!;
    expect(final.usage?.inputTokens).toBe(100);
    expect(final.usage?.outputTokens).toBe(50);
    expect(final.usage?.costUsd).toBeGreaterThan(0);
  });

  it('records usage to session after stream', async () => {
    const session = new Session();
    const provider = new AnthropicProvider({ apiKey: 'test-key', session });
    (provider as any)._client = {
      messages: { stream: vi.fn().mockReturnValue(makeAnthropicStreamMock(['hi'], 200, 100)) },
    };

    for await (const _ of provider.stream([{ role: 'user', content: 'hi' }], 'claude-haiku-4-5')) {
      // drain
    }

    expect(session.report().budgetUsedUsd).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd typescript && npx vitest run tests/providers-p3.test.ts -t "AnthropicProvider.stream"
```

Expected: `TypeError: provider.stream is not a function`

- [ ] **Step 3: Add stream() to anthropic.ts**

In `typescript/src/providers/anthropic.ts`, add the `stream()` method to the `AnthropicProvider` class. Also add `StreamChunk` to the import from `./base.js`:

```typescript
import { COST_TABLE, computeCost, LLMMessage, LLMResponse, LLMUsage, StreamChunk } from './base.js';
```

```typescript
  async *stream(
    messages: LLMMessage[],
    model: string,
    opts: Record<string, unknown> = {},
  ): AsyncGenerator<StreamChunk> {
    if (this.circuitBreaker) {
      const trip = this.circuitBreaker.checkAndRecord(
        messages[0]?.content ?? '',
        0,
        Date.now(),
      );
      if (trip !== null) {
        const { BudgetExceeded } = await import('../errors.js');
        throw new BudgetExceeded(`Circuit breaker tripped: ${String(trip)}`);
      }
    }

    const stream = this._client.messages.stream({
      model,
      max_tokens: (opts['maxTokens'] as number) ?? 1024,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
    });

    for await (const event of stream) {
      if (
        event.type === 'content_block_delta' &&
        event.delta?.type === 'text_delta' &&
        event.delta?.text
      ) {
        yield { text: event.delta.text, isFinal: false };
      }
    }

    const finalMsg = await stream.finalMessage();
    const inputTokens: number = finalMsg.usage?.input_tokens ?? 0;
    const outputTokens: number = finalMsg.usage?.output_tokens ?? 0;
    const costUsd = computeCost(model, inputTokens, outputTokens);

    if (this.session) {
      (this.session as any).recordUsage(inputTokens, outputTokens, costUsd, model);
    }

    yield { text: '', isFinal: true, usage: { inputTokens, outputTokens, costUsd } };
  }
```

- [ ] **Step 4: Run tests**

```
cd typescript && npx vitest run tests/providers-p3.test.ts -t "AnthropicProvider.stream"
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add typescript/src/providers/anthropic.ts typescript/tests/providers-p3.test.ts
git commit -m "feat: add AnthropicProvider.stream() async generator (TypeScript)"
```

---

## Task 3: OpenAIProvider — stream()

**Files:**
- Modify: `typescript/src/providers/openai.ts`
- Modify: `typescript/tests/providers-p3.test.ts`

- [ ] **Step 1: Append OpenAI streaming tests**

Append to `typescript/tests/providers-p3.test.ts`:

```typescript
import { OpenAIProvider } from '../src/providers/openai.js';

function makeOpenAIStreamMock(chunks = ['Hello', ' world'], promptTokens = 10, completionTokens = 5) {
  const events = [
    ...chunks.map(text => ({
      choices: [{ delta: { content: text } }],
      usage: null,
    })),
    {
      choices: [],
      usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens },
    },
  ];
  let i = 0;
  return {
    [Symbol.asyncIterator]() {
      return {
        next: vi.fn().mockImplementation(async () => {
          if (i < events.length) return { done: false, value: events[i++] };
          return { done: true, value: undefined };
        }),
      };
    },
  };
}

describe('OpenAIProvider.stream()', () => {
  it('yields text chunks', async () => {
    const provider = new OpenAIProvider({ apiKey: 'test-key' });
    (provider as any)._client = {
      chat: {
        completions: { create: vi.fn().mockResolvedValue(makeOpenAIStreamMock(['Hi', '!'])) },
      },
    };

    const chunks = [];
    for await (const chunk of provider.stream(
      [{ role: 'user', content: 'hello' }],
      'gpt-4o-mini',
    )) {
      chunks.push(chunk);
    }

    const nonFinal = chunks.filter(c => !c.isFinal);
    expect(nonFinal.map(c => c.text)).toEqual(['Hi', '!']);
  });

  it('final chunk has usage', async () => {
    const provider = new OpenAIProvider({ apiKey: 'test-key' });
    (provider as any)._client = {
      chat: {
        completions: { create: vi.fn().mockResolvedValue(makeOpenAIStreamMock(['x'], 50, 25)) },
      },
    };

    const chunks = [];
    for await (const chunk of provider.stream(
      [{ role: 'user', content: 'hi' }],
      'gpt-4o-mini',
    )) {
      chunks.push(chunk);
    }

    const final = chunks.find(c => c.isFinal)!;
    expect(final.usage?.inputTokens).toBe(50);
    expect(final.usage?.outputTokens).toBe(25);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd typescript && npx vitest run tests/providers-p3.test.ts -t "OpenAIProvider.stream"
```

Expected: `TypeError: provider.stream is not a function`

- [ ] **Step 3: Add stream() to openai.ts**

In `typescript/src/providers/openai.ts`, add the `stream()` method to `OpenAIProvider`. Also add `StreamChunk` to the import from `./base.js`:

```typescript
import { COST_TABLE, computeCost, LLMMessage, LLMResponse, LLMUsage, StreamChunk } from './base.js';
```

```typescript
  async *stream(
    messages: LLMMessage[],
    model: string,
    opts: Record<string, unknown> = {},
  ): AsyncGenerator<StreamChunk> {
    if (this.circuitBreaker) {
      const trip = this.circuitBreaker.checkAndRecord('', 0, Date.now());
      if (trip !== null) {
        const { BudgetExceeded } = await import('../errors.js');
        throw new BudgetExceeded(`Circuit breaker tripped: ${String(trip)}`);
      }
    }

    const streamIter = await this._client.chat.completions.create({
      model,
      max_tokens: (opts['maxTokens'] as number) ?? 1024,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
      stream: true,
      stream_options: { include_usage: true },
    });

    let inputTokens = 0;
    let outputTokens = 0;

    for await (const chunk of streamIter) {
      const content: string | null | undefined = chunk.choices?.[0]?.delta?.content;
      if (content) {
        yield { text: content, isFinal: false };
      }
      if (chunk.usage) {
        inputTokens = chunk.usage.prompt_tokens ?? 0;
        outputTokens = chunk.usage.completion_tokens ?? 0;
      }
    }

    const costUsd = computeCost(model, inputTokens, outputTokens);

    if (this.session && (inputTokens || outputTokens)) {
      (this.session as any).recordUsage(inputTokens, outputTokens, costUsd, model);
    }

    yield { text: '', isFinal: true, usage: { inputTokens, outputTokens, costUsd } };
  }
```

- [ ] **Step 4: Run tests**

```
cd typescript && npx vitest run tests/providers-p3.test.ts -t "OpenAIProvider.stream"
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add typescript/src/providers/openai.ts typescript/tests/providers-p3.test.ts
git commit -m "feat: add OpenAIProvider.stream() async generator (TypeScript)"
```

---

## Task 4: Full GoogleProvider

**Files:**
- Modify: `typescript/src/providers/google.ts` (replace stub)
- Modify: `typescript/tests/providers-p3.test.ts`
- Modify: `typescript/package.json`

- [ ] **Step 1: Add @google/generative-ai as optional peer dep**

In `typescript/package.json`, update `peerDependencies` (or add if missing):

```json
"peerDependencies": {
  "@anthropic-ai/sdk": ">=0.20",
  "openai": ">=4",
  "@google/generative-ai": ">=0.14"
},
"peerDependenciesMeta": {
  "@anthropic-ai/sdk": { "optional": true },
  "openai": { "optional": true },
  "@google/generative-ai": { "optional": true }
},
```

In `typescript/package.json`, add to `devDependencies`:

```json
"@google/generative-ai": "^0.14.0"
```

Install:

```
cd typescript && npm install
```

- [ ] **Step 2: Append Google tests**

Append to `typescript/tests/providers-p3.test.ts`:

```typescript
import { GoogleProvider } from '../src/providers/google.js';

function makeGoogleMock(text = 'Gemini reply', inputTokens = 20, outputTokens = 10) {
  const mockResponse = {
    text: () => text,
    usageMetadata: {
      promptTokenCount: inputTokens,
      candidatesTokenCount: outputTokens,
    },
  };
  const mockResult = { response: mockResponse };
  const mockChat = { sendMessage: vi.fn().mockResolvedValue(mockResult) };
  const mockModel = { startChat: vi.fn().mockReturnValue(mockChat) };
  return {
    getGenerativeModel: vi.fn().mockReturnValue(mockModel),
  };
}

describe('GoogleProvider', () => {
  it('complete returns LLMResponse', async () => {
    const provider = new GoogleProvider({ apiKey: 'test-key' });
    (provider as any)._genai = makeGoogleMock('Gemini reply', 20, 10);

    const result = await provider.complete(
      [{ role: 'user', content: 'hello' }],
      'gemini-1.5-flash',
    );
    expect(result.text).toBe('Gemini reply');
    expect(result.usage.inputTokens).toBe(20);
    expect(result.usage.outputTokens).toBe(10);
    expect(result.usage.costUsd).toBeGreaterThan(0);
  });

  it('system message is prepended to user content', async () => {
    const provider = new GoogleProvider({ apiKey: 'test-key' });
    const mock = makeGoogleMock();
    (provider as any)._genai = mock;

    await provider.complete(
      [
        { role: 'system', content: 'Be concise.' },
        { role: 'user', content: 'Hi' },
      ],
      'gemini-1.5-flash',
    );

    const chatMock = mock.getGenerativeModel.mock.results[0].value;
    const sendArg: string = chatMock.startChat.mock.results[0].value.sendMessage.mock.calls[0][0];
    expect(sendArg).toContain('Be concise.');
    expect(sendArg).toContain('Hi');
  });

  it('records usage to session', async () => {
    const session = new Session();
    const provider = new GoogleProvider({ apiKey: 'test-key', session });
    (provider as any)._genai = makeGoogleMock('hi', 100, 50);

    await provider.complete([{ role: 'user', content: 'hi' }], 'gemini-1.5-flash');
    expect(session.report().budgetUsedUsd).toBeGreaterThan(0);
  });

  it('throws if API key missing', () => {
    const orig = process.env.GOOGLE_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    try {
      expect(() => new GoogleProvider({})).toThrow('GOOGLE_API_KEY');
    } finally {
      if (orig) process.env.GOOGLE_API_KEY = orig;
    }
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

```
cd typescript && npx vitest run tests/providers-p3.test.ts -t "GoogleProvider"
```

Expected: `rejects.toThrow('not yet implemented')` — the current stub.

- [ ] **Step 4: Replace google.ts stub with full implementation**

Replace `typescript/src/providers/google.ts` entirely:

```typescript
import { createRequire } from 'node:module';
import { computeCost, LLMMessage, LLMResponse, LLMUsage, ProviderOptions } from './base.js';

const _req = createRequire(import.meta.url);

export class GoogleProvider {
  _genai: any;
  private session: any;
  private circuitBreaker: any;
  private defaultModel: string;

  constructor(opts: ProviderOptions & { baseUrl?: string } = {}) {
    const key = opts.apiKey ?? process.env['GOOGLE_API_KEY'];
    if (!key) throw new Error('GOOGLE_API_KEY not set and apiKey not provided');

    let mod: any;
    try {
      mod = _req('@google/generative-ai');
    } catch {
      throw new Error(
        'google-generativeai required: npm install @google/generative-ai',
      );
    }

    const Ctor = mod.GoogleGenerativeAI ?? mod.default?.GoogleGenerativeAI ?? mod;
    this._genai = new Ctor(key);
    this.session = opts.session;
    this.circuitBreaker = opts.circuitBreaker;
    this.defaultModel = opts.defaultModel ?? 'gemini-1.5-flash';
  }

  async complete(
    messages: LLMMessage[],
    model?: string,
    opts: Record<string, unknown> = {},
  ): Promise<LLMResponse> {
    const modelId = model ?? this.defaultModel;

    if (this.circuitBreaker) {
      const trip = this.circuitBreaker.checkAndRecord(
        messages[0]?.content ?? '',
        0,
        Date.now(),
      );
      if (trip !== null) {
        const { BudgetExceeded } = await import('../errors.js');
        throw new BudgetExceeded(`Circuit breaker tripped: ${String(trip)}`);
      }
    }

    // Separate system messages — Gemini prepends to first user turn
    let systemContent = '';
    const conversation: LLMMessage[] = [];
    for (const msg of messages) {
      if (msg.role === 'system') {
        systemContent = msg.content;
      } else {
        conversation.push(msg);
      }
    }

    if (conversation.length === 0) {
      throw new Error('At least one user or assistant message is required');
    }

    // Build history (all but last message)
    const history = conversation.slice(0, -1).map(msg => ({
      role: msg.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: msg.content }],
    }));

    const last = conversation[conversation.length - 1];
    const userText = systemContent
      ? `${systemContent}\n\n${last.content}`
      : last.content;

    const genModel = this._genai.getGenerativeModel({ model: modelId });
    const chat = genModel.startChat({ history });
    const result = await chat.sendMessage(userText);
    const response = result.response;

    const inputTokens: number = response.usageMetadata?.promptTokenCount ?? 0;
    const outputTokens: number = response.usageMetadata?.candidatesTokenCount ?? 0;
    const costUsd = computeCost(modelId, inputTokens, outputTokens);

    if (this.session) {
      (this.session as any).recordUsage(inputTokens, outputTokens, costUsd, modelId);
    }

    return {
      text: response.text(),
      model: modelId,
      usage: { inputTokens, outputTokens, costUsd } satisfies LLMUsage,
      raw: response,
    };
  }
}
```

- [ ] **Step 5: Run tests**

```
cd typescript && npx vitest run tests/providers-p3.test.ts -t "GoogleProvider"
```

Expected: `4 passed`

- [ ] **Step 6: Verify previously-passing stub tests are updated**

The file `typescript/tests/providers.test.ts` has a `GoogleProvider stub` suite that tests `rejects.toThrow('not yet implemented')`. Now that we have a full implementation, this test will fail. Update that suite:

Find in `typescript/tests/providers.test.ts`:

```typescript
describe('GoogleProvider stub', () => {
  it('throws not yet implemented', async () => {
    const p = new GoogleProvider();
    await expect(p.complete([{ role: 'user', content: 'hi' }], 'gemini-pro'))
      .rejects.toThrow('not yet implemented');
  });
});
```

Replace with:

```typescript
describe('GoogleProvider', () => {
  it('throws if API key missing', () => {
    const orig = process.env.GOOGLE_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    try {
      expect(() => new GoogleProvider({})).toThrow('GOOGLE_API_KEY');
    } finally {
      if (orig) process.env.GOOGLE_API_KEY = orig;
    }
  });
});
```

- [ ] **Step 7: Run full suite**

```
cd typescript && npx vitest run
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add typescript/src/providers/google.ts typescript/tests/providers-p3.test.ts typescript/tests/providers.test.ts typescript/package.json typescript/package-lock.json
git commit -m "feat: implement full GoogleProvider with Gemini API (TypeScript)"
```

---

## Task 5: Full OllamaProvider

**Files:**
- Modify: `typescript/src/providers/ollama.ts` (replace stub)
- Modify: `typescript/tests/providers-p3.test.ts`

- [ ] **Step 1: Append Ollama tests**

Append to `typescript/tests/providers-p3.test.ts`:

```typescript
import { OllamaProvider } from '../src/providers/ollama.js';

describe('OllamaProvider', () => {
  it('complete returns LLMResponse', async () => {
    const provider = new OllamaProvider({ baseUrl: 'http://localhost:11434' });
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        message: { content: 'Ollama reply' },
        prompt_eval_count: 15,
        eval_count: 8,
      }),
    } as any);

    const result = await provider.complete(
      [{ role: 'user', content: 'hello' }],
      'llama3',
    );
    expect(result.text).toBe('Ollama reply');
    expect(result.usage.inputTokens).toBe(15);
    expect(result.usage.outputTokens).toBe(8);
    expect(result.usage.costUsd).toBe(0);
  });

  it('passes correct payload to fetch', async () => {
    const provider = new OllamaProvider({ baseUrl: 'http://localhost:11434' });
    const spy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        message: { content: 'ok' },
        prompt_eval_count: 0,
        eval_count: 0,
      }),
    } as any);

    await provider.complete(
      [{ role: 'user', content: 'hello' }],
      'mistral',
    );

    const [url, options] = spy.mock.calls[0];
    expect(url).toContain('/api/chat');
    const body = JSON.parse(options!.body as string);
    expect(body.model).toBe('mistral');
    expect(body.stream).toBe(false);
    expect(body.messages[0].role).toBe('user');
  });

  it('records usage with zero cost to session', async () => {
    const session = new Session();
    const provider = new OllamaProvider({ session });
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        message: { content: 'hi' },
        prompt_eval_count: 100,
        eval_count: 50,
      }),
    } as any);

    await provider.complete([{ role: 'user', content: 'hi' }], 'llama3');
    expect(session.report().budgetUsedUsd).toBe(0);
  });

  it('throws on non-ok fetch response', async () => {
    const provider = new OllamaProvider();
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
    } as any);

    await expect(
      provider.complete([{ role: 'user', content: 'hi' }], 'llama3'),
    ).rejects.toThrow('Ollama request failed');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd typescript && npx vitest run tests/providers-p3.test.ts -t "OllamaProvider"
```

Expected: fails because stub throws `'not yet implemented'`.

- [ ] **Step 3: Replace ollama.ts stub with full implementation**

Replace `typescript/src/providers/ollama.ts` entirely:

```typescript
import { computeCost, LLMMessage, LLMResponse, LLMUsage, ProviderOptions } from './base.js';

export class OllamaProvider {
  private baseUrl: string;
  private session: any;
  private circuitBreaker: any;
  private defaultModel: string;

  constructor(opts: ProviderOptions & { baseUrl?: string } = {}) {
    this.baseUrl = (opts as any).baseUrl ?? 'http://localhost:11434';
    this.session = opts.session;
    this.circuitBreaker = opts.circuitBreaker;
    this.defaultModel = opts.defaultModel ?? 'llama3';
  }

  async complete(
    messages: LLMMessage[],
    model?: string,
    opts: Record<string, unknown> = {},
  ): Promise<LLMResponse> {
    const modelId = model ?? this.defaultModel;

    if (this.circuitBreaker) {
      const trip = this.circuitBreaker.checkAndRecord('', 0, Date.now());
      if (trip !== null) {
        const { BudgetExceeded } = await import('../errors.js');
        throw new BudgetExceeded(`Circuit breaker tripped: ${String(trip)}`);
      }
    }

    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelId,
        messages: messages.map(m => ({ role: m.role, content: m.content })),
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Ollama request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    const text: string = data.message?.content ?? '';
    const inputTokens: number = data.prompt_eval_count ?? 0;
    const outputTokens: number = data.eval_count ?? 0;
    const costUsd = computeCost(modelId, inputTokens, outputTokens); // 0 for local models

    if (this.session) {
      (this.session as any).recordUsage(inputTokens, outputTokens, costUsd, modelId);
    }

    return {
      text,
      model: modelId,
      usage: { inputTokens, outputTokens, costUsd } satisfies LLMUsage,
      raw: data,
    };
  }
}
```

- [ ] **Step 4: Update OllamaProvider stub test in providers.test.ts**

Find in `typescript/tests/providers.test.ts`:

```typescript
describe('OllamaProvider stub', () => {
  it('throws not yet implemented', async () => {
    const p = new OllamaProvider();
    await expect(p.complete([{ role: 'user', content: 'hi' }], 'llama3'))
      .rejects.toThrow('not yet implemented');
  });
});
```

Replace with:

```typescript
describe('OllamaProvider', () => {
  it('requires no API key', () => {
    expect(() => new OllamaProvider()).not.toThrow();
  });
});
```

- [ ] **Step 5: Run tests**

```
cd typescript && npx vitest run tests/providers-p3.test.ts -t "OllamaProvider"
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add typescript/src/providers/ollama.ts typescript/tests/providers-p3.test.ts typescript/tests/providers.test.ts
git commit -m "feat: implement full OllamaProvider using fetch (TypeScript)"
```

---

## Task 6: Update public API + run full suite

**Files:**
- Modify: `typescript/src/index.ts`

- [ ] **Step 1: Add StreamChunk and LLMStreamProvider to index.ts**

In `typescript/src/index.ts`, ensure the providers export line re-exports the new types:

```typescript
export type {
  LLMMessage, LLMUsage, LLMResponse, StreamChunk,
  LLMProvider, LLMStreamProvider, ProviderOptions,
} from './providers/base.js';
export { COST_TABLE, computeCost } from './providers/base.js';
export { AnthropicProvider } from './providers/anthropic.js';
export { OpenAIProvider } from './providers/openai.js';
export { GoogleProvider } from './providers/google.js';
export { OllamaProvider } from './providers/ollama.js';
```

- [ ] **Step 2: Verify new exports are accessible**

```
cd typescript && node --input-type=module --loader ts-node/esm <<'EOF'
import { StreamChunk, LLMStreamProvider } from './src/index.js';
console.log('OK');
EOF
```

Or simpler — just run the full test suite to verify no import errors.

- [ ] **Step 3: Run full suite**

```
cd typescript && npx vitest run
```

Expected: all tests pass (should be ~100+ total).

- [ ] **Step 4: Commit**

```bash
git add typescript/src/index.ts
git commit -m "feat: export StreamChunk and LLMStreamProvider from public API (TypeScript)"
```

---

## Self-Review Against Spec

| Requirement | Task |
|---|---|
| `StreamChunk` interface (`text`, `isFinal`, `usage?`) | Task 1 |
| `LLMStreamProvider` interface extending `LLMProvider` | Task 1 |
| `COST_TABLE` Gemini + Ollama entries | Task 1 |
| `AnthropicProvider.stream()` async generator | Task 2 |
| `OpenAIProvider.stream()` async generator | Task 3 |
| `GoogleProvider` full impl — system msg, history, usage | Task 4 |
| `OllamaProvider` full impl — native fetch, zero cost | Task 5 |
| `@google/generative-ai` added as optional peer dep | Task 4 |
| Old stub tests updated to reflect real implementations | Tasks 4 + 5 |
| All symbols exported from `index.ts` | Task 6 |

**Out of scope (Phase 4):**
- LangChain.js adapter
- Mastra/OpenAI Agents SDK adapter
- Multi-provider router integration
- `GoogleProvider.stream()` / `OllamaProvider.stream()`
- Observability / tracing hooks
