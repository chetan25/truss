import { describe, it, expect, vi } from 'vitest';
import { COST_TABLE, computeCost } from '../src/providers/base.js';
import type { StreamChunk, LLMUsage } from '../src/providers/base.js';
import { AnthropicProvider } from '../src/providers/anthropic.js';
import { OpenAIProvider } from '../src/providers/openai.js';
import { GoogleProvider } from '../src/providers/google.js';
import { OllamaProvider } from '../src/providers/ollama.js';
import { Session } from '../src/session.js';

// ── Task 1: StreamChunk + COST_TABLE ─────────────────────────────────────────

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

// ── Task 2: AnthropicProvider.stream() ───────────────────────────────────────

function makeAnthropicStreamMock(chunks = ['Hello', ' world'], inputTokens = 10, outputTokens = 5) {
  const events = chunks.map(text => ({
    type: 'content_block_delta',
    delta: { type: 'text_delta', text },
  }));
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
    finalMessage: vi.fn().mockResolvedValue({
      usage: { input_tokens: inputTokens, output_tokens: outputTokens },
    }),
  };
}

describe('AnthropicProvider.stream()', () => {
  it('yields text chunks before final', async () => {
    const provider = new AnthropicProvider({ apiKey: 'test-key' });
    (provider as any)._client = {
      messages: { stream: vi.fn().mockReturnValue(makeAnthropicStreamMock(['Hi', '!'])) },
    };

    const chunks: StreamChunk[] = [];
    for await (const chunk of provider.stream([{ role: 'user', content: 'hello' }], 'claude-haiku-4-5')) {
      chunks.push(chunk);
    }

    expect(chunks.filter(c => !c.isFinal).map(c => c.text)).toEqual(['Hi', '!']);
  });

  it('emits final chunk with usage', async () => {
    const provider = new AnthropicProvider({ apiKey: 'test-key' });
    (provider as any)._client = {
      messages: { stream: vi.fn().mockReturnValue(makeAnthropicStreamMock(['x'], 100, 50)) },
    };

    const chunks: StreamChunk[] = [];
    for await (const chunk of provider.stream([{ role: 'user', content: 'hi' }], 'claude-haiku-4-5')) {
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

// ── Task 3: OpenAIProvider.stream() ──────────────────────────────────────────

function makeOpenAIStreamMock(chunks = ['Hello', ' world'], promptTokens = 10, completionTokens = 5) {
  const events = [
    ...chunks.map(text => ({ choices: [{ delta: { content: text } }], usage: null })),
    { choices: [], usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens } },
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
      chat: { completions: { create: vi.fn().mockResolvedValue(makeOpenAIStreamMock(['Hi', '!'])) } },
    };

    const chunks: StreamChunk[] = [];
    for await (const chunk of provider.stream([{ role: 'user', content: 'hello' }], 'gpt-4o-mini')) {
      chunks.push(chunk);
    }

    expect(chunks.filter(c => !c.isFinal).map(c => c.text)).toEqual(['Hi', '!']);
  });

  it('final chunk has usage', async () => {
    const provider = new OpenAIProvider({ apiKey: 'test-key' });
    (provider as any)._client = {
      chat: { completions: { create: vi.fn().mockResolvedValue(makeOpenAIStreamMock(['x'], 50, 25)) } },
    };

    const chunks: StreamChunk[] = [];
    for await (const chunk of provider.stream([{ role: 'user', content: 'hi' }], 'gpt-4o-mini')) {
      chunks.push(chunk);
    }

    const final = chunks.find(c => c.isFinal)!;
    expect(final.usage?.inputTokens).toBe(50);
    expect(final.usage?.outputTokens).toBe(25);
  });
});

// ── Task 4: GoogleProvider ────────────────────────────────────────────────────

function makeGoogleMock(text = 'Gemini reply', inputTokens = 20, outputTokens = 10) {
  const mockResponse = {
    text: () => text,
    usageMetadata: { promptTokenCount: inputTokens, candidatesTokenCount: outputTokens },
  };
  const mockResult = { response: mockResponse };
  const mockChat = { sendMessage: vi.fn().mockResolvedValue(mockResult) };
  const mockModel = { startChat: vi.fn().mockReturnValue(mockChat) };
  return { getGenerativeModel: vi.fn().mockReturnValue(mockModel) };
}

describe('GoogleProvider', () => {
  it('complete returns LLMResponse', async () => {
    const provider = new GoogleProvider({ apiKey: 'test-key' });
    (provider as any)._genai = makeGoogleMock('Gemini reply', 20, 10);

    const result = await provider.complete([{ role: 'user', content: 'hello' }], 'gemini-1.5-flash');
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
      [{ role: 'system', content: 'Be concise.' }, { role: 'user', content: 'Hi' }],
      'gemini-1.5-flash',
    );

    const modelMock = mock.getGenerativeModel.mock.results[0].value;
    const sendArg: string = modelMock.startChat.mock.results[0].value.sendMessage.mock.calls[0][0];
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

// ── Task 5: OllamaProvider ────────────────────────────────────────────────────

describe('OllamaProvider', () => {
  it('complete returns LLMResponse', async () => {
    const provider = new OllamaProvider({ baseUrl: 'http://localhost:11434' });
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => ({ message: { content: 'Ollama reply' }, prompt_eval_count: 15, eval_count: 8 }),
    } as any);

    const result = await provider.complete([{ role: 'user', content: 'hello' }], 'llama3');
    expect(result.text).toBe('Ollama reply');
    expect(result.usage.inputTokens).toBe(15);
    expect(result.usage.outputTokens).toBe(8);
    expect(result.usage.costUsd).toBe(0);
  });

  it('passes correct payload to fetch', async () => {
    const provider = new OllamaProvider({ baseUrl: 'http://localhost:11434' });
    const spy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => ({ message: { content: 'ok' }, prompt_eval_count: 0, eval_count: 0 }),
    } as any);

    await provider.complete([{ role: 'user', content: 'hello' }], 'mistral');

    const [url, options] = spy.mock.calls[0];
    expect(url).toContain('/api/chat');
    const body = JSON.parse(options!.body as string);
    expect(body.model).toBe('mistral');
    expect(body.stream).toBe(false);
    expect(body.messages[0].role).toBe('user');
  });

  it('provider-level cost is zero for local models', async () => {
    const session = new Session();
    const provider = new OllamaProvider({ session });
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => ({ message: { content: 'hi' }, prompt_eval_count: 100, eval_count: 50 }),
    } as any);

    const result = await provider.complete([{ role: 'user', content: 'hi' }], 'llama3');
    expect(result.usage.costUsd).toBe(0);
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
