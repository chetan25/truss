import { createRequire } from 'node:module';
import { BudgetExceeded } from '../errors.js';
import { LLMMessage, LLMProvider, LLMResponse, ProviderOptions, computeCost } from './base.js';

const _req = createRequire(import.meta.url);

export class OpenAIProvider implements LLMProvider {
  _client: any;
  private session?: any;
  private circuitBreaker?: any;
  private defaultModel: string;

  constructor(opts: ProviderOptions = {}) {
    const apiKey = opts.apiKey ?? process.env['OPENAI_API_KEY'];
    if (!apiKey) throw new Error('OPENAI_API_KEY not set and apiKey not provided');

    let OpenAICtor: any;
    try {
      const mod = _req('openai');
      OpenAICtor = mod.default ?? mod.OpenAI ?? mod;
    } catch {
      throw new Error('openai package required: npm install openai');
    }

    this._client = new OpenAICtor({ apiKey });
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
