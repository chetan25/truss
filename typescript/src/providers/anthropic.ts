import { createRequire } from 'node:module';
import { BudgetExceeded } from '../errors.js';
import { LLMMessage, LLMProvider, LLMResponse, ProviderOptions, computeCost } from './base.js';

const _req = createRequire(import.meta.url);

export class AnthropicProvider implements LLMProvider {
  _client: any;
  private session?: any;
  private circuitBreaker?: any;
  private defaultModel: string;

  constructor(opts: ProviderOptions = {}) {
    const apiKey = opts.apiKey ?? process.env['ANTHROPIC_API_KEY'];
    if (!apiKey) throw new Error('ANTHROPIC_API_KEY not set and apiKey not provided');

    let AnthropicCtor: any;
    try {
      const mod = _req('@anthropic-ai/sdk');
      AnthropicCtor = mod.default ?? mod.Anthropic ?? mod;
    } catch {
      throw new Error('anthropic package required: npm install @anthropic-ai/sdk');
    }

    this._client = new AnthropicCtor({ apiKey });
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
