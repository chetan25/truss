import { createRequire } from 'node:module';
import { computeCost, LLMMessage, LLMProvider, LLMResponse, LLMUsage, ProviderOptions } from './base.js';

const _req = createRequire(import.meta.url);

export class GoogleProvider implements LLMProvider {
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
      throw new Error('google-generativeai required: npm install @google/generative-ai');
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
      const trip = this.circuitBreaker.checkAndRecord(messages[0]?.content ?? '', 0, Date.now());
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

    const history = conversation.slice(0, -1).map(msg => ({
      role: msg.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: msg.content }],
    }));

    const last = conversation[conversation.length - 1];
    const userText = systemContent ? `${systemContent}\n\n${last.content}` : last.content;

    const genModel = this._genai.getGenerativeModel({ model: modelId });
    const chat = genModel.startChat({ history });
    const result = await chat.sendMessage(userText);
    const response = result.response;

    const inputTokens: number = response.usageMetadata?.promptTokenCount ?? 0;
    const outputTokens: number = response.usageMetadata?.candidatesTokenCount ?? 0;
    const costUsd = computeCost(modelId, inputTokens, outputTokens);

    if (this.session) {
      this.session.recordUsage(inputTokens, outputTokens, costUsd, modelId);
    }

    return {
      text: response.text(),
      model: modelId,
      usage: { inputTokens, outputTokens, costUsd } satisfies LLMUsage,
      raw: response,
    };
  }
}
