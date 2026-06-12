import { computeCost, LLMMessage, LLMProvider, LLMResponse, LLMUsage, ProviderOptions } from './base.js';

export class OllamaProvider implements LLMProvider {
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
      throw new Error(`Ollama request failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    const text: string = data.message?.content ?? '';
    const inputTokens: number = data.prompt_eval_count ?? 0;
    const outputTokens: number = data.eval_count ?? 0;
    const costUsd = computeCost(modelId, inputTokens, outputTokens);

    if (this.session) {
      this.session.recordUsage(inputTokens, outputTokens, costUsd, modelId);
    }

    return {
      text,
      model: modelId,
      usage: { inputTokens, outputTokens, costUsd } satisfies LLMUsage,
      raw: data,
    };
  }
}
