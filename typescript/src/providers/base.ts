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

export interface StreamChunk {
  text: string;
  isFinal: boolean;
  usage?: LLMUsage;
}

export interface LLMProvider {
  complete(messages: LLMMessage[], model: string, opts?: Record<string, unknown>): Promise<LLMResponse>;
}

export interface LLMStreamProvider extends LLMProvider {
  stream(messages: LLMMessage[], model: string, opts?: Record<string, unknown>): AsyncIterable<StreamChunk>;
}

export interface ProviderOptions {
  apiKey?: string;
  session?: any;
  circuitBreaker?: any;
  defaultModel?: string;
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
  // Gemini
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
  const [inRate, outRate] = COST_TABLE[model] ?? DEFAULT_RATES;
  return (inputTokens / 1000) * inRate + (outputTokens / 1000) * outRate;
}
