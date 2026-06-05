import type { LLMMessage, LLMProvider, LLMResponse } from './base.js';

export class OllamaProvider implements LLMProvider {
  async complete(_messages: LLMMessage[], _model = 'llama3'): Promise<LLMResponse> {
    throw new Error(
      'OllamaProvider is not yet implemented. ' +
      'Track progress at github.com/your-org/truss — planned for Phase 3.',
    );
  }
}
