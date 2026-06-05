import type { LLMMessage, LLMProvider, LLMResponse } from './base.js';

export class GoogleProvider implements LLMProvider {
  async complete(_messages: LLMMessage[], _model = 'gemini-pro'): Promise<LLMResponse> {
    throw new Error(
      'GoogleProvider is not yet implemented. ' +
      'Track progress at github.com/your-org/truss — planned for Phase 3.',
    );
  }
}
