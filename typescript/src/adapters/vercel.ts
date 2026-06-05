import { BudgetExceeded } from '../errors.js';
import type { Session } from '../session.js';
import { computeCost } from '../providers/base.js';

export interface WrapModelOptions {
  circuitBreaker?: any;
}

/**
 * Wraps any Vercel AI SDK LanguageModelV1 model.
 * Records token usage to the session after every generate/stream call.
 * Optionally checks the circuit breaker before each call.
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
