export enum CircuitTrip {
  RateLimit = 'rate_limit',
  CostVelocity = 'cost_velocity',
  MaxRetryDepth = 'max_retry_depth',
  RepeatedPrompt = 'repeated_prompt',
}

export class CircuitBreakerConfig {
  maxRequestsPerMinute: number;
  maxCostVelocityUsd: number;
  maxRetryDepth: number;
  tripOnRepeatedPrompt: boolean;

  constructor(partial: Partial<CircuitBreakerConfig> = {}) {
    this.maxRequestsPerMinute = partial.maxRequestsPerMinute ?? 60;
    this.maxCostVelocityUsd = partial.maxCostVelocityUsd ?? 1.0;
    this.maxRetryDepth = partial.maxRetryDepth ?? 3;
    this.tripOnRepeatedPrompt = partial.tripOnRepeatedPrompt ?? true;
  }
}

interface Record { timestampMs: number; costUsd: number; promptHash: bigint }

function fnv1a(s: string): bigint {
  let h = 14695981039346656037n;
  for (const b of new TextEncoder().encode(s)) {
    h ^= BigInt(b);
    h = BigInt.asUintN(64, h * 1099511628211n);
  }
  return h;
}

export class CircuitBreaker {
  private window: Record[] = [];
  private retryDepth = 0;

  constructor(private config: CircuitBreakerConfig) {}

  checkAndRecord(prompt: string, costUsd: number, nowMs: number): CircuitTrip | null {
    const hash = fnv1a(prompt);
    const cutoff = nowMs - 60_000;
    this.window = this.window.filter(r => r.timestampMs > cutoff);

    if (this.window.length >= this.config.maxRequestsPerMinute) return CircuitTrip.RateLimit;

    const runningCost = this.window.reduce((s, r) => s + r.costUsd, 0);
    if (runningCost + costUsd > this.config.maxCostVelocityUsd) return CircuitTrip.CostVelocity;

    if (this.config.tripOnRepeatedPrompt) {
      const recent = this.window.slice(-3);
      if (recent.some(r => r.promptHash === hash)) return CircuitTrip.RepeatedPrompt;
    }

    this.window.push({ timestampMs: nowMs, costUsd, promptHash: hash });
    return null;
  }

  incrementRetry(): CircuitTrip | null {
    this.retryDepth += 1;
    return this.retryDepth > this.config.maxRetryDepth ? CircuitTrip.MaxRetryDepth : null;
  }

  resetRetry(): void { this.retryDepth = 0; }
}
