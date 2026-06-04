import { describe, it, expect } from 'vitest';
import { CircuitBreaker, CircuitBreakerConfig, CircuitTrip } from '../src/budget/circuit-breaker.js';

describe('CircuitBreaker', () => {
  it('trips on rate limit', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxRequestsPerMinute: 3, tripOnRepeatedPrompt: false }));
    expect(cb.checkAndRecord('a', 0.01, 0)).toBeNull();
    expect(cb.checkAndRecord('b', 0.01, 1000)).toBeNull();
    expect(cb.checkAndRecord('c', 0.01, 2000)).toBeNull();
    expect(cb.checkAndRecord('d', 0.01, 3000)).toBe(CircuitTrip.RateLimit);
  });

  it('trips on cost velocity', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxCostVelocityUsd: 0.50 }));
    cb.checkAndRecord('a', 0.40, 0);
    expect(cb.checkAndRecord('b', 0.20, 1000)).toBe(CircuitTrip.CostVelocity);
  });

  it('trips on repeated prompt', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ tripOnRepeatedPrompt: true }));
    cb.checkAndRecord('same prompt', 0.01, 0);
    expect(cb.checkAndRecord('same prompt', 0.01, 1000)).toBe(CircuitTrip.RepeatedPrompt);
  });

  it('different prompts do not trip', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ tripOnRepeatedPrompt: true }));
    cb.checkAndRecord('prompt A', 0.01, 0);
    expect(cb.checkAndRecord('prompt B', 0.01, 1000)).toBeNull();
  });

  it('trips on max retry depth', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxRetryDepth: 2 }));
    expect(cb.incrementRetry()).toBeNull();
    expect(cb.incrementRetry()).toBeNull();
    expect(cb.incrementRetry()).toBe(CircuitTrip.MaxRetryDepth);
  });

  it('resetRetry clears depth', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxRetryDepth: 1 }));
    cb.incrementRetry();
    cb.resetRetry();
    expect(cb.incrementRetry()).toBeNull();
  });

  it('evicts old requests from 60s window', () => {
    const cb = new CircuitBreaker(new CircuitBreakerConfig({ maxRequestsPerMinute: 2, tripOnRepeatedPrompt: false }));
    cb.checkAndRecord('a', 0.01, 0);
    cb.checkAndRecord('b', 0.01, 1000);
    expect(cb.checkAndRecord('c', 0.01, 61_000)).toBeNull();
  });
});
