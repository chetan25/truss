import pytest
from truss.budget.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitTrip


def test_trips_on_rate_limit():
    cb = CircuitBreaker(CircuitBreakerConfig(max_requests_per_minute=3, trip_on_repeated_prompt=False))
    for i in range(3):
        assert cb.check_and_record("prompt", 0.01, i * 1000) is None
    trip = cb.check_and_record("prompt", 0.01, 3000)
    assert trip == CircuitTrip.RATE_LIMIT


def test_trips_on_cost_velocity():
    cb = CircuitBreaker(CircuitBreakerConfig(max_cost_velocity_usd=0.50))
    cb.check_and_record("a", 0.40, 0)
    trip = cb.check_and_record("b", 0.20, 1000)
    assert trip == CircuitTrip.COST_VELOCITY


def test_trips_on_repeated_prompt():
    cb = CircuitBreaker(CircuitBreakerConfig(trip_on_repeated_prompt=True))
    cb.check_and_record("same prompt", 0.01, 0)
    trip = cb.check_and_record("same prompt", 0.01, 1000)
    assert trip == CircuitTrip.REPEATED_PROMPT


def test_different_prompts_do_not_trip():
    cb = CircuitBreaker(CircuitBreakerConfig(trip_on_repeated_prompt=True))
    cb.check_and_record("prompt A", 0.01, 0)
    result = cb.check_and_record("prompt B", 0.01, 1000)
    assert result is None


def test_retry_depth_trips_after_max():
    cb = CircuitBreaker(CircuitBreakerConfig(max_retry_depth=2))
    assert cb.increment_retry() is None
    assert cb.increment_retry() is None
    assert cb.increment_retry() == CircuitTrip.MAX_RETRY_DEPTH


def test_reset_retry_clears_depth():
    cb = CircuitBreaker(CircuitBreakerConfig(max_retry_depth=1))
    cb.increment_retry()
    cb.reset_retry()
    assert cb.increment_retry() is None


def test_old_requests_evicted_from_window():
    cb = CircuitBreaker(CircuitBreakerConfig(max_requests_per_minute=2))
    cb.check_and_record("a", 0.01, 0)
    cb.check_and_record("b", 0.01, 1000)
    result = cb.check_and_record("c", 0.01, 61_000)
    assert result is None
