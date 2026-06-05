import pytest
from truss.providers.base import (
    LLMMessage, LLMUsage, LLMResponse, LLMProvider,
    COST_TABLE, compute_cost,
)


def test_cost_table_has_anthropic_models():
    assert "claude-haiku-4-5" in COST_TABLE
    assert "claude-sonnet-4-6" in COST_TABLE
    assert "claude-opus-4-8" in COST_TABLE


def test_cost_table_has_openai_models():
    assert "gpt-4o" in COST_TABLE
    assert "gpt-4o-mini" in COST_TABLE


def test_compute_cost_for_known_model():
    # claude-haiku-4-5: $0.001/1k input, $0.005/1k output
    cost = compute_cost("claude-haiku-4-5", input_tokens=1000, output_tokens=1000)
    assert abs(cost - 0.006) < 0.0001


def test_compute_cost_for_unknown_model_uses_default():
    cost = compute_cost("unknown-model-xyz", input_tokens=1000, output_tokens=0)
    assert cost == 0.001  # default input rate $0.001/1k


def test_llm_message_dataclass():
    msg = LLMMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_llm_usage_fields():
    usage = LLMUsage(input_tokens=100, output_tokens=50, cost_usd=0.01)
    assert usage.input_tokens + usage.output_tokens == 150


def test_llm_response_fields():
    usage = LLMUsage(input_tokens=10, output_tokens=5, cost_usd=0.001)
    resp = LLMResponse(text="hello", model="test-model", usage=usage)
    assert resp.text == "hello"
    assert resp.raw is None


def test_llm_provider_protocol_satisfied_by_duck_type():
    class FakeProvider:
        def complete(self, messages, model, **opts): ...

    assert isinstance(FakeProvider(), LLMProvider)
