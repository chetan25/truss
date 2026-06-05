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


from unittest.mock import MagicMock


def _make_anthropic_response(text="Hello!", input_tokens=10, output_tokens=5):
    mock_content = MagicMock()
    mock_content.text = text
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    return mock_response


def test_anthropic_provider_complete_returns_response():
    from truss.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.messages.create.return_value = _make_anthropic_response("Test response", 100, 50)

    result = provider.complete(
        [LLMMessage(role="user", content="hello")],
        model="claude-haiku-4-5",
    )
    assert result.text == "Test response"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.usage.cost_usd > 0


def test_anthropic_provider_records_usage_to_session():
    from truss.providers.anthropic import AnthropicProvider
    from truss.session import Session

    session = Session()
    provider = AnthropicProvider(api_key="test-key", session=session)
    provider._client = MagicMock()
    provider._client.messages.create.return_value = _make_anthropic_response(input_tokens=200, output_tokens=100)

    provider.complete([LLMMessage(role="user", content="hi")], model="claude-haiku-4-5")

    report = session.report()
    assert report.budget_used_usd > 0


def test_anthropic_provider_missing_api_key_raises():
    import os
    from truss.providers.anthropic import AnthropicProvider

    original = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider()
    finally:
        if original:
            os.environ["ANTHROPIC_API_KEY"] = original


def test_anthropic_provider_circuit_breaker_trips():
    from truss.providers.anthropic import AnthropicProvider
    from truss.budget.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
    from truss.errors import BudgetExceeded

    cb = CircuitBreaker(CircuitBreakerConfig(max_requests_per_minute=0))
    provider = AnthropicProvider(api_key="test-key", circuit_breaker=cb)
    provider._client = MagicMock()

    with pytest.raises(BudgetExceeded):
        provider.complete([LLMMessage(role="user", content="hi")], model="claude-haiku-4-5")


def _make_openai_response(text="Hi!", prompt_tokens=10, completion_tokens=5):
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.model = "gpt-4o-mini"
    return mock_response


def test_openai_provider_complete_returns_response():
    from truss.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _make_openai_response("OpenAI reply", 50, 20)

    result = provider.complete(
        [LLMMessage(role="user", content="hello")],
        model="gpt-4o-mini",
    )
    assert result.text == "OpenAI reply"
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 20
    assert result.usage.cost_usd > 0


def test_openai_provider_records_usage_to_session():
    from truss.providers.openai import OpenAIProvider
    from truss.session import Session

    session = Session()
    provider = OpenAIProvider(api_key="test-key", session=session)
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _make_openai_response(prompt_tokens=100, completion_tokens=50)

    provider.complete([LLMMessage(role="user", content="hi")], model="gpt-4o-mini")

    report = session.report()
    assert report.budget_used_usd > 0


def test_openai_provider_missing_api_key_raises():
    import os
    from truss.providers.openai import OpenAIProvider

    original = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIProvider()
    finally:
        if original:
            os.environ["OPENAI_API_KEY"] = original


def test_google_provider_raises_not_implemented():
    from truss.providers.google import GoogleProvider

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        provider = GoogleProvider()
        provider.complete([LLMMessage(role="user", content="hi")], model="gemini-pro")


def test_ollama_provider_raises_not_implemented():
    from truss.providers.ollama import OllamaProvider

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        provider = OllamaProvider()
        provider.complete([LLMMessage(role="user", content="hi")], model="llama3")
