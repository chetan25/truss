import pytest
from truss.providers.base import (
    LLMMessage, LLMUsage, StreamChunk, LLMStreamProvider,
    COST_TABLE, compute_cost,
)


def test_stream_chunk_non_final_has_no_usage():
    chunk = StreamChunk(text="hello", is_final=False)
    assert chunk.usage is None


def test_stream_chunk_final_carries_usage():
    usage = LLMUsage(input_tokens=10, output_tokens=5, cost_usd=0.001)
    chunk = StreamChunk(text="", is_final=True, usage=usage)
    assert chunk.usage.input_tokens == 10


def test_cost_table_has_gemini_models():
    assert "gemini-1.5-flash" in COST_TABLE
    assert "gemini-1.5-pro" in COST_TABLE


def test_ollama_cost_is_zero():
    cost = compute_cost("llama3", 1000, 1000)
    assert cost == 0.0


def test_llm_stream_provider_protocol_satisfied():
    from typing import Iterator

    class FakeStreamProvider:
        def complete(self, messages, model, **opts): ...
        def stream(self, messages, model, **opts) -> Iterator[StreamChunk]: ...

    assert isinstance(FakeStreamProvider(), LLMStreamProvider)


# ---------------------------------------------------------------------------
# Task 2: AnthropicProvider.stream()
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock


def _make_anthropic_stream_mock(chunks=("Hello", " world"), input_tokens=10, output_tokens=5):
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cm)
    mock_cm.__exit__ = MagicMock(return_value=False)
    mock_cm.text_stream = iter(chunks)
    final_msg = MagicMock()
    final_msg.usage.input_tokens = input_tokens
    final_msg.usage.output_tokens = output_tokens
    mock_cm.get_final_message.return_value = final_msg
    return mock_cm


def test_anthropic_stream_yields_text_chunks():
    from truss.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.messages.stream.return_value = _make_anthropic_stream_mock(("Hi", "!"))

    chunks = list(provider.stream(
        [LLMMessage(role="user", content="hello")],
        model="claude-haiku-4-5",
    ))
    non_final = [c for c in chunks if not c.is_final]
    assert [c.text for c in non_final] == ["Hi", "!"]


def test_anthropic_stream_final_chunk_has_usage():
    from truss.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.messages.stream.return_value = _make_anthropic_stream_mock(
        input_tokens=100, output_tokens=50
    )

    chunks = list(provider.stream(
        [LLMMessage(role="user", content="hi")],
        model="claude-haiku-4-5",
    ))
    final = next(c for c in chunks if c.is_final)
    assert final.usage.input_tokens == 100
    assert final.usage.output_tokens == 50
    assert final.usage.cost_usd > 0


def test_anthropic_stream_records_usage_to_session():
    from truss.providers.anthropic import AnthropicProvider
    from truss.session import Session

    session = Session()
    provider = AnthropicProvider(api_key="test-key", session=session)
    provider._client = MagicMock()
    provider._client.messages.stream.return_value = _make_anthropic_stream_mock(
        input_tokens=200, output_tokens=100
    )

    list(provider.stream([LLMMessage(role="user", content="hi")], model="claude-haiku-4-5"))
    assert session.report().budget_used_usd > 0


# ---------------------------------------------------------------------------
# Task 3: OpenAIProvider.stream()
# ---------------------------------------------------------------------------

def _make_openai_chunk(content=None, prompt_tokens=None, completion_tokens=None):
    chunk = MagicMock()
    if content is not None:
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
    else:
        chunk.choices = []
    if prompt_tokens is not None:
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = prompt_tokens
        chunk.usage.completion_tokens = completion_tokens
    else:
        chunk.usage = None
    return chunk


def test_openai_stream_yields_text_chunks():
    from truss.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = iter([
        _make_openai_chunk(content="Hello"),
        _make_openai_chunk(content=" world"),
        _make_openai_chunk(prompt_tokens=10, completion_tokens=5),
    ])

    chunks = list(provider.stream(
        [LLMMessage(role="user", content="hello")],
        model="gpt-4o-mini",
    ))
    non_final = [c for c in chunks if not c.is_final]
    assert [c.text for c in non_final] == ["Hello", " world"]


def test_openai_stream_final_chunk_has_usage():
    from truss.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = iter([
        _make_openai_chunk(content="hi"),
        _make_openai_chunk(prompt_tokens=50, completion_tokens=25),
    ])

    chunks = list(provider.stream(
        [LLMMessage(role="user", content="hi")],
        model="gpt-4o-mini",
    ))
    final = next(c for c in chunks if c.is_final)
    assert final.usage.input_tokens == 50
    assert final.usage.output_tokens == 25


# ---------------------------------------------------------------------------
# Task 4: async_complete()
# ---------------------------------------------------------------------------
from unittest.mock import AsyncMock


def _make_anthropic_response_mock(text="Hello!", input_tokens=10, output_tokens=5):
    mock_content = MagicMock()
    mock_content.text = text
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    return mock_response


def _make_openai_response_mock(text="Hi!", prompt_tokens=10, completion_tokens=5):
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    return mock_response


async def test_anthropic_async_complete_returns_response():
    from truss.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    provider._async_client = MagicMock()
    provider._async_client.messages.create = AsyncMock(
        return_value=_make_anthropic_response_mock("Async reply", 100, 50)
    )

    result = await provider.async_complete(
        [LLMMessage(role="user", content="hello")],
        model="claude-haiku-4-5",
    )
    assert result.text == "Async reply"
    assert result.usage.input_tokens == 100


async def test_openai_async_complete_returns_response():
    from truss.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")
    provider._async_client = MagicMock()
    provider._async_client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response_mock("Async OpenAI", 80, 40)
    )

    result = await provider.async_complete(
        [LLMMessage(role="user", content="hello")],
        model="gpt-4o-mini",
    )
    assert result.text == "Async OpenAI"
    assert result.usage.input_tokens == 80


async def test_anthropic_async_complete_records_to_session():
    from truss.providers.anthropic import AnthropicProvider
    from truss.session import Session

    session = Session()
    provider = AnthropicProvider(api_key="test-key", session=session)
    provider._async_client = MagicMock()
    provider._async_client.messages.create = AsyncMock(
        return_value=_make_anthropic_response_mock(input_tokens=200, output_tokens=100)
    )

    await provider.async_complete([LLMMessage(role="user", content="hi")], model="claude-haiku-4-5")
    assert session.report().budget_used_usd > 0


# ---------------------------------------------------------------------------
# Task 5: GoogleProvider
# ---------------------------------------------------------------------------

def _make_google_mock(text="Gemini reply", input_tokens=20, output_tokens=10):
    mock_response = MagicMock()
    mock_response.text = text
    mock_response.usage_metadata.prompt_token_count = input_tokens
    mock_response.usage_metadata.candidates_token_count = output_tokens
    mock_result = MagicMock()
    mock_result.response = mock_response
    mock_chat = MagicMock()
    mock_chat.send_message.return_value = mock_result
    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_chat
    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    return mock_genai


def test_google_provider_complete_returns_response():
    from truss.providers.google import GoogleProvider

    provider = GoogleProvider(api_key="test-key")
    provider._genai = _make_google_mock("Gemini reply", 20, 10)

    result = provider.complete(
        [LLMMessage(role="user", content="hello")],
        model="gemini-1.5-flash",
    )
    assert result.text == "Gemini reply"
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 10
    assert result.usage.cost_usd > 0


def test_google_provider_system_message_prepended():
    from truss.providers.google import GoogleProvider

    provider = GoogleProvider(api_key="test-key")
    mock_genai = _make_google_mock()
    provider._genai = mock_genai

    provider.complete(
        [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hi"),
        ],
        model="gemini-1.5-flash",
    )

    chat = mock_genai.GenerativeModel.return_value.start_chat.return_value
    call_args = chat.send_message.call_args[0][0]
    assert "You are helpful." in call_args
    assert "Hi" in call_args


def test_google_provider_records_usage_to_session():
    from truss.providers.google import GoogleProvider
    from truss.session import Session

    session = Session()
    provider = GoogleProvider(api_key="test-key", session=session)
    provider._genai = _make_google_mock(input_tokens=100, output_tokens=50)

    provider.complete([LLMMessage(role="user", content="hi")], model="gemini-1.5-flash")
    assert session.report().budget_used_usd > 0


def test_google_provider_missing_api_key_raises():
    import os
    from truss.providers.google import GoogleProvider

    original = os.environ.pop("GOOGLE_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GoogleProvider()
    finally:
        if original:
            os.environ["GOOGLE_API_KEY"] = original
