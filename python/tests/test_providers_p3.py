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
