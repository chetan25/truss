import pytest
from unittest.mock import MagicMock


def make_llm_result(input_tokens=100, output_tokens=50, model="claude-haiku-4-5", format="anthropic"):
    from langchain_core.outputs import LLMResult

    if format == "anthropic":
        llm_output = {"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}, "model": model}
    else:
        llm_output = {"token_usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}, "model_name": model}

    return LLMResult(generations=[[]], llm_output=llm_output)


def test_callback_handler_records_usage_anthropic_format():
    from truss.adapters.langchain import TrussCallbackHandler
    from truss.session import Session

    session = Session()
    handler = TrussCallbackHandler(session)
    handler.on_llm_end(make_llm_result(100, 50, format="anthropic"))

    report = session.report()
    assert report.budget_used_usd > 0


def test_callback_handler_records_usage_openai_format():
    from truss.adapters.langchain import TrussCallbackHandler
    from truss.session import Session

    session = Session()
    handler = TrussCallbackHandler(session)
    handler.on_llm_end(make_llm_result(200, 100, format="openai"))

    report = session.report()
    assert report.budget_used_usd > 0


def test_callback_handler_silent_on_missing_usage():
    from truss.adapters.langchain import TrussCallbackHandler
    from truss.session import Session
    from langchain_core.outputs import LLMResult

    session = Session()
    handler = TrussCallbackHandler(session)
    handler.on_llm_end(LLMResult(generations=[[]], llm_output={}))

    report = session.report()
    assert report.budget_used_usd == 0.0


def test_callback_handler_on_llm_error_does_not_raise():
    from truss.adapters.langchain import TrussCallbackHandler
    from truss.session import Session

    session = Session()
    handler = TrussCallbackHandler(session)
    handler.on_llm_error(Exception("test error"))


def test_truss_llm_call_returns_text():
    from truss.adapters.langchain import TrussLLM
    from truss.providers.base import LLMMessage, LLMResponse, LLMUsage

    mock_provider = MagicMock()
    mock_provider.complete.return_value = LLMResponse(
        text="LLM response text",
        model="claude-haiku-4-5",
        usage=LLMUsage(input_tokens=10, output_tokens=5, cost_usd=0.0001),
    )

    llm = TrussLLM(provider=mock_provider, default_model="claude-haiku-4-5")
    result = llm._call("What is 2+2?")
    assert result == "LLM response text"


def test_truss_llm_passes_prompt_as_user_message():
    from truss.adapters.langchain import TrussLLM
    from truss.providers.base import LLMMessage, LLMResponse, LLMUsage

    mock_provider = MagicMock()
    mock_provider.complete.return_value = LLMResponse(
        text="answer",
        model="claude-haiku-4-5",
        usage=LLMUsage(input_tokens=5, output_tokens=3, cost_usd=0.00005),
    )

    llm = TrussLLM(provider=mock_provider, default_model="claude-haiku-4-5")
    llm._call("my prompt")

    called_messages = mock_provider.complete.call_args[1]["messages"]
    assert called_messages[0].role == "user"
    assert called_messages[0].content == "my prompt"


def test_truss_llm_type_property():
    from truss.adapters.langchain import TrussLLM

    llm = TrussLLM(provider=MagicMock(), default_model="claude-haiku-4-5")
    assert llm._llm_type == "truss"
