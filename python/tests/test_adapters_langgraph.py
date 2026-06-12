import pytest
from unittest.mock import patch

from truss.adapters.langgraph import TrussNode
from truss.session import Session


def test_truss_node_checkpoints_before_call():
    session = Session()

    @TrussNode(session)
    def node(state: dict) -> dict:
        return state

    node({"x": 1})
    assert session.report().checkpoint_count == 1


def test_truss_node_passes_state_through():
    session = Session()

    @TrussNode(session)
    def node(state: dict) -> dict:
        return {**state, "processed": True}

    result = node({"input": "hello"})
    assert result["processed"] is True
    assert result["input"] == "hello"


def test_truss_node_checkpoints_multiple_calls():
    session = Session()

    @TrussNode(session)
    def node(state: dict) -> dict:
        return state

    node({})
    node({})
    node({})
    assert session.report().checkpoint_count == 3


def test_truss_node_uses_function_name_in_checkpoint():
    session = Session()
    recorded: list[str] = []
    original = session.checkpoint

    def recording_checkpoint(desc: str = "") -> object:
        recorded.append(desc)
        return original(desc)

    session.checkpoint = recording_checkpoint  # type: ignore[method-assign]

    @TrussNode(session)
    def my_research_node(state: dict) -> dict:
        return state

    my_research_node({})
    assert recorded[0] == "before-my_research_node"


def test_truss_node_custom_name():
    session = Session()
    recorded: list[str] = []
    original = session.checkpoint

    def recording_checkpoint(desc: str = "") -> object:
        recorded.append(desc)
        return original(desc)

    session.checkpoint = recording_checkpoint  # type: ignore[method-assign]

    wrapped = TrussNode(session, name="research-phase")(lambda state: state)
    wrapped({})
    assert recorded[0] == "before-research-phase"


def test_truss_node_checkpoint_before_node_error():
    session = Session()

    @TrussNode(session)
    def failing_node(state: dict) -> dict:
        raise ValueError("node failed")

    with pytest.raises(ValueError, match="node failed"):
        failing_node({})

    assert session.report().checkpoint_count == 1


def test_truss_node_records_usage_from_state():
    session = Session()

    @TrussNode(session)
    def llm_node(state: dict) -> dict:
        return {
            "messages": ["reply"],
            "__truss_usage__": {"input_tokens": 100, "output_tokens": 50, "model": "gpt-4o-mini"},
        }

    llm_node({})
    assert session.report().budget_used_usd > 0


def test_truss_node_ignores_missing_usage_key():
    session = Session()

    @TrussNode(session)
    def node(state: dict) -> dict:
        return {"messages": ["reply"]}

    node({})
    assert session.report().checkpoint_count == 1


def test_truss_node_preserves_function_name():
    session = Session()

    def my_special_node(state: dict) -> dict:
        return state

    wrapped = TrussNode(session)(my_special_node)
    assert wrapped.__name__ == "my_special_node"


def test_truss_node_checkpoint_failure_does_not_block_execution():
    session = Session()

    @TrussNode(session)
    def node(state: dict) -> dict:
        return {**state, "done": True}

    with patch.object(session, "checkpoint", side_effect=RuntimeError("store unavailable")):
        result = node({"x": 1})

    assert result["done"] is True


async def test_truss_node_async():
    session = Session()

    @TrussNode(session)
    async def async_node(state: dict) -> dict:
        return {**state, "processed": True}

    result = await async_node({"input": "test"})
    assert result["processed"] is True
    assert session.report().checkpoint_count == 1


async def test_truss_node_async_records_usage():
    session = Session()

    @TrussNode(session)
    async def async_llm_node(state: dict) -> dict:
        return {"__truss_usage__": {"input_tokens": 200, "output_tokens": 80, "model": "gpt-4o"}}

    await async_llm_node({})
    assert session.report().budget_used_usd > 0


async def test_truss_node_async_checkpoint_before_error():
    session = Session()

    @TrussNode(session)
    async def failing_async_node(state: dict) -> dict:
        raise RuntimeError("async node failed")

    with pytest.raises(RuntimeError, match="async node failed"):
        await failing_async_node({})

    assert session.report().checkpoint_count == 1
