import pytest
from truss.errors import BudgetExceeded, ToolOutOfScope, CheckpointNotFound, FenceLockConflict
from truss.types import (
    ContextBlock, ContextRole, ContextWeight, estimate_tokens,
    AgentEnvelope, ModelTier, EvidenceRef, DecisionRecord,
)


def test_estimate_tokens_ceiling_division():
    assert estimate_tokens("Hello") == 2        # ceil(5/4) = 2
    assert estimate_tokens("Hello world") == 3  # ceil(11/4) = 3
    assert estimate_tokens("") == 0


def test_context_block_auto_estimates_tokens():
    block = ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL, content="Hello world", source="test")
    assert block.token_count == 3


def test_context_block_explicit_token_count_not_overridden():
    block = ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.NORMAL, content="Hello world", source="test", token_count=99)
    assert block.token_count == 99


def test_context_weight_is_comparable():
    assert ContextWeight.CRITICAL > ContextWeight.NORMAL
    assert ContextWeight.BACKGROUND < ContextWeight.HIGH


def test_context_block_serialises_to_json():
    block = ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.NORMAL, content="data", source="agent-1")
    json_str = block.model_dump_json()
    assert "finding" in json_str


def test_agent_envelope_has_no_checkpoint_by_default():
    env = AgentEnvelope(task="test task")
    assert env.checkpoint_id is None


def test_agent_envelope_round_trips_json():
    env = AgentEnvelope(task="analyse pricing")
    json_str = env.model_dump_json()
    back = AgentEnvelope.model_validate_json(json_str)
    assert back.task == env.task
    assert back.id == env.id


def test_error_messages_include_key_info():
    assert "wallet" in str(BudgetExceeded("wallet exceeded $5"))
    assert "readFile" in str(ToolOutOfScope("readFile denied by manifest"))
    assert "abc-123" in str(CheckpointNotFound("abc-123"))
    assert "agent-a" in str(FenceLockConflict("doc-1 held by agent-a"))
