import pytest
from truss.types import AgentEnvelope, ContextBlock, ContextRole, ContextWeight, ModelTier
from truss.handoff.envelope import pack, unpack, BudgetCarve


def parent_envelope() -> AgentEnvelope:
    env = AgentEnvelope(task="parent task", budget_usd_remaining=1.0)
    env.context.append(ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL, content="critical info", source="planner"))
    env.context.append(ContextBlock(role=ContextRole.BACKGROUND, weight=ContextWeight.BACKGROUND, content="background fluff", source="loader"))
    return env


def test_pack_filters_context_by_weight():
    parent = parent_envelope()
    child = pack(parent, "child task", carry_weights=[ContextWeight.CRITICAL], budget_carve=BudgetCarve.fixed_usd(0.20))
    assert len(child.context) == 1
    assert child.context[0].weight == ContextWeight.CRITICAL


def test_pack_sets_parent_agent_id():
    parent = parent_envelope()
    child = pack(parent, "child", carry_weights=[ContextWeight.CRITICAL], budget_carve=BudgetCarve.fixed_usd(0.1))
    assert child.parent_agent == str(parent.id)


def test_pack_does_not_exceed_parent_budget():
    parent = parent_envelope()
    child = pack(parent, "child", carry_weights=[], budget_carve=BudgetCarve.fixed_usd(999.0))
    assert child.budget_usd_remaining is None or child.budget_usd_remaining <= 1.0


def test_pack_percent_carve():
    parent = parent_envelope()
    child = pack(parent, "child", carry_weights=[], budget_carve=BudgetCarve.percent(0.5))
    assert child.budget_usd_remaining is not None
    assert abs(child.budget_usd_remaining - 0.5) < 0.001


def test_unpack_returns_all_context_blocks():
    parent = parent_envelope()
    blocks = unpack(parent)
    assert len(blocks) == len(parent.context)


def test_pack_inherits_model_hint():
    parent = parent_envelope()
    parent.model_hint = ModelTier.PREMIUM
    child = pack(parent, "child", carry_weights=[], budget_carve=BudgetCarve.fixed_usd(0.1))
    assert child.model_hint == ModelTier.PREMIUM
