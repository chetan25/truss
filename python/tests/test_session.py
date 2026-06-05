import pytest
from truss.session import Session, SessionReport
from truss.types import AgentEnvelope, ContextBlock, ContextRole, ContextWeight


def sample_blocks(n: int = 10) -> list[ContextBlock]:
    weights = [ContextWeight.CRITICAL, ContextWeight.HIGH, ContextWeight.NORMAL, ContextWeight.BACKGROUND]
    return [
        ContextBlock(role=ContextRole.FINDING, weight=weights[i % 4], content=f"block {i} " + "content " * 20, source=f"s{i}")
        for i in range(n)
    ]


def test_session_report_after_compress():
    s = Session(target_tokens=100, preserve_recent=2)
    s.compress(sample_blocks(20))
    report = s.report()
    assert report.tokens_saved >= 0
    assert report.tokens_before > 0


def test_session_tracks_budget():
    s = Session(budget_usd=1.0)
    s.record_usage(500, 100, cost_usd=0.05, model="test")
    report = s.report()
    assert abs(report.budget_used_usd - 0.05) < 0.001
    assert report.budget_limit_usd == 1.0


def test_session_compress_keeps_critical_blocks():
    critical = ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL, content="must keep this", source="agent")
    bg = [ContextBlock(role=ContextRole.BACKGROUND, weight=ContextWeight.BACKGROUND, content="noise " * 50, source="x") for _ in range(20)]
    s = Session(target_tokens=50, preserve_recent=0)
    result = s.compress([critical] + bg)
    ids = {b.id for b in result.blocks}
    assert critical.id in ids


def test_session_checkpoint_and_rollback():
    s = Session()
    s._envelope.task = "original"
    cp_id = s.checkpoint("before change")
    s._envelope.task = "modified"
    s.rollback(cp_id)
    assert s._envelope.task == "original"


def test_session_checkpoint_count_in_report():
    s = Session()
    s.checkpoint("cp1")
    s.checkpoint("cp2")
    report = s.report()
    assert report.checkpoint_count == 2


async def test_session_async_context_manager():
    async with Session(budget_usd=0.50) as s:
        result = s.compress(sample_blocks(5))
        assert result.tokens_after <= result.tokens_before


def test_session_envelope_property_readable():
    s = Session()
    env = s.envelope
    assert env is s._envelope


def test_session_envelope_property_returns_agent_envelope():
    s = Session()
    assert isinstance(s.envelope, AgentEnvelope)
