import pytest
import json
from unittest.mock import MagicMock


def test_pack_handoff_tool_returns_json_string():
    from truss.adapters.crewai import PackHandoffTool
    from truss.session import Session
    from truss.types import ContextBlock, ContextRole, ContextWeight, AgentEnvelope

    session = Session()
    session._envelope.context.append(
        ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL, content="important info", source="user")
    )
    session._envelope.budget_usd_remaining = 1.0

    tool = PackHandoffTool(session=session)
    result = tool._run(json.dumps({
        "task": "analyse pricing",
        "carry_weights": ["critical"],
        "budget_fraction": 0.3,
    }))

    envelope_data = json.loads(result)
    assert envelope_data["task"] == "analyse pricing"


def test_pack_handoff_carries_only_requested_weights():
    from truss.adapters.crewai import PackHandoffTool
    from truss.session import Session
    from truss.types import ContextBlock, ContextRole, ContextWeight

    session = Session()
    session._envelope.context = [
        ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL, content="critical", source="u"),
        ContextBlock(role=ContextRole.BACKGROUND, weight=ContextWeight.BACKGROUND, content="noise", source="u"),
    ]
    session._envelope.budget_usd_remaining = 1.0

    tool = PackHandoffTool(session=session)
    result = tool._run(json.dumps({"task": "sub-task", "carry_weights": ["critical"], "budget_fraction": 0.5}))
    data = json.loads(result)
    assert len(data["context"]) == 1
    assert data["context"][0]["weight"] == 3  # CRITICAL = 3


def test_unpack_handoff_tool_returns_context_blocks():
    from truss.adapters.crewai import PackHandoffTool, UnpackHandoffTool
    from truss.session import Session
    from truss.types import ContextBlock, ContextRole, ContextWeight

    session = Session()
    session._envelope.context = [
        ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.HIGH, content="finding A", source="agent"),
    ]
    session._envelope.budget_usd_remaining = 1.0

    pack_tool = PackHandoffTool(session=session)
    envelope_json = pack_tool._run(json.dumps({"task": "next", "carry_weights": ["high"], "budget_fraction": 0.5}))

    unpack_tool = UnpackHandoffTool()
    blocks_json = unpack_tool._run(envelope_json)
    blocks = json.loads(blocks_json)
    assert len(blocks) == 1
    assert blocks[0]["content"] == "finding A"


def test_truss_crew_callback_creates_checkpoint():
    from truss.adapters.crewai import TrussCrewCallback
    from truss.session import Session

    session = Session()
    callback = TrussCrewCallback(session=session)
    callback(MagicMock())
    callback(MagicMock())

    report = session.report()
    assert report.checkpoint_count == 2


def test_truss_crew_callback_does_not_raise_on_error():
    from truss.adapters.crewai import TrussCrewCallback
    from truss.session import Session
    from unittest.mock import patch

    session = Session()
    callback = TrussCrewCallback(session=session)

    with patch.object(session, "checkpoint", side_effect=RuntimeError("checkpoint failed")):
        callback(MagicMock())  # must not raise
