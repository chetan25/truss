from __future__ import annotations

import json
from typing import Any

from truss.types import ContextBlock, ContextRole, ContextWeight, AgentEnvelope
from truss.handoff.envelope import pack, unpack, BudgetCarve


_WEIGHT_MAP: dict[str, ContextWeight] = {
    "critical":   ContextWeight.CRITICAL,
    "high":       ContextWeight.HIGH,
    "normal":     ContextWeight.NORMAL,
    "background": ContextWeight.BACKGROUND,
}


class PackHandoffTool:
    """CrewAI-compatible tool that packs the current session context into a JSON envelope.

    Input JSON: {"task": str, "carry_weights": list[str], "budget_fraction": float}
    Returns: AgentEnvelope as a JSON string.
    """

    name: str = "pack_handoff"
    description: str = (
        "Pack the current session context into a structured JSON envelope to pass to another agent. "
        "Input must be JSON with keys: 'task' (str), "
        "'carry_weights' (list of 'critical'/'high'/'normal'/'background'), "
        "'budget_fraction' (float 0.0-1.0)."
    )

    def __init__(self, session: Any) -> None:
        self._session = session

    def _run(self, tool_input: str) -> str:
        data = json.loads(tool_input)
        carry_weights = [_WEIGHT_MAP[w] for w in data.get("carry_weights", ["critical", "high"])]
        fraction = float(data.get("budget_fraction", 0.3))

        child = pack(
            self._session.envelope,
            task=data["task"],
            carry_weights=carry_weights,
            budget_carve=BudgetCarve.percent(fraction),
        )
        return child.model_dump_json()


class UnpackHandoffTool:
    """CrewAI-compatible tool that unpacks a received JSON envelope into context blocks.

    Input: the JSON envelope string produced by PackHandoffTool.
    Returns: JSON array of context blocks.
    """

    name: str = "unpack_handoff"
    description: str = (
        "Unpack a JSON envelope received from another agent into a list of context blocks. "
        "Input: the JSON envelope string."
    )

    def _run(self, envelope_json: str) -> str:
        envelope = AgentEnvelope.model_validate_json(envelope_json)
        blocks = unpack(envelope)
        return json.dumps([
            {"content": b.content, "role": b.role.value, "weight": int(b.weight), "source": b.source}
            for b in blocks
        ])


class TrussCrewCallback:
    """CrewAI step_callback that auto-checkpoints the Truss session after every agent step.

    Usage:
        crew = Crew(agents=[...], tasks=[...], step_callback=TrussCrewCallback(session))
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self._step_count = 0

    def __call__(self, step_output: Any) -> None:
        self._step_count += 1
        try:
            self._session.checkpoint(f"step-{self._step_count}")
        except Exception:
            pass
