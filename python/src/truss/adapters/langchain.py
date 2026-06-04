from __future__ import annotations

from typing import Any

from truss.context.surgeon import SurgeonConfig, compress
from truss.types import ContextBlock, ContextRole, ContextWeight


class TrussMemory:
    """Drop-in replacement for LangChain ConversationBufferMemory.

    Compresses conversation history using Truss context surgeon.
    No dependency on langchain-core at import time.
    """

    def __init__(
        self,
        target_tokens: int = 8_000,
        preserve_recent: int = 5,
        strategy: str = "hybrid",
        memory_key: str = "history",
    ) -> None:
        self.memory_key = memory_key
        self.blocks: list[ContextBlock] = []
        self._config = SurgeonConfig(target_tokens=target_tokens, preserve_recent=preserve_recent)

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        human = inputs.get("input", "")
        ai = outputs.get("output", "")
        if human:
            self.blocks.append(ContextBlock(
                role=ContextRole.TASK, weight=ContextWeight.NORMAL,
                content=f"Human: {human}", source="user",
            ))
        if ai:
            self.blocks.append(ContextBlock(
                role=ContextRole.FINDING, weight=ContextWeight.NORMAL,
                content=f"AI: {ai}", source="assistant",
            ))

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        if not self.blocks:
            return {self.memory_key: ""}
        result = compress(self.blocks, self._config)
        self.blocks = result.blocks
        return {self.memory_key: "\n".join(b.content for b in self.blocks)}

    def clear(self) -> None:
        self.blocks = []
