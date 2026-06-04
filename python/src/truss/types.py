from __future__ import annotations

import math
from enum import IntEnum, Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


class ContextRole(str, Enum):
    TASK = "task"
    CONSTRAINT = "constraint"
    FINDING = "finding"
    DECISION = "decision"
    WARNING = "warning"
    BACKGROUND = "background"


class ContextWeight(IntEnum):
    BACKGROUND = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class ContextBlock(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: ContextRole
    weight: ContextWeight
    content: str
    source: str
    token_count: int = 0
    created_at: int = 0
    referenced_by: list[UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_token_count(self) -> "ContextBlock":
        if "token_count" not in self.model_fields_set:
            self.token_count = estimate_tokens(self.content)
        return self


class ModelTier(str, Enum):
    CHEAP = "cheap"
    STANDARD = "standard"
    PREMIUM = "premium"
    AUTO = "auto"


class EvidenceRef(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    source_url: Optional[str] = None
    tool_name: Optional[str] = None
    confidence: float = 1.0


class DecisionRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    decision: str
    reasoning: str
    evidence_ids: list[UUID] = Field(default_factory=list)
    confidence: float = 1.0
    decided_by: str
    timestamp: int = 0


class AgentEnvelope(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    task: str
    context: list[ContextBlock] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    budget_usd_remaining: Optional[float] = None  # None means unlimited
    checkpoint_id: Optional[UUID] = None
    model_hint: ModelTier = ModelTier.AUTO
    parent_agent: Optional[str] = None
    scope: list[str] = Field(default_factory=list)
    created_at: int = 0
