from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from truss.types import ModelTier


@dataclass
class ModelSpec:
    name: str
    tier: ModelTier
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float

    @property
    def cost_per_1k_total(self) -> float:
        return self.cost_per_1k_input + self.cost_per_1k_output


@dataclass
class RouterRule:
    keywords: list[str]
    preferred_tier: ModelTier


@dataclass
class RouterConfig:
    models: list[ModelSpec]
    rules: list[RouterRule] = field(default_factory=list)
    default_tier: ModelTier = ModelTier.AUTO


def route(task: str, config: RouterConfig) -> ModelSpec:
    task_lower = task.lower()
    tier = config.default_tier

    for rule in config.rules:
        if any(kw in task_lower for kw in rule.keywords):
            tier = rule.preferred_tier
            break

    if tier == ModelTier.AUTO:
        tier = ModelTier.STANDARD

    candidates = [m for m in config.models if m.tier == tier]
    pool = candidates if candidates else config.models
    return min(pool, key=lambda m: m.cost_per_1k_total)
