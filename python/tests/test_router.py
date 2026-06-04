import pytest
from truss.router.router import ModelSpec, RouterConfig, RouterRule, route
from truss.types import ModelTier


def sample_models():
    return [
        ModelSpec(name="claude-haiku-4-5", tier=ModelTier.CHEAP, max_tokens=8192, cost_per_1k_input=0.001, cost_per_1k_output=0.005),
        ModelSpec(name="claude-sonnet-4-6", tier=ModelTier.STANDARD, max_tokens=16384, cost_per_1k_input=0.003, cost_per_1k_output=0.015),
        ModelSpec(name="claude-opus-4-8", tier=ModelTier.PREMIUM, max_tokens=32768, cost_per_1k_input=0.015, cost_per_1k_output=0.075),
    ]


def test_route_uses_rule_matching_keyword():
    config = RouterConfig(
        models=sample_models(),
        rules=[RouterRule(keywords=["summarise", "summarize"], preferred_tier=ModelTier.CHEAP)],
        default_tier=ModelTier.STANDARD,
    )
    model = route("summarise this document", config)
    assert model.tier == ModelTier.CHEAP


def test_route_falls_back_to_default_tier():
    config = RouterConfig(models=sample_models(), rules=[], default_tier=ModelTier.STANDARD)
    model = route("analyse this deeply", config)
    assert model.tier == ModelTier.STANDARD


def test_route_auto_tier_defaults_to_standard():
    config = RouterConfig(models=sample_models(), rules=[], default_tier=ModelTier.AUTO)
    model = route("any task", config)
    assert model.tier == ModelTier.STANDARD


def test_route_returns_cheapest_in_tier():
    models = [
        ModelSpec(name="cheap-a", tier=ModelTier.CHEAP, max_tokens=4096, cost_per_1k_input=0.002, cost_per_1k_output=0.010),
        ModelSpec(name="cheap-b", tier=ModelTier.CHEAP, max_tokens=4096, cost_per_1k_input=0.001, cost_per_1k_output=0.005),
    ]
    config = RouterConfig(models=models, rules=[], default_tier=ModelTier.CHEAP)
    model = route("any task", config)
    assert model.name == "cheap-b"


def test_route_first_matching_rule_wins():
    config = RouterConfig(
        models=sample_models(),
        rules=[
            RouterRule(keywords=["fast"], preferred_tier=ModelTier.CHEAP),
            RouterRule(keywords=["fast", "deep"], preferred_tier=ModelTier.PREMIUM),
        ],
        default_tier=ModelTier.STANDARD,
    )
    model = route("fast analysis", config)
    assert model.tier == ModelTier.CHEAP
