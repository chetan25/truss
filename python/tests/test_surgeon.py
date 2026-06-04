import pytest
from truss.types import ContextBlock, ContextRole, ContextWeight
from truss.context.surgeon import (
    compress, SurgeonConfig, SurgeonResult, CompressionStrategy,
    score_relevance, detect_contradiction,
)


def make_block(weight: ContextWeight, tokens: int, content: str = "") -> ContextBlock:
    b = ContextBlock(role=ContextRole.FINDING, weight=weight, content=content or ("x " * tokens), source="test")
    object.__setattr__(b, "token_count", tokens)
    return b


def test_sliding_window_keeps_recent_n_blocks():
    blocks = [make_block(ContextWeight.NORMAL, 100, f"block {i}") for i in range(10)]
    config = SurgeonConfig(strategy=CompressionStrategy.SLIDING_WINDOW, target_tokens=300, preserve_recent=3, keep_recent=3)
    result = compress(blocks, config)
    assert result.tokens_after <= 300


def test_weighted_prune_drops_background_before_normal():
    bg = make_block(ContextWeight.BACKGROUND, 500, "background")
    normal = make_block(ContextWeight.NORMAL, 500, "normal")
    critical = make_block(ContextWeight.CRITICAL, 100, "critical")
    config = SurgeonConfig(strategy=CompressionStrategy.WEIGHTED_PRUNE, target_tokens=700, preserve_recent=0)
    result = compress([bg, normal, critical], config)
    ids = {b.id for b in result.blocks}
    assert bg.id not in ids
    assert critical.id in ids


def test_hybrid_applies_prune_then_sliding_window():
    blocks = [make_block(ContextWeight.NORMAL, 1000, f"block {i}") for i in range(5)]
    config = SurgeonConfig(strategy=CompressionStrategy.HYBRID, target_tokens=2000, preserve_recent=2)
    result = compress(blocks, config)
    assert result.tokens_after <= 3000


def test_compress_never_removes_critical_blocks():
    critical = make_block(ContextWeight.CRITICAL, 9000, "must keep")
    bg = make_block(ContextWeight.BACKGROUND, 100, "droppable")
    config = SurgeonConfig(strategy=CompressionStrategy.HYBRID, target_tokens=500, preserve_recent=0)
    result = compress([critical, bg], config)
    assert any(b.id == critical.id for b in result.blocks)


def test_surgeon_result_tokens_saved():
    blocks = [make_block(ContextWeight.BACKGROUND, 500) for _ in range(4)]
    config = SurgeonConfig(strategy=CompressionStrategy.WEIGHTED_PRUNE, target_tokens=500, preserve_recent=0)
    result = compress(blocks, config)
    assert result.tokens_saved == result.tokens_before - result.tokens_after


def test_score_relevance_empty_task_returns_zero():
    block = make_block(ContextWeight.NORMAL, 10, "some content")
    assert score_relevance(block, "") == 0.0


def test_score_relevance_exact_match():
    block = make_block(ContextWeight.NORMAL, 10, "pricing cloud storage")
    score = score_relevance(block, "pricing cloud storage")
    assert score == 1.0


def test_detect_contradiction_catches_not_pattern():
    a = make_block(ContextWeight.NORMAL, 10, "the service is available")
    b = make_block(ContextWeight.NORMAL, 10, "the service is not available")
    assert detect_contradiction(a, b) is True


def test_detect_contradiction_no_false_positive():
    a = make_block(ContextWeight.NORMAL, 10, "the weather is sunny today")
    b = make_block(ContextWeight.NORMAL, 10, "the price is twenty dollars")
    assert detect_contradiction(a, b) is False
