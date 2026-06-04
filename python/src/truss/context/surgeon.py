from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from truss.types import ContextBlock, ContextWeight


class CompressionStrategy(Enum):
    SLIDING_WINDOW = "sliding_window"
    WEIGHTED_PRUNE = "weighted_prune"
    HYBRID = "hybrid"


@dataclass
class SurgeonConfig:
    strategy: CompressionStrategy = CompressionStrategy.HYBRID
    target_tokens: int = 8_000
    preserve_recent: int = 5
    keep_recent: int = 0


@dataclass
class SurgeonResult:
    blocks: list[ContextBlock]
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    strategy_applied: str


def compress(blocks: list[ContextBlock], config: SurgeonConfig) -> SurgeonResult:
    tokens_before = sum(b.token_count for b in blocks)

    if config.strategy == CompressionStrategy.SLIDING_WINDOW:
        kept = _sliding_window(blocks, config.keep_recent or config.preserve_recent, config.preserve_recent)
    elif config.strategy == CompressionStrategy.WEIGHTED_PRUNE:
        kept = _weighted_prune(blocks, config.target_tokens, config.preserve_recent)
    else:  # HYBRID: prune first, then window if still over budget
        after_prune = _weighted_prune(blocks, config.target_tokens, config.preserve_recent)
        if sum(b.token_count for b in after_prune) > config.target_tokens:
            kept = _sliding_window(after_prune, config.keep_recent or config.preserve_recent, config.preserve_recent)
        else:
            kept = after_prune

    tokens_after = sum(b.token_count for b in kept)
    return SurgeonResult(
        blocks=kept,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_before - tokens_after,
        strategy_applied=config.strategy.value,
    )


def _sliding_window(blocks: list[ContextBlock], keep_recent: int, preserve_recent: int) -> list[ContextBlock]:
    if keep_recent == 0:
        return list(blocks)
    always_keep = max(preserve_recent, keep_recent)
    if len(blocks) <= always_keep:
        return list(blocks)
    pinned = [b for b in blocks if b.weight >= ContextWeight.HIGH]
    pinned_ids = {b.id for b in pinned}
    recent_start = max(0, len(blocks) - keep_recent)
    result = list(pinned)
    for b in blocks[recent_start:]:
        if b.id not in pinned_ids:
            result.append(b)
    return result


def _weighted_prune(blocks: list[ContextBlock], target_tokens: int, preserve_recent: int) -> list[ContextBlock]:
    total = sum(b.token_count for b in blocks)
    if total <= target_tokens:
        return list(blocks)

    preserve_ids = {b.id for b in blocks[-preserve_recent:]} if preserve_recent > 0 else set()

    removable = sorted(
        [b for b in blocks if b.id not in preserve_ids and b.weight < ContextWeight.HIGH],
        key=lambda b: (int(b.weight), b.created_at),
    )

    to_remove: set = set()
    running = total
    for b in removable:
        if running <= target_tokens:
            break
        running -= b.token_count
        to_remove.add(b.id)

    return [b for b in blocks if b.id not in to_remove]


def score_relevance(block: ContextBlock, task: str) -> float:
    """Keyword overlap between block content and task string. Returns 0.0-1.0."""
    task_words = set(task.split())
    if not task_words:
        return 0.0
    matches = sum(1 for w in block.content.split() if w in task_words)
    return min(matches / len(task_words), 1.0)


def detect_contradiction(a: ContextBlock, b: ContextBlock) -> bool:
    """Heuristic: True if A asserts X and B asserts 'not X'.

    Single-direction by design: checks words from A against negations in B only.
    This avoids false positives where A contains incidental negations (e.g. "do not press").
    """
    a_lower = a.content.lower()
    b_lower = b.content.lower()
    for word in a_lower.split():
        if len(word) > 4 and f"not {word}" in b_lower:
            return True
    return False
