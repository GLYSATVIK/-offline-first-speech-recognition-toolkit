"""Character error rate."""

from __future__ import annotations

from collections.abc import Iterable

from .edit_distance import EditCounts, levenshtein
from .text_norm import TextNormalizer


def cer(reference: str, hypothesis: str, normalizer: TextNormalizer | None = None) -> float:
    """Compute character error rate between two strings after normalisation."""
    norm = normalizer or TextNormalizer()
    return levenshtein(list(norm(reference)), list(norm(hypothesis))).error_rate


def corpus_cer(
    references: Iterable[str],
    hypotheses: Iterable[str],
    normalizer: TextNormalizer | None = None,
) -> float:
    """Aggregate CER across a corpus by summing edit counts."""
    norm = normalizer or TextNormalizer()
    total = EditCounts(0, 0, 0, 0, 0)
    for ref, hyp in zip(references, hypotheses, strict=True):
        total = total + levenshtein(list(norm(ref)), list(norm(hyp)))
    return total.error_rate
