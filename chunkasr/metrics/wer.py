"""Word error rate."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .edit_distance import EditCounts, levenshtein
from .text_norm import TextNormalizer


@dataclass(frozen=True)
class WERResult:
    """Word-error-rate result carrying the underlying edit counts."""

    counts: EditCounts

    @property
    def wer(self) -> float:
        return self.counts.error_rate

    @property
    def substitutions(self) -> int:
        return self.counts.substitutions

    @property
    def insertions(self) -> int:
        return self.counts.insertions

    @property
    def deletions(self) -> int:
        return self.counts.deletions

    def as_percent(self) -> float:
        return 100.0 * self.wer


def wer(reference: str, hypothesis: str, normalizer: TextNormalizer | None = None) -> WERResult:
    """Compute WER between two strings after normalisation."""
    norm = normalizer or TextNormalizer()
    ref_words = norm(reference).split()
    hyp_words = norm(hypothesis).split()
    return WERResult(levenshtein(ref_words, hyp_words))


def corpus_wer(
    references: Iterable[str],
    hypotheses: Iterable[str],
    normalizer: TextNormalizer | None = None,
) -> WERResult:
    """Aggregate WER across a corpus by summing edit counts."""
    norm = normalizer or TextNormalizer()
    total = EditCounts(0, 0, 0, 0, 0)
    for ref, hyp in zip(references, hypotheses, strict=True):
        total = total + levenshtein(norm(ref).split(), norm(hyp).split())
    return WERResult(total)
