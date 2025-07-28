"""Levenshtein edit distance with a substitution/insertion/deletion breakdown.

The DP is generic over any sequence of comparable tokens, so the same routine
backs both word- and character-error rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EditCounts:
    """Aligned edit statistics between a reference and hypothesis."""

    substitutions: int
    insertions: int
    deletions: int
    hits: int
    ref_length: int

    @property
    def distance(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def error_rate(self) -> float:
        if self.ref_length == 0:
            return 0.0 if self.distance == 0 else float("inf")
        return self.distance / self.ref_length

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            self.substitutions + other.substitutions,
            self.insertions + other.insertions,
            self.deletions + other.deletions,
            self.hits + other.hits,
            self.ref_length + other.ref_length,
        )


def levenshtein(ref: Sequence[Any], hyp: Sequence[Any]) -> EditCounts:
    """Edit distance between ``ref`` and ``hyp`` with an operation breakdown."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return _backtrace(dp, ref, hyp)


def _backtrace(dp: list[list[int]], ref: Sequence[Any], hyp: Sequence[Any]) -> EditCounts:
    i, j = len(ref), len(hyp)
    sub = ins = dele = hits = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            hits += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            sub += 1
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dele += 1
            i -= 1
    return EditCounts(sub, ins, dele, hits, len(ref))
