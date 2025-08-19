from __future__ import annotations

from chunkasr.metrics.edit_distance import EditCounts, levenshtein


def test_levenshtein_distance() -> None:
    counts = levenshtein(list("kitten"), list("sitting"))
    assert counts.distance == 3


def test_operation_breakdown() -> None:
    counts = levenshtein(["a", "b", "c"], ["a", "x", "c"])
    assert counts.substitutions == 1
    assert counts.insertions == 0
    assert counts.deletions == 0
    assert counts.hits == 2


def test_insertion_and_deletion() -> None:
    assert levenshtein(["a", "b"], ["a", "b", "c"]).insertions == 1
    assert levenshtein(["a", "b", "c"], ["a", "b"]).deletions == 1


def test_error_rate() -> None:
    counts = levenshtein(["a", "b", "c", "d"], ["a", "x", "c", "d"])
    assert counts.error_rate == 0.25


def test_counts_add() -> None:
    total = EditCounts(1, 0, 0, 1, 2) + EditCounts(0, 1, 0, 1, 2)
    assert total.substitutions == 1
    assert total.insertions == 1
    assert total.ref_length == 4


def test_empty_reference() -> None:
    assert levenshtein([], []).error_rate == 0.0
