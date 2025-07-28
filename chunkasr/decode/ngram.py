"""N-gram counting utilities.

Separated from :mod:`chunkasr.decode.lm` so the counting logic can be tested in
isolation and reused (e.g. for corpus statistics).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence


def iter_ngrams(tokens: Sequence[str], order: int) -> Iterable[tuple[str, ...]]:
    """Yield padded ``order``-grams from a single token sequence.

    Sequences are padded with ``order - 1`` start markers and one end marker so
    that boundary contexts are modelled explicitly.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    padded = ["<s>"] * (order - 1) + list(tokens) + ["</s>"]
    for i in range(len(padded) - order + 1):
        yield tuple(padded[i : i + order])


def count_ngrams(
    corpus: Iterable[Sequence[str]],
    order: int,
) -> tuple[Counter[tuple[str, ...]], Counter[tuple[str, ...]], set[str]]:
    """Count full n-grams and their (n-1)-gram contexts across a corpus.

    Returns ``(ngram_counts, context_counts, vocabulary)``.
    """
    ngram_counts: Counter[tuple[str, ...]] = Counter()
    context_counts: Counter[tuple[str, ...]] = Counter()
    vocab: set[str] = {"</s>"}
    for tokens in corpus:
        vocab.update(tokens)
        for gram in iter_ngrams(tokens, order):
            ngram_counts[gram] += 1
            context_counts[gram[:-1]] += 1
    return ngram_counts, context_counts, vocab
