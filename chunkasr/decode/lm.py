"""Shallow n-gram language model and prefix-scorer adapters.

The LM is a back-off n-gram with add-k smoothing built directly from a text
corpus. :class:`CharLMScorer` adapts it to the interface the CTC prefix beam
search expects, so LM fusion is a drop-in weight rather than a special case.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

from .ngram import count_ngrams
from .vocab import Vocabulary


class NGramLM:
    """Back-off n-gram model with add-k smoothing over a fixed vocabulary."""

    def __init__(
        self,
        order: int,
        counts: dict[int, Counter[tuple[str, ...]]],
        contexts: dict[int, Counter[tuple[str, ...]]],
        vocab: set[str],
        k: float = 0.5,
        backoff: float = 0.4,
    ) -> None:
        self.order = order
        self.counts = counts
        self.contexts = contexts
        self.vocab = vocab
        self.vocab_size = max(1, len(vocab))
        self.k = k
        self.backoff = backoff

    @classmethod
    def train(
        cls,
        corpus: Iterable[Sequence[str]],
        order: int = 3,
        k: float = 0.5,
        backoff: float = 0.4,
    ) -> NGramLM:
        """Estimate counts for every order up to ``order`` from ``corpus``."""
        corpus_list = [list(tokens) for tokens in corpus]
        counts: dict[int, Counter[tuple[str, ...]]] = {}
        contexts: dict[int, Counter[tuple[str, ...]]] = {}
        vocab: set[str] = set()
        for n in range(1, order + 1):
            ngram_counts, context_counts, seen = count_ngrams(corpus_list, n)
            counts[n] = ngram_counts
            contexts[n] = context_counts
            vocab |= seen
        return cls(order, counts, contexts, vocab, k, backoff)

    @classmethod
    def train_chars(cls, texts: Iterable[str], order: int = 3) -> NGramLM:
        """Train a character-level LM from raw strings."""
        return cls.train(([*text] for text in texts), order=order)

    def _prob(self, ctx: tuple[str, ...], token: str) -> float:
        n = len(ctx) + 1
        c_full = self.counts.get(n, Counter()).get((*ctx, token), 0)
        c_ctx = self.contexts.get(n, Counter()).get(ctx, 0)
        if c_ctx > 0:
            return (c_full + self.k) / (c_ctx + self.k * self.vocab_size)
        if n == 1:
            return 1.0 / self.vocab_size
        return self.backoff * self._prob(ctx[1:], token)

    def log_prob(self, context: Sequence[str], token: str) -> float:
        """Log-probability of ``token`` given a (possibly long) ``context``."""
        window = self.order - 1
        ctx = tuple(context)[-window:] if window > 0 else ()
        return math.log(self._prob(ctx, token))

    def score(self, tokens: Sequence[str]) -> float:
        """Total log-probability of a token sequence."""
        total = 0.0
        history: list[str] = []
        for token in tokens:
            total += self.log_prob(history, token)
            history.append(token)
        return total


class NoLM:
    """A null scorer: contributes nothing to the beam score."""

    def score(self, prefix_ids: Sequence[int], token: int) -> float:
        return 0.0


class CharLMScorer:
    """Adapts a character :class:`NGramLM` to the prefix-scorer interface."""

    def __init__(self, lm: NGramLM, vocab: Vocabulary, weight: float = 0.5) -> None:
        self.lm = lm
        self.vocab = vocab
        self.weight = weight

    def score(self, prefix_ids: Sequence[int], token: int) -> float:
        if self.weight == 0.0:
            return 0.0
        history = [self.vocab.id_to_symbol(i) for i in prefix_ids]
        symbol = self.vocab.id_to_symbol(token)
        return self.weight * self.lm.log_prob(history, symbol)
