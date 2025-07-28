"""CTC prefix beam search with optional shallow LM fusion.

Standard log-domain prefix beam search: each prefix tracks the probability of
paths ending in blank versus a real symbol, and an optional :class:`PrefixScorer`
adds a language-model term when a new symbol extends a prefix.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from ..model.backend import log_add
from ..types import FloatArray

NEG_INF = float("-inf")


class PrefixScorer(Protocol):
    """Scores extending ``prefix_ids`` with ``token`` (e.g. an n-gram LM)."""

    def score(self, prefix_ids: tuple[int, ...] | list[int], token: int) -> float: ...


class _NullScorer:
    def score(self, prefix_ids: tuple[int, ...] | list[int], token: int) -> float:
        return 0.0


def ctc_prefix_beam_search(
    log_probs: FloatArray,
    beam_size: int = 8,
    blank_id: int = 0,
    scorer: PrefixScorer | None = None,
    prune_threshold: float = NEG_INF,
) -> list[int]:
    """Return the highest-scoring token id sequence under prefix beam search."""
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be 2-D (n_frames, vocab_size)")
    scorer = scorer or _NullScorer()
    vocab_size = log_probs.shape[1]

    beams: dict[tuple[int, ...], list[float]] = {(): [0.0, NEG_INF]}

    for t in range(log_probs.shape[0]):
        frame = log_probs[t]
        nxt: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF])
        for prefix, (p_b, p_nb) in beams.items():
            p_total = log_add(p_b, p_nb)
            # (1) emit blank: prefix unchanged, mass flows to its blank slot.
            nxt[prefix][0] = log_add(nxt[prefix][0], p_total + float(frame[blank_id]))
            # (2) repeat last symbol: stays in the same prefix, non-blank slot.
            if prefix:
                last = prefix[-1]
                nxt[prefix][1] = log_add(nxt[prefix][1], p_nb + float(frame[last]))
            # (3) extend by a new symbol.
            for s in range(vocab_size):
                if s == blank_id:
                    continue
                ls = float(frame[s])
                if ls < prune_threshold:
                    continue
                new_prefix = (*prefix, s)
                base = p_b if (prefix and s == prefix[-1]) else p_total
                add = base + ls + scorer.score(prefix, s)
                nxt[new_prefix][1] = log_add(nxt[new_prefix][1], add)
        beams = _prune(nxt, beam_size)

    best = max(beams.items(), key=lambda kv: log_add(kv[1][0], kv[1][1]))
    return list(best[0])


def _prune(
    beams: dict[tuple[int, ...], list[float]],
    beam_size: int,
) -> dict[tuple[int, ...], list[float]]:
    """Keep the ``beam_size`` most probable prefixes."""
    ranked = sorted(beams.items(), key=lambda kv: log_add(kv[1][0], kv[1][1]), reverse=True)
    return dict(ranked[:beam_size])
