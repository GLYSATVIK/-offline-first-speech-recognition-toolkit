"""Derive token and word timings from a CTC emission matrix.

The greedy best path gives, for every collapsed token, the contiguous frame
range it occupied. Multiplying frame indices by the hop size turns those ranges
into seconds; grouping on the space symbol turns tokens into word timings.
"""

from __future__ import annotations

import numpy as np

from ..decode.ctc_greedy import greedy_path
from ..decode.vocab import Vocabulary
from ..results import TokenSpan, WordTiming
from ..types import FloatArray


def token_timestamps(
    log_probs: FloatArray,
    vocab: Vocabulary,
    blank_id: int = 0,
) -> list[TokenSpan]:
    """Return per-token frame spans from the greedy CTC path."""
    path = greedy_path(log_probs)
    spans: list[TokenSpan] = []
    current: int | None = None
    start = 0

    def close(token: int, begin: int, end: int) -> None:
        score = float(np.mean(log_probs[begin:end, token])) if end > begin else 0.0
        spans.append(TokenSpan(vocab.id_to_symbol(token), begin, end, score))

    for i, raw in enumerate(path.tolist()):
        token = int(raw)
        if token == blank_id:
            if current is not None:
                close(current, start, i)
                current = None
        elif current is None:
            current, start = token, i
        elif token != current:
            close(current, start, i)
            current, start = token, i
    if current is not None:
        close(current, start, path.shape[0])
    return spans


def word_timings(
    spans: list[TokenSpan],
    vocab: Vocabulary,
    frame_shift: int = 160,
    sample_rate: int = 16000,
) -> list[WordTiming]:
    """Group token spans into words, splitting on the space symbol."""
    seconds = frame_shift / float(sample_rate)
    words: list[WordTiming] = []
    letters: list[str] = []
    begin: int | None = None
    last_end = 0

    def flush() -> None:
        if letters and begin is not None:
            words.append(WordTiming("".join(letters), begin * seconds, last_end * seconds))

    for span in spans:
        if span.token == vocab.space_symbol:
            flush()
            letters = []
            begin = None
            continue
        if begin is None:
            begin = span.start_frame
        letters.append(span.token)
        last_end = span.end_frame
    flush()
    return words
