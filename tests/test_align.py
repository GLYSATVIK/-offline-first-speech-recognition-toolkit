from __future__ import annotations

import numpy as np

from chunkasr.align.timestamps import token_timestamps, word_timings
from chunkasr.decode.vocab import Vocabulary
from chunkasr.model.backend import log_softmax


def _peaked(ids: list[int], vocab_size: int) -> np.ndarray:
    logits = np.full((len(ids), vocab_size), -10.0, dtype=np.float32)
    for t, c in enumerate(ids):
        logits[t, c] = 10.0
    return log_softmax(logits)


def test_token_timestamps() -> None:
    vocab = Vocabulary.ascii_lower()
    h, i = vocab.symbol_to_id("h"), vocab.symbol_to_id("i")
    lp = _peaked([h, h, vocab.blank_id, i], len(vocab))
    spans = token_timestamps(lp, vocab)
    assert [s.token for s in spans] == ["h", "i"]
    assert spans[0].start_frame == 0
    assert spans[0].end_frame == 2
    assert spans[1].start_frame == 3


def test_word_timings_group_on_space() -> None:
    vocab = Vocabulary.ascii_lower()
    ids = [
        vocab.symbol_to_id("h"),
        vocab.symbol_to_id("i"),
        vocab.symbol_to_id(vocab.space_symbol),
        vocab.symbol_to_id("y"),
        vocab.symbol_to_id("o"),
        vocab.symbol_to_id("u"),
    ]
    lp = _peaked(ids, len(vocab))
    spans = token_timestamps(lp, vocab)
    words = word_timings(spans, vocab, frame_shift=160, sample_rate=16000)
    assert [w.word for w in words] == ["hi", "you"]
    assert words[0].start <= words[0].end
