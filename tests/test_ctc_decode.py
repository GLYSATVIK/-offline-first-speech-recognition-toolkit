from __future__ import annotations

import numpy as np

from chunkasr.decode.ctc_greedy import collapse, ctc_greedy_decode, greedy_path
from chunkasr.decode.ctc_prefix_beam import ctc_prefix_beam_search
from chunkasr.model.backend import log_softmax


def peaked(seq: list[int], vocab_size: int = 4) -> np.ndarray:
    logits = np.full((len(seq), vocab_size), -10.0, dtype=np.float32)
    for t, c in enumerate(seq):
        logits[t, c] = 10.0
    return log_softmax(logits)


def test_greedy_path_and_collapse() -> None:
    lp = peaked([1, 1, 0, 2, 2])
    assert greedy_path(lp).tolist() == [1, 1, 0, 2, 2]
    assert collapse(greedy_path(lp)) == [1, 2]


def test_greedy_decode() -> None:
    lp = peaked([1, 1, 0, 0, 3, 3])
    assert ctc_greedy_decode(lp) == [1, 3]


def test_beam_matches_greedy_on_peaked_input() -> None:
    lp = peaked([1, 1, 0, 2, 0, 3])
    assert ctc_prefix_beam_search(lp, beam_size=8) == ctc_greedy_decode(lp)


def test_beam_handles_empty_output() -> None:
    lp = peaked([0, 0, 0])  # all blank
    assert ctc_prefix_beam_search(lp) == []
