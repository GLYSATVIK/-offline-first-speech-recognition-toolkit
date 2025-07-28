"""CTC greedy (best-path) decoding."""

from __future__ import annotations

import numpy as np

from ..types import FloatArray, IntArray


def greedy_path(log_probs: FloatArray) -> IntArray:
    """Return the per-frame argmax token ids (the raw CTC alignment path)."""
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be 2-D (n_frames, vocab_size)")
    return np.argmax(log_probs, axis=1).astype(np.int64)


def collapse(path: IntArray, blank_id: int = 0) -> list[int]:
    """Collapse a CTC path: merge repeats, then drop blanks."""
    out: list[int] = []
    prev = -1
    for raw in path.tolist():
        token = int(raw)
        if token != prev and token != blank_id:
            out.append(token)
        prev = token
    return out


def ctc_greedy_decode(log_probs: FloatArray, blank_id: int = 0) -> list[int]:
    """Greedy CTC decode: argmax path collapsed to a token id sequence."""
    return collapse(greedy_path(log_probs), blank_id=blank_id)
