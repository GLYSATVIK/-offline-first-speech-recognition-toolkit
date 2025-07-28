"""Greedy RNN-T (transducer) decoding.

Frame-synchronous greedy search: at each encoder frame we emit non-blank
symbols (advancing the predictor) until the joiner prefers blank, then move on.
"""

from __future__ import annotations

import numpy as np

from ..model.transducer import Transducer
from ..types import FloatArray


def rnnt_greedy_decode(
    encoder_out: FloatArray,
    transducer: Transducer,
    max_symbols_per_step: int = 5,
) -> list[int]:
    """Greedily decode encoder outputs with an RNN-T ``transducer``."""
    if encoder_out.ndim != 2:
        raise ValueError("encoder_out must be 2-D (n_frames, enc_dim)")
    predictor = transducer.predictor
    joiner = transducer.joiner
    blank = transducer.blank_id

    state = predictor.initial_state()
    pred_out = state
    tokens: list[int] = []

    for t in range(encoder_out.shape[0]):
        frame = encoder_out[t]
        for _ in range(max_symbols_per_step):
            log_probs = joiner.join(frame, pred_out)
            token = int(np.argmax(log_probs))
            if token == blank:
                break
            tokens.append(token)
            state = predictor.step(state, token)
            pred_out = state
    return tokens
