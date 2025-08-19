from __future__ import annotations

import numpy as np

from chunkasr.decode.rnnt_greedy import rnnt_greedy_decode
from chunkasr.model.transducer import Transducer


def test_transducer_greedy_returns_tokens() -> None:
    transducer = Transducer.random(enc_dim=16, vocab_size=6, seed=0)
    encoder_out = np.random.default_rng(0).standard_normal((12, 16)).astype(np.float32)
    tokens = rnnt_greedy_decode(encoder_out, transducer)
    assert isinstance(tokens, list)
    assert all(0 <= t < 6 and t != transducer.blank_id for t in tokens)


def test_transducer_is_deterministic() -> None:
    transducer = Transducer.random(enc_dim=16, vocab_size=6, seed=0)
    encoder_out = np.random.default_rng(1).standard_normal((8, 16)).astype(np.float32)
    first = rnnt_greedy_decode(encoder_out, transducer)
    second = rnnt_greedy_decode(encoder_out, transducer)
    assert first == second


def test_max_symbols_bounds_output() -> None:
    transducer = Transducer.random(enc_dim=8, vocab_size=4, seed=3)
    encoder_out = np.zeros((5, 8), dtype=np.float32)
    tokens = rnnt_greedy_decode(encoder_out, transducer, max_symbols_per_step=2)
    assert len(tokens) <= 5 * 2
