from __future__ import annotations

import numpy as np

from chunkasr.model.linear import LinearAcousticModel
from chunkasr.streaming.chunker import iter_chunks
from chunkasr.streaming.encoder import StreamingEncoder


def _features(n: int = 40, dim: int = 8) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)


def test_iter_chunks_covers_everything() -> None:
    feats = _features(n=20)
    chunks = list(iter_chunks(feats, chunk_size=8, right_context=2))
    assert sum(c.num_frames for c in chunks) == 20
    assert chunks[-1].is_last
    assert chunks[0].context.shape[0] == 2


def test_streaming_matches_offline() -> None:
    feats = _features(n=40)
    model = LinearAcousticModel.random(8, 5, seed=1)
    enc = StreamingEncoder(model, chunk_size=8, right_context=4)

    full = enc.encode(feats)
    assert full.shape == (40, 5)

    enc.reset()
    a = enc.accept(feats[:20])
    b = enc.accept(feats[20:])
    c = enc.flush()
    streamed = np.concatenate([a, b, c], axis=0)

    assert streamed.shape == full.shape
    assert np.allclose(streamed, full)


def test_streaming_emits_incrementally() -> None:
    feats = _features(n=32)
    model = LinearAcousticModel.random(8, 5, seed=2)
    enc = StreamingEncoder(model, chunk_size=8, right_context=4)
    emitted = enc.accept(feats)
    # With right context 4, the last few frames wait for flush.
    assert emitted.shape[0] < 32
    tail = enc.flush()
    assert emitted.shape[0] + tail.shape[0] == 32
