"""A small reference RNN-T (transducer) for the streaming decode path.

Like :mod:`chunkasr.model.linear` this is a reference implementation rather than
a trained model: a recurrent predictor and a feed-forward joiner with random
but seeded weights. It exists so the RNN-T greedy decoder has something concrete
and deterministic to run against.
"""

from __future__ import annotations

import numpy as np

from ..types import FloatArray
from .backend import log_softmax


class _Affine:
    """A minimal affine layer ``x @ W + b``."""

    def __init__(self, weight: FloatArray, bias: FloatArray) -> None:
        self.weight = weight
        self.bias = bias

    def __call__(self, x: FloatArray) -> FloatArray:
        return (x @ self.weight + self.bias).astype(np.float32, copy=False)

    @classmethod
    def random(cls, in_dim: int, out_dim: int, gen: np.random.Generator, scale: float) -> _Affine:
        weight = (gen.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)
        bias = np.zeros(out_dim, dtype=np.float32)
        return cls(weight, bias)


class Predictor:
    """Recurrent label predictor mapping token history to a state vector."""

    def __init__(self, embedding: FloatArray, recurrent: _Affine, blank_id: int = 0) -> None:
        self.embedding = embedding
        self.recurrent = recurrent
        self.blank_id = blank_id
        self.dim = embedding.shape[1]

    def initial_state(self) -> FloatArray:
        return np.zeros(self.dim, dtype=np.float32)

    def step(self, state: FloatArray, token: int) -> FloatArray:
        """Advance the predictor by one emitted (non-blank) token."""
        emb = self.embedding[token]
        combined = np.concatenate([state, emb]).astype(np.float32)
        return np.tanh(self.recurrent(combined)).astype(np.float32)

    @classmethod
    def random(cls, vocab_size: int, dim: int, gen: np.random.Generator) -> Predictor:
        embedding = (gen.standard_normal((vocab_size, dim)) * 0.1).astype(np.float32)
        recurrent = _Affine.random(dim * 2, dim, gen, scale=0.1)
        return cls(embedding, recurrent)


class Joiner:
    """Feed-forward joiner combining encoder and predictor representations."""

    def __init__(self, enc_proj: _Affine, pred_proj: _Affine, out_proj: _Affine) -> None:
        self.enc_proj = enc_proj
        self.pred_proj = pred_proj
        self.out_proj = out_proj

    def join(self, enc_frame: FloatArray, pred_out: FloatArray) -> FloatArray:
        """Return log-probabilities for one (time, label) step."""
        hidden = np.tanh(self.enc_proj(enc_frame) + self.pred_proj(pred_out))
        return log_softmax(self.out_proj(hidden.astype(np.float32)), axis=-1)

    @classmethod
    def random(
        cls,
        enc_dim: int,
        pred_dim: int,
        vocab_size: int,
        hidden: int,
        gen: np.random.Generator,
    ) -> Joiner:
        return cls(
            _Affine.random(enc_dim, hidden, gen, scale=0.1),
            _Affine.random(pred_dim, hidden, gen, scale=0.1),
            _Affine.random(hidden, vocab_size, gen, scale=0.1),
        )


class Transducer:
    """Bundle of a predictor and joiner with a fixed blank symbol."""

    def __init__(self, predictor: Predictor, joiner: Joiner, vocab_size: int, blank_id: int = 0):
        self.predictor = predictor
        self.joiner = joiner
        self.vocab_size = vocab_size
        self.blank_id = blank_id

    @classmethod
    def random(
        cls,
        enc_dim: int,
        vocab_size: int,
        pred_dim: int = 32,
        hidden: int = 32,
        seed: int = 1234,
    ) -> Transducer:
        gen = np.random.default_rng(seed)
        predictor = Predictor.random(vocab_size, pred_dim, gen)
        joiner = Joiner.random(enc_dim, pred_dim, vocab_size, hidden, gen)
        return cls(predictor, joiner, vocab_size)
