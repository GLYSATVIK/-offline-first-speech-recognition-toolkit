"""A linear reference acoustic model.

This is intentionally simple: a single affine projection from features to
vocabulary logits followed by a log-softmax. It is not competitive, but it is
deterministic, trivially serialisable, and lets the whole decode/align/score
stack run end to end without a trained network. The optional PyTorch backend
computes the identical projection on tensors.
"""

from __future__ import annotations

import numpy as np

from ..types import FloatArray
from .backend import log_softmax


class LinearAcousticModel:
    """Affine ``features @ W + b`` projection with a log-softmax head."""

    def __init__(self, weight: FloatArray, bias: FloatArray, backend: str = "numpy") -> None:
        if weight.ndim != 2:
            raise ValueError("weight must be 2-D (input_dim, vocab_size)")
        if bias.shape != (weight.shape[1],):
            raise ValueError("bias shape must match the vocabulary dimension")
        self.weight = weight.astype(np.float32)
        self.bias = bias.astype(np.float32)
        self.backend = backend

    @property
    def input_dim(self) -> int:
        return int(self.weight.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.weight.shape[1])

    @classmethod
    def random(
        cls,
        input_dim: int,
        vocab_size: int,
        seed: int = 1234,
        scale: float = 0.1,
    ) -> LinearAcousticModel:
        """Create a model with small random weights for demos and tests."""
        gen = np.random.default_rng(seed)
        weight = gen.standard_normal((input_dim, vocab_size)).astype(np.float32) * scale
        bias = np.zeros(vocab_size, dtype=np.float32)
        return cls(weight, bias)

    def forward(self, features: FloatArray) -> FloatArray:
        """Project ``features`` to log-probabilities."""
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"feature dim {features.shape[-1]} != model input dim {self.input_dim}"
            )
        if self.backend == "torch":
            return self._forward_torch(features)
        logits = features.astype(np.float32) @ self.weight + self.bias
        return log_softmax(logits, axis=-1)

    def _forward_torch(self, features: FloatArray) -> FloatArray:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("torch backend requested but torch is not installed") from exc
        with torch.no_grad():
            feats = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
            weight = torch.from_numpy(self.weight)
            bias = torch.from_numpy(self.bias)
            logits = feats @ weight + bias
            out = torch.log_softmax(logits, dim=-1)
        return out.cpu().numpy().astype(np.float32)
