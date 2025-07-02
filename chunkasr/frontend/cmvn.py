"""Cepstral mean and variance normalisation (CMVN)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import FloatArray


@dataclass
class CMVN:
    """Global CMVN statistics that can be fit once and reused."""

    mean: FloatArray
    std: FloatArray

    @classmethod
    def fit(cls, features: FloatArray, eps: float = 1e-8) -> CMVN:
        """Estimate per-dimension mean and standard deviation."""
        mean = features.mean(axis=0)
        std = features.std(axis=0) + eps
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def __call__(self, features: FloatArray) -> FloatArray:
        return ((features - self.mean) / self.std).astype(np.float32, copy=False)


def apply_cmvn(features: FloatArray, eps: float = 1e-8) -> FloatArray:
    """Utterance-level CMVN: normalise ``features`` by their own statistics."""
    if features.shape[0] == 0:
        return features
    return CMVN.fit(features, eps)(features)


def sliding_window_cmvn(
    features: FloatArray,
    window: int = 300,
    center: bool = True,
    eps: float = 1e-8,
) -> FloatArray:
    """Online-style CMVN using a sliding context window per frame.

    ``window`` is measured in frames. This mirrors the streaming case where
    global statistics are not yet available.
    """
    n = features.shape[0]
    if n == 0:
        return features
    out = np.empty_like(features, dtype=np.float32)
    half = window // 2
    for t in range(n):
        if center:
            lo, hi = max(0, t - half), min(n, t + half + 1)
        else:
            lo, hi = max(0, t - window + 1), t + 1
        ctx = features[lo:hi]
        mean = ctx.mean(axis=0)
        std = ctx.std(axis=0) + eps
        out[t] = (features[t] - mean) / std
    return out
