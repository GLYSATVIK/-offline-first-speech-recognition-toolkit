"""End-to-end feature extraction pipeline.

Composes the log-Mel frontend with optional delta features and CMVN into the
single object the recogniser calls.
"""

from __future__ import annotations

import numpy as np

from ..config import FeatureConfig
from ..types import FloatArray
from .cmvn import apply_cmvn
from .fbank import LogMelExtractor


def add_deltas(features: FloatArray, order: int = 2, width: int = 2) -> FloatArray:
    """Append delta (and delta-delta) features along the feature axis."""
    outputs = [features]
    current = features
    for _ in range(order):
        current = _delta(current, width)
        outputs.append(current)
    return np.concatenate(outputs, axis=1).astype(np.float32, copy=False)


def _delta(features: FloatArray, width: int) -> FloatArray:
    """Regression-based delta coefficients with edge padding."""
    n = features.shape[0]
    if n == 0:
        return features
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    denom = 2.0 * sum(i * i for i in range(1, width + 1))
    delta = np.zeros_like(features, dtype=np.float64)
    for i in range(1, width + 1):
        delta += i * (padded[width + i : width + i + n] - padded[width - i : width - i + n])
    return (delta / denom).astype(np.float32)


class FeatureExtractor:
    """Log-Mel features with optional deltas and utterance-level CMVN."""

    def __init__(
        self,
        config: FeatureConfig | None = None,
        apply_cmvn_norm: bool = True,
        delta_order: int = 0,
    ) -> None:
        self.config = config or FeatureConfig()
        self.apply_cmvn_norm = apply_cmvn_norm
        self.delta_order = delta_order
        self._mel = LogMelExtractor(self.config)

    @property
    def output_dim(self) -> int:
        dim = self._mel.output_dim
        return dim * (1 + self.delta_order)

    def __call__(self, signal: FloatArray) -> FloatArray:
        features = self._mel(signal)
        if self.delta_order > 0:
            features = add_deltas(features, order=self.delta_order)
        if self.apply_cmvn_norm:
            features = apply_cmvn(features)
        return features
