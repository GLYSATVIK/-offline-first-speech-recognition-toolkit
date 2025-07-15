"""Streaming encoder with limited right context.

Wraps any :class:`~chunkasr.model.base.AcousticModel` and feeds it feature
frames online. A frame is only emitted once its right-context lookahead has
arrived, which is what makes the encoder *causal with bounded latency*: the
output for frame ``t`` depends on frames up to ``t + right_context`` and no
further. Fed the whole utterance at once it reproduces the offline result.
"""

from __future__ import annotations

import numpy as np

from ..model.base import AcousticModel
from ..types import FloatArray


class StreamingEncoder:
    """Online driver around an acoustic model with a right-context buffer."""

    def __init__(
        self,
        model: AcousticModel,
        chunk_size: int = 16,
        right_context: int = 4,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if right_context < 0:
            raise ValueError("right_context must be non-negative")
        self.model = model
        self.chunk_size = chunk_size
        self.right_context = right_context
        self._buffer: FloatArray = np.zeros((0, model.input_dim), dtype=np.float32)
        self._emitted = 0

    def reset(self) -> None:
        """Clear all buffered state so the encoder can process a new stream."""
        self._buffer = np.zeros((0, self.model.input_dim), dtype=np.float32)
        self._emitted = 0

    def _smooth(self, index: int, total: int) -> FloatArray:
        """Average frame ``index`` with its available right-context window."""
        end = min(index + self.right_context + 1, total)
        return self._buffer[index:end].mean(axis=0).astype(np.float32)

    def _emit_range(self, start: int, stop: int, total: int) -> FloatArray:
        rows = [self._smooth(i, total) for i in range(start, stop)]
        window = np.stack(rows, axis=0) if rows else np.zeros((0, self.model.input_dim), np.float32)
        return self.model.forward(window)

    def accept(self, frames: FloatArray) -> FloatArray:
        """Push new feature frames and return any log-probs now emittable."""
        if frames.shape[0] == 0:
            return np.zeros((0, self.model.vocab_size), dtype=np.float32)
        self._buffer = np.concatenate([self._buffer, frames.astype(np.float32)], axis=0)
        total = self._buffer.shape[0]
        ready = ((total - self.right_context) // self.chunk_size) * self.chunk_size
        ready = max(ready, 0)
        return self._flush_to(ready, total)

    def flush(self) -> FloatArray:
        """Emit every remaining frame, padding the missing right context."""
        total = self._buffer.shape[0]
        return self._flush_to(total, total)

    def _flush_to(self, ready: int, total: int) -> FloatArray:
        outputs = []
        while self._emitted < ready:
            end = min(self._emitted + self.chunk_size, ready)
            outputs.append(self._emit_range(self._emitted, end, total))
            self._emitted = end
        if outputs:
            return np.concatenate(outputs, axis=0)
        return np.zeros((0, self.model.vocab_size), dtype=np.float32)

    def encode(self, features: FloatArray) -> FloatArray:
        """Offline convenience: stream ``features`` through in one shot."""
        self.reset()
        head = self.accept(features)
        tail = self.flush()
        return np.concatenate([head, tail], axis=0)
