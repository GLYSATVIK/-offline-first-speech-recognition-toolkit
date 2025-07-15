"""Acoustic-model interface.

Decoders operate purely on emission matrices, so any object that can turn
features into per-frame log-probabilities is a valid acoustic model. The
:class:`AcousticModel` protocol pins that contract down.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import FloatArray


@runtime_checkable
class AcousticModel(Protocol):
    """Maps a feature matrix to frame-synchronous log-probabilities."""

    @property
    def vocab_size(self) -> int:
        """Number of output symbols including blank."""
        ...

    @property
    def input_dim(self) -> int:
        """Expected feature dimensionality."""
        ...

    def forward(self, features: FloatArray) -> FloatArray:
        """Return log-probabilities of shape ``(n_frames, vocab_size)``."""
        ...
