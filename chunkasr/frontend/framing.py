"""Signal framing.

Splits a 1-D signal into overlapping frames, the first step shared by the STFT
frontend and the VAD.
"""

from __future__ import annotations

import numpy as np

from ..types import FloatArray


def num_frames(signal_length: int, frame_length: int, frame_shift: int) -> int:
    """Number of full frames obtainable without padding."""
    if signal_length < frame_length:
        return 0
    return 1 + (signal_length - frame_length) // frame_shift


def frame_signal(
    signal: FloatArray,
    frame_length: int,
    frame_shift: int,
) -> FloatArray:
    """Return overlapping frames as a ``(n_frames, frame_length)`` array.

    Uses a strided view for zero-copy framing where possible, then a defensive
    copy so downstream in-place operations (windowing, dither) are safe.
    """
    if frame_length <= 0 or frame_shift <= 0:
        raise ValueError("frame_length and frame_shift must be positive")
    signal = np.ascontiguousarray(signal, dtype=np.float32)
    n = num_frames(signal.shape[0], frame_length, frame_shift)
    if n == 0:
        return np.zeros((0, frame_length), dtype=np.float32)
    stride = signal.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n, frame_length),
        strides=(stride * frame_shift, stride),
        writeable=False,
    )
    return np.array(frames, dtype=np.float32, copy=True)
