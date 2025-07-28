"""Small numeric helpers shared across the DSP kernels.

Nothing here is ASR-specific; they are the kind of utilities that would
otherwise be copy-pasted into every module.
"""

from __future__ import annotations

import numpy as np

from .types import FloatArray


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two greater than or equal to ``n``."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def ensure_float32(x: object) -> FloatArray:
    """Coerce an array-like to a contiguous ``float32`` array."""
    arr = np.ascontiguousarray(np.asarray(x))
    return arr.astype(np.float32, copy=False)


def to_mono(samples: FloatArray) -> FloatArray:
    """Average multi-channel audio down to a single channel.

    Accepts either ``(n,)`` or ``(channels, n)`` / ``(n, channels)`` layouts;
    the shorter axis is treated as the channel axis.
    """
    if samples.ndim == 1:
        return samples
    if samples.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {samples.shape}")
    channel_axis = 0 if samples.shape[0] < samples.shape[1] else 1
    mono = samples.mean(axis=channel_axis)
    return mono.astype(np.float32, copy=False)


def pad_or_trim(samples: FloatArray, length: int) -> FloatArray:
    """Right-pad with zeros or trim ``samples`` to exactly ``length``."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if samples.shape[0] == length:
        return samples
    if samples.shape[0] > length:
        return samples[:length]
    pad = np.zeros(length - samples.shape[0], dtype=np.float32)
    return np.concatenate([samples, pad])


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert a duration in seconds to a whole number of samples."""
    return round(seconds * sample_rate)


def frames_to_seconds(num_frames: int, hop_length: int, sample_rate: int) -> float:
    """Convert a frame count to seconds given the hop size."""
    return num_frames * hop_length / float(sample_rate)
