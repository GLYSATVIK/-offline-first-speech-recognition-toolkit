"""Short-time Fourier transform and power spectrogram.

Thin wrappers over :mod:`scipy.fft` (falling back to :mod:`numpy.fft`) that
apply framing and windowing consistently.
"""

from __future__ import annotations

import numpy as np

from ..types import FloatArray
from .framing import frame_signal
from .window import get_window


def _rfft(frames: FloatArray, n_fft: int) -> np.ndarray:
    try:
        from scipy.fft import rfft
    except ImportError:  # pragma: no cover - SciPy is a hard dep in practice
        return np.fft.rfft(frames, n=n_fft, axis=1)
    return np.asarray(rfft(frames, n=n_fft, axis=1))


def stft(
    signal: FloatArray,
    frame_length: int,
    frame_shift: int,
    n_fft: int | None = None,
    window: str = "povey",
) -> np.ndarray:
    """Compute the complex STFT of ``signal``.

    Returns an array of shape ``(n_frames, n_fft // 2 + 1)``.
    """
    n_fft = n_fft or frame_length
    frames = frame_signal(signal, frame_length, frame_shift)
    if frames.shape[0] == 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.complex64)
    win = get_window(window, frame_length).astype(np.float64)
    windowed = frames.astype(np.float64) * win[None, :]
    return _rfft(windowed.astype(np.float32), n_fft)


def power_spectrogram(
    signal: FloatArray,
    frame_length: int,
    frame_shift: int,
    n_fft: int | None = None,
    window: str = "povey",
) -> FloatArray:
    """Compute the power spectrogram ``|STFT|**2``."""
    spec = stft(signal, frame_length, frame_shift, n_fft, window)
    power = np.abs(spec) ** 2
    return power.astype(np.float32, copy=False)
