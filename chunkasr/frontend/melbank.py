"""Mel-scale filterbank construction."""

from __future__ import annotations

import numpy as np

from ..types import FloatArray


def hz_to_mel(hz: FloatArray | float) -> FloatArray:
    """Convert frequency in hertz to the (Slaney/HTK-style) Mel scale."""
    return np.asarray(2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)).astype(
        np.float32
    )


def mel_to_hz(mel: FloatArray | float) -> FloatArray:
    """Inverse of :func:`hz_to_mel`."""
    return np.asarray(700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)).astype(
        np.float32
    )


def mel_filterbank(
    num_bins: int,
    n_fft: int,
    sample_rate: int,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
) -> FloatArray:
    """Build a triangular Mel filterbank matrix.

    Returns an array of shape ``(num_bins, n_fft // 2 + 1)`` that maps a power
    spectrum onto ``num_bins`` Mel channels.
    """
    if high_freq <= 0.0:
        high_freq = sample_rate / 2.0
    if not 0.0 <= low_freq < high_freq <= sample_rate / 2.0:
        raise ValueError("require 0 <= low_freq < high_freq <= nyquist")

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float64)
    mel_low = float(hz_to_mel(low_freq))
    mel_high = float(hz_to_mel(high_freq))
    mel_points = np.linspace(mel_low, mel_high, num_bins + 2)
    hz_points = mel_to_hz(mel_points).astype(np.float64)

    fbank = np.zeros((num_bins, n_freqs), dtype=np.float64)
    for m in range(num_bins):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        rising = (fft_freqs - left) / max(center - left, 1e-9)
        falling = (right - fft_freqs) / max(right - center, 1e-9)
        fbank[m] = np.maximum(0.0, np.minimum(rising, falling))
    return fbank.astype(np.float32)
