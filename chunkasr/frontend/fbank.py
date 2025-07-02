"""Log-Mel filterbank feature extraction.

Ties framing, windowing, the STFT and the Mel bank into the ``fbank`` features
that feed the acoustic model.
"""

from __future__ import annotations

import numpy as np

from ..config import FeatureConfig
from ..types import FloatArray
from .melbank import mel_filterbank
from .stft import power_spectrogram


def preemphasis(signal: FloatArray, coeff: float) -> FloatArray:
    """Apply a first-order high-pass pre-emphasis filter."""
    if coeff <= 0.0:
        return signal
    out = np.empty_like(signal)
    out[0] = signal[0]
    out[1:] = signal[1:] - coeff * signal[:-1]
    return out


def log_mel_spectrogram(signal: FloatArray, config: FeatureConfig) -> FloatArray:
    """Compute a ``(n_frames, num_mel_bins)`` log-Mel spectrogram."""
    return LogMelExtractor(config)(signal)


class LogMelExtractor:
    """Reusable log-Mel extractor that caches the filterbank matrix."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._fbank = mel_filterbank(
            self.config.num_mel_bins,
            self.config.fft_length,
            self.config.sample_rate,
            self.config.low_freq,
            self.config.high_freq,
        )

    @property
    def output_dim(self) -> int:
        return self.config.num_mel_bins

    def __call__(self, signal: FloatArray) -> FloatArray:
        cfg = self.config
        sig = signal.astype(np.float32, copy=True)
        if cfg.dither > 0.0:
            noise = np.asarray(
                np.random.default_rng(0).standard_normal(sig.shape), dtype=np.float32
            )
            sig = (sig + cfg.dither * noise).astype(np.float32, copy=False)
        sig = preemphasis(sig, cfg.preemphasis)
        power = power_spectrogram(
            sig, cfg.frame_length, cfg.frame_shift, cfg.fft_length, cfg.window
        )
        if power.shape[0] == 0:
            return np.zeros((0, cfg.num_mel_bins), dtype=np.float32)
        mel = power @ self._fbank.T
        if cfg.use_log:
            mel = np.log(np.maximum(mel, 1e-10))
        return mel.astype(np.float32, copy=False)
