from __future__ import annotations

import numpy as np

from chunkasr.frontend.melbank import hz_to_mel, mel_filterbank, mel_to_hz
from chunkasr.frontend.stft import power_spectrogram, stft


def test_stft_shape(tone) -> None:
    samples, _ = tone
    spec = stft(samples[:4000], 400, 160, n_fft=512)
    assert spec.shape[1] == 512 // 2 + 1
    assert np.iscomplexobj(spec)


def test_power_spectrogram_nonneg(tone) -> None:
    samples, _ = tone
    power = power_spectrogram(samples[:4000], 400, 160, n_fft=512)
    assert np.all(power >= 0.0)


def test_mel_scale_roundtrip() -> None:
    hz = np.array([100.0, 1000.0, 4000.0], dtype=np.float32)
    back = mel_to_hz(hz_to_mel(hz))
    assert np.allclose(back, hz, rtol=1e-3)


def test_mel_scale_monotonic() -> None:
    mels = hz_to_mel(np.array([100.0, 500.0, 2000.0], dtype=np.float32))
    assert mels[0] < mels[1] < mels[2]


def test_mel_filterbank() -> None:
    fb = mel_filterbank(num_bins=40, n_fft=512, sample_rate=16000)
    assert fb.shape == (40, 257)
    assert np.all(fb >= 0.0)
    # Every filter has some support.
    assert np.all(fb.sum(axis=1) > 0.0)
