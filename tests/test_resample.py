from __future__ import annotations

import numpy as np

from chunkasr.audio.resample import resample


def test_resample_downsample_length(tone) -> None:
    samples, sr = tone
    out = resample(samples, sr, 8000)
    assert abs(out.shape[0] - samples.shape[0] // 2) <= 2
    assert out.dtype == np.float32


def test_resample_identity(tone) -> None:
    samples, sr = tone
    out = resample(samples, sr, sr)
    assert np.shares_memory(out, samples) or np.allclose(out, samples)


def test_resample_upsample_length(tone) -> None:
    samples, sr = tone
    out = resample(samples[:1000], sr, 32000)
    assert abs(out.shape[0] - 2000) <= 4
