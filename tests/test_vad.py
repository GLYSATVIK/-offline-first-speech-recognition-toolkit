from __future__ import annotations

from chunkasr.audio.vad import EnergyVAD, detect_speech


def test_detects_voiced_region(speechlike) -> None:
    samples, sr = speechlike
    segments = detect_speech(samples, sr)
    assert len(segments) >= 1
    total = sum(seg.duration for seg in segments)
    # The voiced burst is ~0.5s; padding widens it a little.
    assert 0.3 < total < 1.0


def test_silence_has_no_speech() -> None:
    import numpy as np

    silence = np.zeros(16000, dtype=np.float32)
    assert detect_speech(silence, 16000) == []


def test_frame_decisions_shape(speechlike) -> None:
    samples, sr = speechlike
    vad = EnergyVAD(sample_rate=sr)
    mask = vad.frame_decisions(samples)
    assert mask.dtype == bool
    assert mask.any()
