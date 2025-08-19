from __future__ import annotations

import numpy as np
import pytest

from chunkasr.audio.io import audio_duration, read_wav, write_wav
from chunkasr.exceptions import AudioError


def test_wav_roundtrip(tmp_path, tone) -> None:
    samples, sr = tone
    path = tmp_path / "tone.wav"
    write_wav(path, samples, sr)
    loaded, loaded_sr = read_wav(path)
    assert loaded_sr == sr
    assert loaded.dtype == np.float32
    assert loaded.shape == samples.shape
    # 16-bit quantisation error is bounded.
    assert np.max(np.abs(loaded - samples)) < 1e-3


def test_audio_duration(tone) -> None:
    samples, sr = tone
    assert audio_duration(samples, sr) == pytest.approx(1.0)


def test_read_missing_file_raises(tmp_path) -> None:
    with pytest.raises(AudioError):
        read_wav(tmp_path / "does-not-exist.wav")
