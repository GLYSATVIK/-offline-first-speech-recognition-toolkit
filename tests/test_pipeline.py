from __future__ import annotations

import pytest

from chunkasr.config import DecodeConfig
from chunkasr.decode.lm import NGramLM
from chunkasr.pipeline import Recognizer
from chunkasr.results import Transcript


def test_transcribe_returns_transcript(tone) -> None:
    samples, sr = tone
    result = Recognizer.demo().transcribe(samples, sr)
    assert isinstance(result, Transcript)
    assert isinstance(result.text, str)
    assert result.audio_seconds == pytest.approx(1.0)
    assert result.compute_seconds >= 0.0


@pytest.mark.parametrize("method", ["ctc_greedy", "ctc_beam", "rnnt_greedy"])
def test_decode_methods_run(tone, method) -> None:
    samples, sr = tone
    rec = Recognizer.demo(decode=DecodeConfig(method=method, beam_size=4))
    result = rec.transcribe(samples[:4800], sr)
    assert isinstance(result.text, str)


def test_streaming_matches_length(tone) -> None:
    samples, sr = tone
    offline = Recognizer.demo().transcribe(samples[:4800], sr)
    streamed = Recognizer.demo().enable_streaming(8, 4).transcribe(samples[:4800], sr)
    assert isinstance(streamed.text, str)
    assert offline.audio_seconds == streamed.audio_seconds


def test_with_lm_switches_to_beam(tone) -> None:
    samples, sr = tone
    lm = NGramLM.train_chars(["hello world", "the cat"], order=3)
    rec = Recognizer.demo().with_lm(lm, weight=0.3)
    assert rec.decode.method == "ctc_beam"
    result = rec.transcribe(samples[:3200], sr)
    assert isinstance(result.text, str)


def test_resamples_when_needed(tone) -> None:
    samples, _ = tone
    # Feed 8 kHz audio to a 16 kHz recogniser; it should resample internally.
    result = Recognizer.demo().transcribe(samples[:4000], sample_rate=8000)
    assert result.audio_seconds == pytest.approx(0.5)
