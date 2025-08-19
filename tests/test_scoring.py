from __future__ import annotations

import numpy as np
import pytest

from chunkasr.metrics.cer import cer, corpus_cer
from chunkasr.metrics.rtf import RTFMeter, real_time_factor
from chunkasr.metrics.wer import corpus_wer, wer


def test_wer_single() -> None:
    result = wer("the cat sat", "the bat sat")
    assert result.wer == pytest.approx(1 / 3)
    assert result.substitutions == 1
    assert result.as_percent() == pytest.approx(100.0 / 3.0)


def test_corpus_wer() -> None:
    result = corpus_wer(["a b c", "d e"], ["a b c", "d x"])
    # 1 error over 5 reference words.
    assert np.isclose(result.wer, 0.2)


def test_cer() -> None:
    assert np.isclose(cer("hello", "hallo"), 0.2)


def test_corpus_cer() -> None:
    assert corpus_cer(["ab", "cd"], ["ab", "cd"]) == 0.0


def test_rtf() -> None:
    assert real_time_factor(0.5, 1.0) == 0.5
    assert np.isnan(real_time_factor(0.5, 0.0))


def test_rtf_meter() -> None:
    meter = RTFMeter()
    meter.update(0.5, 1.0)
    meter.update(1.5, 1.0)
    assert meter.count == 2
    assert np.isclose(meter.rtf, 1.0)
