from __future__ import annotations

import numpy as np

from chunkasr.utils import (
    ensure_float32,
    frames_to_seconds,
    next_power_of_two,
    pad_or_trim,
    to_mono,
)


def test_next_power_of_two() -> None:
    assert next_power_of_two(1) == 1
    assert next_power_of_two(5) == 8
    assert next_power_of_two(16) == 16
    assert next_power_of_two(17) == 32


def test_ensure_float32() -> None:
    out = ensure_float32([1, 2, 3])
    assert out.dtype == np.float32
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_to_mono_stereo() -> None:
    stereo = np.array([[1.0, 3.0], [1.0, 3.0]], dtype=np.float32)  # (2, 2)
    mono = to_mono(stereo)
    assert mono.shape == (2,)


def test_pad_or_trim() -> None:
    x = np.arange(4, dtype=np.float32)
    assert pad_or_trim(x, 2).tolist() == [0.0, 1.0]
    assert pad_or_trim(x, 6).tolist() == [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
    assert pad_or_trim(x, 4).tolist() == [0.0, 1.0, 2.0, 3.0]


def test_frames_to_seconds() -> None:
    assert frames_to_seconds(100, 160, 16000) == 1.0
