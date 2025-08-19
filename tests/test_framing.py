from __future__ import annotations

import numpy as np

from chunkasr.frontend.framing import frame_signal, num_frames
from chunkasr.frontend.window import get_window


def test_num_frames() -> None:
    assert num_frames(1000, 400, 160) == 4
    assert num_frames(399, 400, 160) == 0
    assert num_frames(400, 400, 160) == 1


def test_frame_signal_shape() -> None:
    sig = np.arange(1000, dtype=np.float32)
    frames = frame_signal(sig, 400, 160)
    assert frames.shape == (4, 400)
    # Second frame starts one hop in.
    assert frames[1, 0] == 160.0


def test_frame_signal_too_short() -> None:
    frames = frame_signal(np.zeros(10, dtype=np.float32), 400, 160)
    assert frames.shape == (0, 400)


def test_windows() -> None:
    for name in ("hann", "hamming", "povey", "rectangular"):
        win = get_window(name, 400)
        assert win.shape == (400,)
        assert np.all(win >= 0.0)
    assert get_window("hann", 400)[0] < 1e-6
    assert np.allclose(get_window("rectangular", 8), 1.0)
