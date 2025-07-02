"""Analysis windows.

The ``povey`` window is the Kaldi default and is what most published
filterbank recipes assume, so it is the toolkit default too.
"""

from __future__ import annotations

import numpy as np

from ..types import FloatArray


def get_window(name: str, length: int) -> FloatArray:
    """Return an analysis window of ``length`` samples.

    Supported: ``hann``, ``hamming``, ``povey``, ``rectangular``.
    """
    if length <= 0:
        raise ValueError("window length must be positive")
    n = np.arange(length, dtype=np.float64)
    if name == "rectangular":
        win = np.ones(length, dtype=np.float64)
    elif name == "hann":
        win = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))
    elif name == "hamming":
        win = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (length - 1))
    elif name == "povey":
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))
        win = hann**0.85
    else:
        raise ValueError(f"unknown window: {name!r}")
    return win.astype(np.float32)
