"""Sample-rate conversion.

Uses SciPy's polyphase resampler for the rational case (the common one, e.g.
44.1 kHz -> 16 kHz) and falls back to linear interpolation only when SciPy is
unavailable, so the dependency stays soft.
"""

from __future__ import annotations

from math import gcd

import numpy as np

from ..types import FloatArray


def resample(samples: FloatArray, orig_sr: int, target_sr: int) -> FloatArray:
    """Resample ``samples`` from ``orig_sr`` to ``target_sr`` hertz."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")
    if orig_sr == target_sr:
        return samples.astype(np.float32, copy=False)
    divisor = gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    try:
        from scipy.signal import resample_poly
    except ImportError:  # pragma: no cover - SciPy is a hard dep in practice
        return _linear_resample(samples, orig_sr, target_sr)
    out = resample_poly(samples.astype(np.float64), up, down)
    return np.asarray(out, dtype=np.float32)


def _linear_resample(samples: FloatArray, orig_sr: int, target_sr: int) -> FloatArray:
    """Naive linear-interpolation resampler used as a SciPy-free fallback."""
    duration = samples.shape[0] / float(orig_sr)
    n_out = round(duration * target_sr)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.arange(samples.shape[0], dtype=np.float64)
    dst_idx = np.linspace(0.0, samples.shape[0] - 1, n_out)
    out = np.interp(dst_idx, src_idx, samples.astype(np.float64))
    return out.astype(np.float32, copy=False)
