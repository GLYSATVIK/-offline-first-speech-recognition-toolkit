"""Numeric kernels shared by the models and decoders.

Everything defaults to NumPy. An optional PyTorch path exists for the linear
model's forward pass; it is selected explicitly and never imported eagerly, so
the base install stays dependency-light.
"""

from __future__ import annotations

import numpy as np

from ..types import FloatArray


def log_softmax(x: FloatArray, axis: int = -1) -> FloatArray:
    """Numerically stable log-softmax."""
    x64 = x.astype(np.float64)
    shifted = x64 - np.max(x64, axis=axis, keepdims=True)
    log_denom = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return (shifted - log_denom).astype(np.float32)


def softmax(x: FloatArray, axis: int = -1) -> FloatArray:
    """Numerically stable softmax."""
    x64 = x.astype(np.float64)
    shifted = np.exp(x64 - np.max(x64, axis=axis, keepdims=True))
    return (shifted / np.sum(shifted, axis=axis, keepdims=True)).astype(np.float32)


def logsumexp(values: FloatArray, axis: int | None = None) -> FloatArray:
    """Stable log-sum-exp reduction returning a float32 array."""
    arr = values.astype(np.float64)
    m = np.max(arr, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=True))
    out = out.reshape(()) if axis is None else np.squeeze(out, axis=axis)
    return out.astype(np.float32)


def log_add(a: float, b: float) -> float:
    """Add two values in the log domain: ``log(exp(a) + exp(b))``."""
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + float(np.log1p(np.exp(lo - hi)))


def torch_available() -> bool:
    """Return ``True`` if the optional PyTorch backend can be imported."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True
