"""Shared type aliases used across the package.

Keeping these in one place makes the numeric contracts explicit: almost every
kernel operates on ``float32`` arrays, and mixing dtypes silently is a common
source of subtle bugs in DSP code.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: 1-D or 2-D single-precision float array (audio samples, features, logits).
FloatArray = npt.NDArray[np.float32]

#: Double-precision float array, used where accumulation precision matters.
Float64Array = npt.NDArray[np.float64]

#: Integer array (token ids, frame indices).
IntArray = npt.NDArray[np.int64]

#: Boolean mask array (voice-activity decisions, padding masks).
BoolArray = npt.NDArray[np.bool_]

__all__ = ["BoolArray", "Float64Array", "FloatArray", "IntArray"]
