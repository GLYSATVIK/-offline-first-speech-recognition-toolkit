"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from chunkasr.seed import rng


@pytest.fixture
def tone() -> tuple[np.ndarray, int]:
    """A 1-second 220 Hz tone at 16 kHz."""
    sr = 16000
    t = np.arange(sr) / sr
    x = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    return x.astype(np.float32), sr


@pytest.fixture
def speechlike() -> tuple[np.ndarray, int]:
    """Silence, a voiced tone burst, then silence — a VAD smoke signal."""
    sr = 16000
    gen = rng(0)
    silence = np.zeros(int(0.3 * sr), dtype=np.float32)
    n = int(0.5 * sr)
    voiced = 0.3 * np.sin(2.0 * np.pi * 180.0 * np.arange(n) / sr)
    voiced = voiced.astype(np.float32) + 0.01 * gen.standard_normal(n).astype(np.float32)
    return np.concatenate([silence, voiced, silence]), sr


@pytest.fixture
def corpus() -> list[str]:
    return ["hello world", "hello there", "the cat sat", "the dog ran on the mat"]
