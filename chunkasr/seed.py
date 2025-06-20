"""Deterministic seeding helpers.

Reproducibility is a first-class goal of the toolkit, so every stochastic
component (SpecAugment, beam tie-breaking, synthetic fixtures) draws from a
:class:`numpy.random.Generator` produced here rather than the global RNG.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 1234


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python's ``random`` and NumPy's legacy global RNG.

    This is a convenience for scripts and examples. Library code should prefer
    :func:`rng`, which returns an isolated generator instead of touching global
    state.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def rng(seed: int | None = DEFAULT_SEED) -> np.random.Generator:
    """Return an independent :class:`numpy.random.Generator`.

    Passing ``None`` yields a nondeterministic generator seeded from the OS
    entropy pool; passing an int gives a reproducible stream.
    """
    return np.random.default_rng(seed)
