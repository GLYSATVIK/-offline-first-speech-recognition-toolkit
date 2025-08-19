from __future__ import annotations

import numpy as np

from chunkasr.model.backend import log_add, log_softmax, logsumexp, softmax, torch_available


def test_log_softmax_normalised() -> None:
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    lp = log_softmax(x)
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-5)


def test_softmax_normalised() -> None:
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert np.allclose(softmax(x).sum(axis=1), 1.0)


def test_logsumexp_matches_reference() -> None:
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert np.isclose(float(logsumexp(x)), np.log(3.0), atol=1e-5)


def test_log_add() -> None:
    assert np.isclose(log_add(-1.0, -2.0), float(np.logaddexp(-1.0, -2.0)))
    assert log_add(float("-inf"), -3.0) == -3.0
    assert log_add(-3.0, float("-inf")) == -3.0


def test_torch_available_is_bool() -> None:
    assert isinstance(torch_available(), bool)
