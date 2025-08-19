from __future__ import annotations

import numpy as np
import pytest

from chunkasr.model.linear import LinearAcousticModel


def test_random_model_shapes() -> None:
    model = LinearAcousticModel.random(input_dim=8, vocab_size=5, seed=0)
    assert model.input_dim == 8
    assert model.vocab_size == 5


def test_forward_is_log_prob() -> None:
    model = LinearAcousticModel.random(8, 5, seed=0)
    feats = np.random.default_rng(1).standard_normal((10, 8)).astype(np.float32)
    log_probs = model.forward(feats)
    assert log_probs.shape == (10, 5)
    assert np.allclose(np.exp(log_probs).sum(axis=1), 1.0, atol=1e-5)


def test_forward_dim_mismatch() -> None:
    model = LinearAcousticModel.random(8, 5, seed=0)
    with pytest.raises(ValueError, match="input dim"):
        model.forward(np.zeros((3, 7), dtype=np.float32))


def test_reproducible_weights() -> None:
    a = LinearAcousticModel.random(4, 4, seed=42)
    b = LinearAcousticModel.random(4, 4, seed=42)
    assert np.array_equal(a.weight, b.weight)


def test_bias_shape_validation() -> None:
    with pytest.raises(ValueError, match="bias shape"):
        LinearAcousticModel(np.zeros((4, 3), dtype=np.float32), np.zeros(2, dtype=np.float32))
