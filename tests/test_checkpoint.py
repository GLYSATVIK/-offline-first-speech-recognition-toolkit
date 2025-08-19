from __future__ import annotations

import numpy as np
import pytest

from chunkasr.checkpoint import load_checkpoint, save_checkpoint
from chunkasr.config import ExperimentConfig
from chunkasr.exceptions import CheckpointError
from chunkasr.model.linear import LinearAcousticModel


def test_save_and_load(tmp_path) -> None:
    model = LinearAcousticModel.random(8, 5, seed=3)
    cfg = ExperimentConfig(name="ckpt-test")
    path = save_checkpoint(tmp_path / "model", model, cfg)
    assert path.suffix == ".npz"

    ckpt = load_checkpoint(path)
    assert np.array_equal(ckpt.model.weight, model.weight)
    assert np.array_equal(ckpt.model.bias, model.bias)
    assert ckpt.config == cfg


def test_load_without_config(tmp_path) -> None:
    model = LinearAcousticModel.random(4, 4, seed=1)
    path = save_checkpoint(tmp_path / "m.npz", model)
    ckpt = load_checkpoint(path)
    assert ckpt.config is None
    assert ckpt.model.input_dim == 4


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "nope.npz")
