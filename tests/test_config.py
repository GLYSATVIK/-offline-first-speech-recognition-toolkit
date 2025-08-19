from __future__ import annotations

import pytest

from chunkasr.config import (
    DecodeConfig,
    ExperimentConfig,
    FeatureConfig,
    load_config,
    save_config,
)
from chunkasr.exceptions import ConfigError


def test_defaults() -> None:
    cfg = ExperimentConfig()
    assert cfg.feature.sample_rate == 16000
    assert cfg.decode.method == "ctc_greedy"


def test_frame_length_derivation() -> None:
    feat = FeatureConfig(sample_rate=16000, frame_length_ms=25.0, frame_shift_ms=10.0)
    assert feat.frame_length == 400
    assert feat.frame_shift == 160


def test_dict_roundtrip() -> None:
    cfg = ExperimentConfig(name="exp", decode=DecodeConfig(method="ctc_beam", beam_size=16))
    restored = ExperimentConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_json_roundtrip(tmp_path) -> None:
    cfg = ExperimentConfig(name="json-exp", seed=7)
    path = tmp_path / "config.json"
    save_config(cfg, path)
    assert load_config(path) == cfg


def test_yaml_roundtrip(tmp_path) -> None:
    cfg = ExperimentConfig(name="yaml-exp")
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    assert load_config(path) == cfg


def test_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError):
        ExperimentConfig.from_dict({"not_a_field": 1})
