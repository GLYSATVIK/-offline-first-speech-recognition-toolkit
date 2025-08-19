from __future__ import annotations

from pathlib import Path

import pytest

from chunkasr.config import ExperimentConfig, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_config_loads(path: Path) -> None:
    cfg = load_config(path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name
    assert cfg.feature.num_mel_bins > 0
