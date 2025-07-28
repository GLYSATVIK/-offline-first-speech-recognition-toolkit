"""Typed, serialisable configuration objects.

Every stage of the pipeline is driven by a dataclass here. Configs round-trip
through JSON (always) and YAML (when :mod:`yaml` is installed), which is what
makes experiments reproducible: the exact settings live next to the results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .exceptions import ConfigError


@dataclass
class FeatureConfig:
    """Settings for the log-Mel / filterbank frontend."""

    sample_rate: int = 16000
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    num_mel_bins: int = 80
    fft_length: int = 512
    low_freq: float = 20.0
    high_freq: float = 0.0  # 0 means Nyquist
    preemphasis: float = 0.97
    window: str = "povey"
    dither: float = 0.0
    use_log: bool = True

    @property
    def frame_length(self) -> int:
        return round(self.sample_rate * self.frame_length_ms / 1000.0)

    @property
    def frame_shift(self) -> int:
        return round(self.sample_rate * self.frame_shift_ms / 1000.0)


@dataclass
class VADConfig:
    """Energy + spectral voice-activity detection thresholds."""

    energy_threshold_db: float = -36.0
    flatness_threshold: float = 0.45
    min_speech_ms: float = 120.0
    min_silence_ms: float = 180.0
    speech_pad_ms: float = 80.0


@dataclass
class DecodeConfig:
    """Decoder selection and beam-search hyper-parameters."""

    method: str = "ctc_greedy"  # ctc_greedy | ctc_beam | rnnt_greedy
    beam_size: int = 8
    blank_id: int = 0
    lm_weight: float = 0.0
    word_insertion_penalty: float = 0.0
    prune_threshold: float = -12.0


@dataclass
class ModelConfig:
    """Shape/description of the acoustic model."""

    kind: str = "linear"  # linear | transducer
    input_dim: int = 80
    vocab_size: int = 32
    chunk_size: int = 16
    right_context: int = 4


@dataclass
class ExperimentConfig:
    """Top-level config bundling every stage.

    This is the object that gets written next to checkpoints and results.
    """

    name: str = "default"
    seed: int = 1234
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        return _build_dataclass(cls, data)


_NESTED = {
    "feature": FeatureConfig,
    "vad": VADConfig,
    "decode": DecodeConfig,
    "model": ModelConfig,
}


def _build_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    """Recursively construct a config dataclass, rejecting unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in _NESTED and isinstance(value, dict):
            kwargs[key] = _NESTED[key](**value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a JSON or YAML file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    loader = _load_yaml if path.suffix.lower() in {".yaml", ".yml"} else json.loads
    data = loader(text)
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")
    return ExperimentConfig.from_dict(data)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Write ``config`` to disk as JSON or YAML based on the extension."""
    path = Path(path)
    data = config.to_dict()
    if path.suffix.lower() in {".yaml", ".yml"}:
        text = _dump_yaml(data)
    else:
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ConfigError("PyYAML is required to read YAML configs") from exc
    return yaml.safe_load(text)


def _dump_yaml(data: dict[str, Any]) -> str:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ConfigError("PyYAML is required to write YAML configs") from exc
    return str(yaml.safe_dump(data, sort_keys=True, default_flow_style=False))
