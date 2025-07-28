"""Checkpoint import/export.

A checkpoint is a single ``.npz`` archive holding the model weights plus a JSON
metadata blob (format version, toolkit version, model dimensions, and the
experiment config). No pickling, so checkpoints are safe to share.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .config import ExperimentConfig
from .exceptions import CheckpointError
from .model.linear import LinearAcousticModel

FORMAT_VERSION = 1


@dataclass
class Checkpoint:
    """A loaded checkpoint: the model plus its metadata."""

    model: LinearAcousticModel
    config: ExperimentConfig | None
    toolkit_version: str
    format_version: int


def save_checkpoint(
    path: str | Path,
    model: LinearAcousticModel,
    config: ExperimentConfig | None = None,
) -> Path:
    """Export ``model`` (and optionally ``config``) to a ``.npz`` checkpoint."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "toolkit_version": __version__,
        "model_kind": "linear",
        "input_dim": model.input_dim,
        "vocab_size": model.vocab_size,
    }
    if config is not None:
        meta["config"] = config.to_dict()
    np.savez(
        path,
        weight=model.weight,
        bias=model.bias,
        meta=np.array(json.dumps(meta)),
    )
    return path


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Import a checkpoint previously written by :func:`save_checkpoint`."""
    path = Path(path)
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"could not read checkpoint {path}: {exc}") from exc
    if "weight" not in data or "meta" not in data:
        raise CheckpointError(f"checkpoint {path} is missing required arrays")
    meta = json.loads(str(data["meta"]))
    if meta.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(f"unsupported checkpoint format {meta.get('format_version')!r}")
    weight = np.asarray(data["weight"], dtype=np.float32)
    bias = np.asarray(data["bias"], dtype=np.float32)
    model = LinearAcousticModel(weight, bias)
    config = None
    if "config" in meta:
        config = ExperimentConfig.from_dict(meta["config"])
    return Checkpoint(
        model=model,
        config=config,
        toolkit_version=str(meta.get("toolkit_version", "unknown")),
        format_version=int(meta["format_version"]),
    )
