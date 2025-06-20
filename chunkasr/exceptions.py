"""Exception hierarchy for :mod:`chunkasr`.

All library-specific errors derive from :class:`ChunkASRError` so callers can
catch everything the toolkit raises with a single ``except`` clause.
"""

from __future__ import annotations


class ChunkASRError(Exception):
    """Base class for every error raised by :mod:`chunkasr`."""


class AudioError(ChunkASRError):
    """Raised when audio cannot be read, written, or decoded."""


class ConfigError(ChunkASRError):
    """Raised when a configuration value is missing or invalid."""


class ManifestError(ChunkASRError):
    """Raised when a dataset manifest is malformed."""


class CheckpointError(ChunkASRError):
    """Raised when a checkpoint cannot be loaded or is incompatible."""


class DecodeError(ChunkASRError):
    """Raised when a decoding routine is given inconsistent inputs."""


class BackendError(ChunkASRError):
    """Raised when an optional compute backend is unavailable or misused."""
