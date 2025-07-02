"""Reading and writing audio files.

WAV support is built on the standard-library :mod:`wave` module so the core
toolkit has no hard dependency on ``libsndfile``. Other container formats are
handled through :mod:`soundfile` when it happens to be installed.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..exceptions import AudioError
from ..types import FloatArray
from ..utils import to_mono

_INT_SCALE = {1: 128.0, 2: 32768.0, 4: 2147483648.0}


def _decode_pcm(raw: bytes, sampwidth: int, n_channels: int) -> FloatArray:
    """Decode interleaved little-endian PCM bytes to float32 in [-1, 1]."""
    if sampwidth == 3:
        return _decode_pcm24(raw, n_channels)
    if sampwidth not in _INT_SCALE:
        raise AudioError(f"unsupported PCM sample width: {sampwidth} bytes")
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sampwidth]
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sampwidth == 1:  # 8-bit WAV is unsigned, centred at 128
        data = data - 128.0
    data /= _INT_SCALE[sampwidth]
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
    return data.astype(np.float32, copy=False)


def _decode_pcm24(raw: bytes, n_channels: int) -> FloatArray:
    """Decode 24-bit little-endian PCM, which NumPy has no native dtype for."""
    a = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
    a = a.reshape(-1, 3)
    values = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
    # sign-extend from 24 to 32 bits
    values = np.where(values & 0x800000, values - (1 << 24), values)
    data = values.astype(np.float32) / 8388608.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
    return data.astype(np.float32, copy=False)


def read_wav(path: str | Path) -> tuple[FloatArray, int]:
    """Read a PCM WAV file, returning ``(mono_samples, sample_rate)``.

    Multi-channel files are averaged to mono. Samples are float32 in [-1, 1].
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioError(f"could not read WAV file {path}: {exc}") from exc
    data = _decode_pcm(raw, sampwidth, n_channels)
    return to_mono(data), sample_rate


def write_wav(path: str | Path, samples: FloatArray, sample_rate: int) -> None:
    """Write float32 samples in [-1, 1] to a 16-bit PCM WAV file."""
    path = Path(path)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    try:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm.tobytes())
    except wave.Error as exc:
        raise AudioError(f"could not write WAV file {path}: {exc}") from exc


def load_audio(path: str | Path) -> tuple[FloatArray, int]:
    """Load audio from any supported container.

    ``.wav`` files use the standard library; everything else is delegated to
    :mod:`soundfile` if available.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return read_wav(path)
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise AudioError(
            f"reading {path.suffix} files requires the optional 'soundfile' package"
        ) from exc
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono(np.asarray(data, dtype=np.float32)), int(sample_rate)


def audio_duration(samples: FloatArray, sample_rate: int) -> float:
    """Return the duration of ``samples`` in seconds."""
    if sample_rate <= 0:
        raise AudioError("sample_rate must be positive")
    return samples.shape[0] / float(sample_rate)
