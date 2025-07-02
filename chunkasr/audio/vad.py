"""Energy + spectral-flatness voice-activity detection.

The detector combines two cheap cues: short-time energy (speech is louder than
the noise floor) and spectral flatness (speech is less flat than stationary
noise). A frame is voiced when it is both loud enough and tonal enough, after
which min-duration and padding rules smooth the raw decisions into segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import VADConfig
from ..frontend.framing import frame_signal
from ..types import BoolArray, FloatArray


@dataclass(frozen=True)
class SpeechSegment:
    """A contiguous voiced region, in samples and seconds."""

    start_sample: int
    end_sample: int
    sample_rate: int

    @property
    def start(self) -> float:
        return self.start_sample / float(self.sample_rate)

    @property
    def end(self) -> float:
        return self.end_sample / float(self.sample_rate)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _frame_energy_db(frames: FloatArray) -> FloatArray:
    """Per-frame energy in decibels, floored to avoid ``log(0)``."""
    energy = np.mean(frames.astype(np.float64) ** 2, axis=1)
    energy = np.maximum(energy, 1e-10)
    return (10.0 * np.log10(energy)).astype(np.float32)


def _spectral_flatness(frames: FloatArray) -> FloatArray:
    """Per-frame spectral flatness (geometric mean / arithmetic mean)."""
    spectrum = np.abs(np.fft.rfft(frames.astype(np.float64), axis=1)) ** 2
    spectrum = np.maximum(spectrum, 1e-10)
    geo = np.exp(np.mean(np.log(spectrum), axis=1))
    arith = np.mean(spectrum, axis=1)
    return (geo / arith).astype(np.float32)


class EnergyVAD:
    """Stateful, streaming-friendly voice-activity detector."""

    def __init__(self, config: VADConfig | None = None, sample_rate: int = 16000) -> None:
        self.config = config or VADConfig()
        self.sample_rate = sample_rate
        self._frame_length = round(0.025 * sample_rate)
        self._frame_shift = round(0.010 * sample_rate)

    def frame_decisions(self, samples: FloatArray) -> BoolArray:
        """Return a per-frame boolean voiced/unvoiced mask."""
        frames = frame_signal(samples, self._frame_length, self._frame_shift)
        if frames.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        energy_db = _frame_energy_db(frames)
        flatness = _spectral_flatness(frames)
        voiced = (energy_db > self.config.energy_threshold_db) & (
            flatness < self.config.flatness_threshold
        )
        return np.asarray(voiced, dtype=bool)

    def segments(self, samples: FloatArray) -> list[SpeechSegment]:
        """Detect voiced segments in ``samples``."""
        mask = self.frame_decisions(samples)
        return self._mask_to_segments(mask)

    def _mask_to_segments(self, mask: BoolArray) -> list[SpeechSegment]:
        cfg = self.config
        min_speech = _ms_to_frames(cfg.min_speech_ms, self._frame_shift, self.sample_rate)
        min_sil = _ms_to_frames(cfg.min_silence_ms, self._frame_shift, self.sample_rate)
        pad = _ms_to_frames(cfg.speech_pad_ms, self._frame_shift, self.sample_rate)
        raw = _runs(mask)
        merged = _merge_short_gaps(raw, min_sil)
        segments: list[SpeechSegment] = []
        for start_f, end_f in merged:
            if end_f - start_f < min_speech:
                continue
            start = max(0, (start_f - pad)) * self._frame_shift
            end = (end_f + pad) * self._frame_shift + self._frame_length
            end = min(end, mask.shape[0] * self._frame_shift + self._frame_length)
            segments.append(SpeechSegment(int(start), int(end), self.sample_rate))
        return segments


def _ms_to_frames(ms: float, frame_shift: int, sample_rate: int) -> int:
    return round(ms / 1000.0 * sample_rate / frame_shift)


def _runs(mask: BoolArray) -> list[tuple[int, int]]:
    """Return ``(start, end)`` frame spans of consecutive ``True`` values."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, mask.shape[0]))
    return runs


def _merge_short_gaps(runs: list[tuple[int, int]], min_gap: int) -> list[tuple[int, int]]:
    """Merge adjacent voiced runs separated by less than ``min_gap`` frames."""
    if not runs:
        return runs
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < min_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def detect_speech(
    samples: FloatArray,
    sample_rate: int = 16000,
    config: VADConfig | None = None,
) -> list[SpeechSegment]:
    """Convenience wrapper around :class:`EnergyVAD`."""
    return EnergyVAD(config=config, sample_rate=sample_rate).segments(samples)
