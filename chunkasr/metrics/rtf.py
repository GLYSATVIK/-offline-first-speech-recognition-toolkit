"""Real-time factor (RTF) measurement.

RTF is compute time divided by audio duration: below 1.0 means faster than
real time, which is the bar a streaming recogniser has to clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def real_time_factor(compute_seconds: float, audio_seconds: float) -> float:
    """Return ``compute_seconds / audio_seconds`` (NaN for empty audio)."""
    if audio_seconds <= 0.0:
        return float("nan")
    return compute_seconds / audio_seconds


@dataclass
class RTFMeter:
    """Accumulates compute/audio time across many utterances."""

    total_compute: float = 0.0
    total_audio: float = 0.0
    _samples: list[float] = field(default_factory=list)

    def update(self, compute_seconds: float, audio_seconds: float) -> float:
        """Record one utterance and return its instantaneous RTF."""
        self.total_compute += compute_seconds
        self.total_audio += audio_seconds
        rtf = real_time_factor(compute_seconds, audio_seconds)
        self._samples.append(rtf)
        return rtf

    @property
    def rtf(self) -> float:
        """Corpus-level RTF (total compute over total audio)."""
        return real_time_factor(self.total_compute, self.total_audio)

    @property
    def count(self) -> int:
        return len(self._samples)
