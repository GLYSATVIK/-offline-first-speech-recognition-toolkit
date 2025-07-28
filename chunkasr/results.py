"""Result containers returned by the recognition pipeline.

These are plain dataclasses so they serialise cleanly to JSON and stay easy to
assert against in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenSpan:
    """A single decoded token together with its frame span.

    Frame indices are inclusive of ``start_frame`` and exclusive of
    ``end_frame`` following the usual half-open convention.
    """

    token: str
    start_frame: int
    end_frame: int
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "score": self.score,
        }


@dataclass(frozen=True)
class WordTiming:
    """A word with start/end times in seconds."""

    word: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {"word": self.word, "start": self.start, "end": self.end}


@dataclass
class Transcript:
    """Full transcription result for one utterance."""

    text: str
    tokens: list[TokenSpan] = field(default_factory=list)
    words: list[WordTiming] = field(default_factory=list)
    audio_seconds: float = 0.0
    compute_seconds: float = 0.0

    @property
    def rtf(self) -> float:
        """Real-time factor: compute time divided by audio duration."""
        if self.audio_seconds <= 0.0:
            return float("nan")
        return self.compute_seconds / self.audio_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens],
            "words": [w.to_dict() for w in self.words],
            "audio_seconds": self.audio_seconds,
            "compute_seconds": self.compute_seconds,
            "rtf": self.rtf,
        }
