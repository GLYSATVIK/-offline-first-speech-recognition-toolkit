"""JSONL dataset manifests.

Each line is a JSON object with at least ``audio_filepath``, ``duration`` and
``text`` — the same shape NeMo/ESPnet-style recipes use, so manifests are easy
to interoperate with.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..exceptions import ManifestError

_REQUIRED = ("audio_filepath", "duration", "text")


@dataclass
class ManifestEntry:
    """One utterance in a dataset manifest."""

    audio_filepath: str
    duration: float
    text: str
    utt_id: str = ""

    def to_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "audio_filepath": self.audio_filepath,
            "duration": self.duration,
            "text": self.text,
        }
        if self.utt_id:
            record["utt_id"] = self.utt_id
        return record

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> ManifestEntry:
        missing = [key for key in _REQUIRED if key not in record]
        if missing:
            raise ManifestError(f"manifest entry missing keys: {missing}")
        return cls(
            audio_filepath=str(record["audio_filepath"]),
            duration=float(record["duration"]),  # type: ignore[arg-type]
            text=str(record["text"]),
            utt_id=str(record.get("utt_id", "")),
        )


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    """Read a JSONL manifest, raising :class:`ManifestError` on bad lines."""
    entries: list[ManifestEntry] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"invalid JSON on line {lineno}: {exc}") from exc
            entries.append(ManifestEntry.from_dict(record))
    return entries


def write_manifest(path: str | Path, entries: Iterable[ManifestEntry]) -> int:
    """Write entries as JSONL; returns the number of records written."""
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def filter_by_duration(
    entries: Iterable[ManifestEntry],
    min_seconds: float = 0.0,
    max_seconds: float = float("inf"),
) -> list[ManifestEntry]:
    """Keep entries whose duration falls within ``[min_seconds, max_seconds]``."""
    return [e for e in entries if min_seconds <= e.duration <= max_seconds]


def manifest_stats(entries: Iterable[ManifestEntry]) -> dict[str, float]:
    """Return simple corpus statistics (count, total/mean duration, chars)."""
    durations = []
    total_chars = 0
    for entry in entries:
        durations.append(entry.duration)
        total_chars += len(entry.text)
    n = len(durations)
    total = float(sum(durations))
    return {
        "num_utterances": float(n),
        "total_hours": total / 3600.0,
        "total_seconds": total,
        "mean_seconds": total / n if n else 0.0,
        "total_chars": float(total_chars),
    }
