from __future__ import annotations

import pytest

from chunkasr.data.manifest import (
    ManifestEntry,
    filter_by_duration,
    manifest_stats,
    read_manifest,
    write_manifest,
)
from chunkasr.exceptions import ManifestError


def _entries() -> list[ManifestEntry]:
    return [
        ManifestEntry("a.wav", 1.5, "hello", utt_id="a"),
        ManifestEntry("b.wav", 3.0, "world"),
        ManifestEntry("c.wav", 10.0, "long one"),
    ]


def test_write_read_roundtrip(tmp_path) -> None:
    path = tmp_path / "m.jsonl"
    written = write_manifest(path, _entries())
    assert written == 3
    loaded = read_manifest(path)
    assert loaded[0].utt_id == "a"
    assert loaded[1].duration == 3.0


def test_filter_by_duration() -> None:
    kept = filter_by_duration(_entries(), min_seconds=2.0, max_seconds=5.0)
    assert [e.audio_filepath for e in kept] == ["b.wav"]


def test_stats() -> None:
    stats = manifest_stats(_entries())
    assert stats["num_utterances"] == 3.0
    assert stats["total_seconds"] == 14.5


def test_missing_key_raises() -> None:
    with pytest.raises(ManifestError):
        ManifestEntry.from_dict({"audio_filepath": "x.wav", "duration": 1.0})


def test_bad_json_raises(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        read_manifest(path)
