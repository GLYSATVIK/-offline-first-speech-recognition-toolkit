"""Build a JSONL dataset manifest and print corpus statistics.

Run: uv run python examples/build_manifest.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from chunkasr.data.manifest import ManifestEntry, manifest_stats, read_manifest, write_manifest

ENTRIES = [
    ManifestEntry("audio/utt1.wav", 2.3, "hello world", utt_id="utt1"),
    ManifestEntry("audio/utt2.wav", 4.1, "the quick brown fox", utt_id="utt2"),
    ManifestEntry("audio/utt3.wav", 1.2, "yes", utt_id="utt3"),
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.jsonl"
        write_manifest(path, ENTRIES)
        entries = read_manifest(path)
        print(f"wrote and re-read {len(entries)} entries")
        for key, value in manifest_stats(entries).items():
            print(f"  {key:16s}: {value:.3f}")


if __name__ == "__main__":
    main()
