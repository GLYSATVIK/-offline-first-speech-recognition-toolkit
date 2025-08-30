"""Generate a tiny synthetic dataset and its JSONL manifest for smoke testing.

Run: uv run python scripts/make_toy_manifest.py /tmp/toy --count 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chunkasr.audio.io import write_wav
from chunkasr.data.manifest import ManifestEntry, write_manifest

WORDS = ["alpha", "bravo", "charlie", "delta", "echo"]


def main() -> None:
    parser = argparse.ArgumentParser(description="make a toy chunkasr dataset")
    parser.add_argument("out", help="output directory")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000, dest="sample_rate")
    args = parser.parse_args()

    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(args.count):
        duration = 1.0 + 0.5 * i
        n = int(duration * args.sample_rate)
        freq = 150.0 + 40.0 * i
        t = np.arange(n) / args.sample_rate
        waveform = (0.2 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        path = audio_dir / f"utt{i:03d}.wav"
        write_wav(path, waveform, args.sample_rate)
        entries.append(
            ManifestEntry(str(path), duration, WORDS[i % len(WORDS)], utt_id=f"utt{i:03d}")
        )

    manifest = out / "manifest.jsonl"
    write_manifest(manifest, entries)
    print(f"wrote {len(entries)} utterances to {manifest}")


if __name__ == "__main__":
    main()
