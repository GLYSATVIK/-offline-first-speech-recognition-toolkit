"""Detect voiced regions in a synthesised signal.

Run: uv run python examples/vad_segments.py
"""

from __future__ import annotations

import numpy as np

from chunkasr.audio import detect_speech


def main() -> None:
    sample_rate = 16000
    rng = np.random.default_rng(0)
    silence = np.zeros(int(0.4 * sample_rate), dtype=np.float32)
    n = int(0.6 * sample_rate)
    burst = 0.3 * np.sin(2.0 * np.pi * 200.0 * np.arange(n) / sample_rate)
    burst = burst.astype(np.float32) + 0.01 * rng.standard_normal(n).astype(np.float32)
    signal = np.concatenate([silence, burst, silence])

    segments = detect_speech(signal, sample_rate)
    print(f"found {len(segments)} voiced segment(s):")
    for seg in segments:
        print(f"  {seg.start:.2f}s - {seg.end:.2f}s  ({seg.duration:.2f}s)")


if __name__ == "__main__":
    main()
