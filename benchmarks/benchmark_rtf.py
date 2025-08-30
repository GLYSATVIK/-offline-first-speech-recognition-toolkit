"""Measure the real-time factor of the reference recogniser.

Run: uv run python benchmarks/benchmark_rtf.py
"""

from __future__ import annotations

import numpy as np

import chunkasr
from chunkasr.config import DecodeConfig
from chunkasr.metrics.rtf import RTFMeter


def bench(method: str, num_utts: int = 5, seconds: float = 2.0) -> float:
    sample_rate = 16000
    rng = np.random.default_rng(0)
    rec = chunkasr.Recognizer.demo(decode=DecodeConfig(method=method, beam_size=8))
    meter = RTFMeter()
    for _ in range(num_utts):
        samples = (0.1 * rng.standard_normal(int(seconds * sample_rate))).astype(np.float32)
        result = rec.transcribe(samples, sample_rate)
        meter.update(result.compute_seconds, result.audio_seconds)
    return meter.rtf


def main() -> None:
    for method in ("ctc_greedy", "ctc_beam", "rnnt_greedy"):
        print(f"{method:12s}: RTF = {bench(method):.4f}")


if __name__ == "__main__":
    main()
