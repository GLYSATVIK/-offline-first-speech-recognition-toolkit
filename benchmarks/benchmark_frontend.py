"""Time the log-Mel frontend over a range of utterance lengths.

Run: uv run python benchmarks/benchmark_frontend.py
"""

from __future__ import annotations

import time

import numpy as np

from chunkasr.config import FeatureConfig
from chunkasr.frontend import FeatureExtractor


def main() -> None:
    sample_rate = 16000
    extractor = FeatureExtractor(FeatureConfig(num_mel_bins=80))
    rng = np.random.default_rng(0)
    for seconds in (1.0, 5.0, 30.0):
        samples = rng.standard_normal(int(seconds * sample_rate)).astype(np.float32)
        start = time.perf_counter()
        feats = extractor(samples)
        elapsed = time.perf_counter() - start
        rtf = elapsed / seconds
        n_frames = feats.shape[0]
        print(f"{seconds:5.1f}s -> {n_frames:5d} frames in {elapsed * 1e3:7.2f} ms (rtf {rtf:.4f})")


if __name__ == "__main__":
    main()
