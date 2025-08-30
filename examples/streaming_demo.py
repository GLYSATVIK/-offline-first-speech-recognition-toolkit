"""Show that streaming and offline encoding produce identical emissions.

Run: uv run python examples/streaming_demo.py
"""

from __future__ import annotations

import numpy as np

from chunkasr.model.linear import LinearAcousticModel
from chunkasr.streaming.encoder import StreamingEncoder


def main() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((50, 8)).astype(np.float32)
    model = LinearAcousticModel.random(input_dim=8, vocab_size=6, seed=1)

    encoder = StreamingEncoder(model, chunk_size=8, right_context=4)
    offline = encoder.encode(features)

    encoder.reset()
    emitted = []
    for start in range(0, len(features), 7):  # arbitrary packet sizes
        emitted.append(encoder.accept(features[start : start + 7]))
    emitted.append(encoder.flush())
    streamed = np.concatenate(emitted, axis=0)

    print(f"offline shape : {offline.shape}")
    print(f"streamed shape: {streamed.shape}")
    print(f"identical     : {np.allclose(offline, streamed)}")


if __name__ == "__main__":
    main()
