"""Synthesise a tone, transcribe it, and print word timings.

Run: uv run python examples/transcribe_wav.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import chunkasr


def main() -> None:
    sample_rate = 16000
    t = np.arange(sample_rate) / sample_rate
    tone = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "tone.wav"
        chunkasr.write_wav(wav_path, tone, sample_rate)
        samples, sr = chunkasr.read_wav(wav_path)

    rec = chunkasr.Recognizer.demo()
    result = rec.transcribe(samples, sample_rate=sr)

    print(f"text : {result.text!r}")
    print(f"audio: {result.audio_seconds:.2f}s   rtf: {result.rtf:.3f}")
    for word in result.words:
        print(f"  [{word.start:5.2f}-{word.end:5.2f}] {word.word}")


if __name__ == "__main__":
    main()
