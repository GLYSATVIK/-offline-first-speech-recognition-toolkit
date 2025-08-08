"""``chunkasr mic`` — real-time transcription from the microphone.

Depends on the optional :mod:`sounddevice` package. Without it the command
exits cleanly with an explanatory message rather than crashing, so the base
install stays lean.
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import add_decode_arguments, build_recognizer


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("mic", help="transcribe live microphone audio")
    parser.add_argument("--seconds", type=float, default=5.0, help="capture duration")
    parser.add_argument("--sample-rate", type=int, default=16000, dest="sample_rate")
    add_decode_arguments(parser)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print("microphone capture requires the optional 'sounddevice' package")
        print("install it with: pip install 'chunkasr[mic]'")
        return 1

    recognizer = build_recognizer(args)
    frames = int(args.seconds * args.sample_rate)
    print(f"recording {args.seconds:.1f}s at {args.sample_rate} Hz ...")
    audio = sd.rec(frames, samplerate=args.sample_rate, channels=1, dtype="float32")
    sd.wait()
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    result = recognizer.transcribe(samples, args.sample_rate)
    print(result.text)
    return 0
