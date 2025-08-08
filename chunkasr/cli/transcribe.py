"""``chunkasr transcribe`` — batch transcription of audio files."""

from __future__ import annotations

import argparse
import json

from ..audio.io import load_audio
from .common import add_decode_arguments, build_recognizer


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("transcribe", help="transcribe one or more audio files")
    parser.add_argument("audio", nargs="+", help="audio file(s) to transcribe")
    parser.add_argument("--timestamps", action="store_true", help="print word timings")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    add_decode_arguments(parser)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    recognizer = build_recognizer(args)
    for path in args.audio:
        samples, sample_rate = load_audio(path)
        result = recognizer.transcribe(samples, sample_rate)
        if args.json:
            print(json.dumps({"audio": path, **result.to_dict()}))
            continue
        print(f"{path}\t{result.text}")
        if args.timestamps:
            for word in result.words:
                print(f"  [{word.start:6.2f} - {word.end:6.2f}] {word.word}")
    return 0
