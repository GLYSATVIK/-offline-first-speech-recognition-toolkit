"""Shared helpers for CLI subcommands."""

from __future__ import annotations

import argparse

from ..config import DecodeConfig, ExperimentConfig, load_config
from ..decode.vocab import Vocabulary
from ..pipeline import Recognizer


def build_recognizer(args: argparse.Namespace) -> Recognizer:
    """Construct a :class:`Recognizer` from parsed CLI arguments."""
    config: ExperimentConfig | None = None
    if getattr(args, "config", None):
        config = load_config(args.config)

    decode = DecodeConfig(
        method=getattr(args, "method", "ctc_greedy"),
        beam_size=getattr(args, "beam_size", 8),
    )
    if config is not None:
        recognizer = Recognizer(
            model=Recognizer.demo(seed=config.seed).model,
            vocab=Vocabulary.ascii_lower(),
            feature=config.feature,
            decode=decode,
        )
    else:
        recognizer = Recognizer.demo(seed=getattr(args, "seed", 1234), decode=decode)

    if getattr(args, "streaming", False):
        recognizer.enable_streaming(
            chunk_size=getattr(args, "chunk_size", 16),
            right_context=getattr(args, "right_context", 4),
        )
    return recognizer


def add_decode_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the flags shared by decoding subcommands."""
    parser.add_argument(
        "--method",
        default="ctc_greedy",
        choices=["ctc_greedy", "ctc_beam", "rnnt_greedy"],
        help="decoding strategy",
    )
    parser.add_argument("--beam-size", type=int, default=8, dest="beam_size")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--config", default=None, help="experiment config (JSON/YAML)")
    parser.add_argument("--streaming", action="store_true", help="use the chunked encoder")
    parser.add_argument("--chunk-size", type=int, default=16, dest="chunk_size")
    parser.add_argument("--right-context", type=int, default=4, dest="right_context")
