"""``chunkasr score`` — WER/CER/RTF between reference and hypothesis files."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..metrics.cer import corpus_cer
from ..metrics.wer import corpus_wer


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("score", help="score hypotheses against references")
    parser.add_argument("--ref", required=True, help="reference transcript file (one per line)")
    parser.add_argument("--hyp", required=True, help="hypothesis transcript file (one per line)")
    parser.set_defaults(func=run)


def _read_lines(path: str) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def run(args: argparse.Namespace) -> int:
    refs = _read_lines(args.ref)
    hyps = _read_lines(args.hyp)
    if len(refs) != len(hyps):
        print(f"error: {len(refs)} references but {len(hyps)} hypotheses")
        return 2
    result = corpus_wer(refs, hyps)
    char_rate = corpus_cer(refs, hyps)
    print(f"utterances : {len(refs)}")
    print(f"WER        : {result.as_percent():.2f}%")
    print(f"CER        : {100.0 * char_rate:.2f}%")
    print(f"breakdown  : sub={result.substitutions} ins={result.insertions} del={result.deletions}")
    return 0
