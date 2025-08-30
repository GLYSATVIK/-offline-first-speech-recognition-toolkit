"""Compute WER and CER between reference and hypothesis lists.

Run: uv run python examples/score_wer.py
"""

from __future__ import annotations

from chunkasr.metrics import corpus_cer, corpus_wer

REFERENCES = [
    "the quick brown fox jumps over the lazy dog",
    "speech recognition is fun",
    "hello world",
]
HYPOTHESES = [
    "the quick brown fox jumped over a lazy dog",
    "speech recognition is fun",
    "hello word",
]


def main() -> None:
    result = corpus_wer(REFERENCES, HYPOTHESES)
    print(f"WER : {result.as_percent():.2f}%")
    print(
        f"  substitutions={result.substitutions} "
        f"insertions={result.insertions} deletions={result.deletions}"
    )
    print(f"CER : {100.0 * corpus_cer(REFERENCES, HYPOTHESES):.2f}%")


if __name__ == "__main__":
    main()
