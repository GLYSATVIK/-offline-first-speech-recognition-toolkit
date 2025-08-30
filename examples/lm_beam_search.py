"""Train a character n-gram LM and decode with shallow fusion.

Run: uv run python examples/lm_beam_search.py
"""

from __future__ import annotations

import numpy as np

from chunkasr.decode.ctc_prefix_beam import ctc_prefix_beam_search
from chunkasr.decode.lm import CharLMScorer, NGramLM
from chunkasr.decode.vocab import Vocabulary
from chunkasr.model.backend import log_softmax


def main() -> None:
    vocab = Vocabulary.ascii_lower()
    corpus = ["the cat sat on the mat", "the dog ran", "a cat and a dog"]
    lm = NGramLM.train_chars(corpus, order=3)
    scorer = CharLMScorer(lm, vocab, weight=0.5)

    rng = np.random.default_rng(7)
    log_probs = log_softmax(rng.standard_normal((30, len(vocab))).astype(np.float32))

    without_lm = vocab.decode_ids(ctc_prefix_beam_search(log_probs, beam_size=8))
    with_lm = vocab.decode_ids(ctc_prefix_beam_search(log_probs, beam_size=8, scorer=scorer))

    print(f"no LM   : {without_lm!r}")
    print(f"with LM : {with_lm!r}")


if __name__ == "__main__":
    main()
