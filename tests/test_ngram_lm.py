from __future__ import annotations

import numpy as np

from chunkasr.decode.ctc_prefix_beam import ctc_prefix_beam_search
from chunkasr.decode.lm import CharLMScorer, NGramLM, NoLM
from chunkasr.decode.ngram import count_ngrams
from chunkasr.decode.vocab import Vocabulary
from chunkasr.model.backend import log_softmax


def test_count_ngrams() -> None:
    counts, contexts, vocab = count_ngrams([["a", "b", "c"]], order=2)
    assert counts[("a", "b")] == 1
    assert contexts[("a",)] == 1
    assert "a" in vocab


def test_lm_prefers_seen_continuation() -> None:
    lm = NGramLM.train_chars(["hello", "hello", "help"], order=3)
    seen = lm.log_prob(list("hell"), "o")
    unseen = lm.log_prob(list("hell"), "z")
    assert seen > unseen


def test_lm_score_is_sum_of_log_probs() -> None:
    lm = NGramLM.train_chars(["abcabc"], order=2)
    tokens = list("abc")
    total = lm.score(tokens)
    manual = sum(lm.log_prob(tokens[:i], tokens[i]) for i in range(len(tokens)))
    assert np.isclose(total, manual)


def test_char_lm_scorer_shifts_nothing_at_zero_weight() -> None:
    vocab = Vocabulary.ascii_lower()
    lm = NGramLM.train_chars(["hello world"], order=3)
    scorer = CharLMScorer(lm, vocab, weight=0.0)
    assert scorer.score([2, 3], 4) == 0.0


def test_beam_with_lm_runs() -> None:
    vocab = Vocabulary.ascii_lower()
    lm = NGramLM.train_chars(["the cat", "the dog"], order=3)
    scorer = CharLMScorer(lm, vocab, weight=0.5)
    logits = np.random.default_rng(0).standard_normal((15, len(vocab))).astype(np.float32)
    lp = log_softmax(logits)
    hyp = ctc_prefix_beam_search(lp, beam_size=6, scorer=scorer)
    assert isinstance(hyp, list)


def test_no_lm_scorer() -> None:
    assert NoLM().score([1, 2, 3], 4) == 0.0
