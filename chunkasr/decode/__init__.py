"""Decoding: CTC and RNN-T search plus shallow n-gram LM fusion."""

from __future__ import annotations

from .ctc_greedy import collapse, ctc_greedy_decode, greedy_path
from .ctc_prefix_beam import PrefixScorer, ctc_prefix_beam_search
from .lm import CharLMScorer, NGramLM, NoLM
from .ngram import count_ngrams
from .rnnt_greedy import rnnt_greedy_decode
from .vocab import Vocabulary

__all__ = [
    "CharLMScorer",
    "NGramLM",
    "NoLM",
    "PrefixScorer",
    "Vocabulary",
    "collapse",
    "count_ngrams",
    "ctc_greedy_decode",
    "ctc_prefix_beam_search",
    "greedy_path",
    "rnnt_greedy_decode",
]
