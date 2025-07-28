"""Text normalisation for fair error-rate scoring.

Word-error rate is meaningless unless reference and hypothesis are normalised
the same way, so this lives next to the scoring code rather than in a utility
grab-bag.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, cast

NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]

_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


@dataclass(frozen=True)
class TextNormalizer:
    """Configurable normaliser applied identically to references and hypotheses."""

    lowercase: bool = True
    strip_punctuation: bool = True
    collapse_whitespace: bool = True
    unicode_form: str = "NFKC"
    spell_digits: bool = False

    def __call__(self, text: str) -> str:
        if self.unicode_form:
            form = cast("NormalizationForm", self.unicode_form)
            text = unicodedata.normalize(form, text)
        if self.lowercase:
            text = text.lower()
        if self.spell_digits:
            text = "".join(f" {_NUMBER_WORDS[c]} " if c in _NUMBER_WORDS else c for c in text)
        if self.strip_punctuation:
            text = _PUNCT_RE.sub(" ", text)
        if self.collapse_whitespace:
            text = _WS_RE.sub(" ", text).strip()
        return text


_DEFAULT = TextNormalizer()


def normalize(text: str) -> str:
    """Apply the default normalisation (lowercase, de-punctuate, squeeze space)."""
    return _DEFAULT(text)
