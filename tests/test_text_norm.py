from __future__ import annotations

from chunkasr.metrics.text_norm import TextNormalizer, normalize


def test_default_normalisation() -> None:
    assert normalize("Hello,  World!") == "hello world"


def test_keeps_apostrophes() -> None:
    assert normalize("don't stop") == "don't stop"


def test_collapse_whitespace() -> None:
    assert normalize("a\t b\n c") == "a b c"


def test_disable_lowercase() -> None:
    norm = TextNormalizer(lowercase=False)
    assert norm("Hello World") == "Hello World"


def test_spell_digits() -> None:
    norm = TextNormalizer(spell_digits=True)
    assert norm("route 66") == "route six six"
