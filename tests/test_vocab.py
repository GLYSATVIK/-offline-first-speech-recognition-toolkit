from __future__ import annotations

import pytest

from chunkasr.decode.vocab import Vocabulary
from chunkasr.exceptions import DecodeError


def test_ascii_lower_size() -> None:
    vocab = Vocabulary.ascii_lower()
    assert len(vocab) == 28  # blank + space + 26 letters
    assert vocab.blank_id == 0


def test_encode_decode_roundtrip() -> None:
    vocab = Vocabulary.ascii_lower()
    ids = vocab.encode("hello world")
    assert vocab.decode_ids(ids) == "hello world"


def test_decode_skips_blanks() -> None:
    vocab = Vocabulary.ascii_lower()
    ids = [vocab.blank_id, *vocab.encode("hi"), vocab.blank_id]
    assert vocab.decode_ids(ids) == "hi"


def test_from_text_covers_symbols() -> None:
    vocab = Vocabulary.from_text("abc abc")
    assert vocab.decode_ids(vocab.encode("cab")) == "cab"


def test_unknown_symbol_raises() -> None:
    vocab = Vocabulary.ascii_lower()
    with pytest.raises(DecodeError):
        vocab.encode("A")  # uppercase not in the a-z vocabulary


def test_duplicate_symbols_rejected() -> None:
    with pytest.raises(DecodeError):
        Vocabulary(["<blank>", "a", "a"])
