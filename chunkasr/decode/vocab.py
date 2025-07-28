"""Symbol table mapping between token strings and integer ids.

The default vocabulary is character-level with an explicit blank symbol at id 0
(the CTC convention) and a word-boundary symbol that renders as a space.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..exceptions import DecodeError

BLANK = "<blank>"
SPACE = "|"


class Vocabulary:
    """Bidirectional map between symbols and ids with blank/space handling."""

    def __init__(self, symbols: Sequence[str], blank_id: int = 0, space_symbol: str = SPACE):
        if len(set(symbols)) != len(symbols):
            raise DecodeError("vocabulary symbols must be unique")
        self._symbols = list(symbols)
        self._index = {s: i for i, s in enumerate(self._symbols)}
        if not 0 <= blank_id < len(self._symbols):
            raise DecodeError("blank_id out of range")
        self.blank_id = blank_id
        self.space_symbol = space_symbol

    def __len__(self) -> int:
        return len(self._symbols)

    def id_to_symbol(self, idx: int) -> str:
        return self._symbols[idx]

    def symbol_to_id(self, symbol: str) -> int:
        try:
            return self._index[symbol]
        except KeyError as exc:
            raise DecodeError(f"unknown symbol: {symbol!r}") from exc

    def encode(self, text: str) -> list[int]:
        """Turn a string into token ids, mapping spaces to the space symbol."""
        ids: list[int] = []
        for ch in text:
            symbol = self.space_symbol if ch == " " else ch
            ids.append(self.symbol_to_id(symbol))
        return ids

    def decode_ids(self, ids: Sequence[int]) -> str:
        """Turn token ids into text, dropping blanks and expanding spaces."""
        chars: list[str] = []
        for idx in ids:
            if idx == self.blank_id:
                continue
            symbol = self._symbols[idx]
            chars.append(" " if symbol == self.space_symbol else symbol)
        return "".join(chars)

    @classmethod
    def from_text(cls, text: str, blank: str = BLANK, space_symbol: str = SPACE) -> Vocabulary:
        """Build a character vocabulary covering every symbol in ``text``."""
        seen: list[str] = [blank]
        for ch in text:
            symbol = space_symbol if ch == " " else ch
            if symbol not in seen:
                seen.append(symbol)
        return cls(seen, blank_id=0, space_symbol=space_symbol)

    @classmethod
    def ascii_lower(cls) -> Vocabulary:
        """A fixed a-z + space vocabulary, handy for reproducible demos."""
        symbols = [BLANK, SPACE, *(chr(c) for c in range(ord("a"), ord("z") + 1))]
        return cls(symbols, blank_id=0)
