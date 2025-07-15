"""Split a feature matrix into fixed-size chunks with right context."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ..types import FloatArray


@dataclass(frozen=True)
class Chunk:
    """One streaming chunk: an output region plus its right-context lookahead."""

    index: int
    data: FloatArray
    context: FloatArray
    is_last: bool

    @property
    def num_frames(self) -> int:
        return int(self.data.shape[0])


def iter_chunks(
    features: FloatArray,
    chunk_size: int,
    right_context: int = 0,
) -> Iterator[Chunk]:
    """Yield :class:`Chunk` objects covering ``features`` left to right.

    Each chunk owns ``chunk_size`` output frames and carries up to
    ``right_context`` following frames as lookahead (never consumed as output).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if right_context < 0:
        raise ValueError("right_context must be non-negative")
    n = features.shape[0]
    idx = 0
    ci = 0
    while idx < n:
        end = min(idx + chunk_size, n)
        ctx_end = min(end + right_context, n)
        yield Chunk(
            index=ci,
            data=features[idx:end],
            context=features[end:ctx_end],
            is_last=end >= n,
        )
        idx = end
        ci += 1
