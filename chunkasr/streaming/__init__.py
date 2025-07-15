"""Chunked streaming: split the feature stream and drive the encoder online."""

from __future__ import annotations

from .chunker import Chunk, iter_chunks
from .encoder import StreamingEncoder

__all__ = ["Chunk", "StreamingEncoder", "iter_chunks"]
