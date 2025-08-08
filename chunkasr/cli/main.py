"""CLI entry point and subcommand dispatch."""

from __future__ import annotations

import argparse

from .._version import __version__
from . import info, manifest, mic, score, transcribe


def build_parser() -> argparse.ArgumentParser:
    """Assemble the top-level parser and register every subcommand."""
    parser = argparse.ArgumentParser(
        prog="chunkasr",
        description="Streaming, offline-first speech-recognition toolkit",
    )
    parser.add_argument("--version", action="version", version=f"chunkasr {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for module in (transcribe, mic, score, manifest, info):
        module.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Program entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
