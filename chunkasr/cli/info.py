"""``chunkasr info`` — report version and optional-backend availability."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("info", help="show version and backend availability")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    import numpy as np

    from .._version import __version__
    from ..model.backend import torch_available

    print(f"chunkasr    : {__version__}")
    print(f"numpy       : {np.__version__}")
    try:
        import scipy

        print(f"scipy       : {scipy.__version__}")
    except ImportError:
        print("scipy       : (not installed)")
    print(f"torch       : {'available' if torch_available() else 'not installed'}")
    try:
        import sounddevice  # noqa: F401

        print("sounddevice : available")
    except (ImportError, OSError):
        print("sounddevice : not installed")
    return 0
