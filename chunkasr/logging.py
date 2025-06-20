"""Package-wide logging configuration.

A tiny wrapper so every module logs through the ``chunkasr`` namespace and CLI
entry points can raise or lower verbosity in one call.
"""

from __future__ import annotations

import logging

_ROOT_NAME = "chunkasr"
_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a single stream handler to the package logger.

    Idempotent: calling it more than once only adjusts the level so importing
    the library never installs duplicate handlers.
    """
    global _CONFIGURED
    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``chunkasr`` namespace."""
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
