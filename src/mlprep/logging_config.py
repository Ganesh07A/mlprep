"""Logging configuration for MLPrep.

Usage::

    from mlprep.logging_config import setup_logging, get_logger

    setup_logging(verbose=True)
    log = get_logger(__name__)
    log.info("Dataset loaded: %d rows", len(df))

Normal mode (verbose=False): only WARNING and above go to stderr.
Verbose mode (verbose=True):  INFO and above go to stderr with timestamps.

Terminal display (Rich console) is completely separate from logging — logs
are intentionally not mixed with Rich output so the terminal stays clean.
"""

from __future__ import annotations

import logging

_LOG_FORMAT_VERBOSE = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_FORMAT_QUIET = "[%(levelname)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger for MLPrep.

    Call once at CLI startup before any other module logs.

    Args:
        verbose: If True, set level to DEBUG and include timestamps.
                 If False, suppress everything below WARNING.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    fmt = _LOG_FORMAT_VERBOSE if verbose else _LOG_FORMAT_QUIET

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=_DATE_FORMAT,
        force=True,  # override any existing handlers (e.g. from pytest)
    )

    # Silence noisy third-party loggers even in verbose mode
    for noisy in ("sklearn", "joblib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``mlprep.<name>``."""
    return logging.getLogger(
        f"mlprep.{name}" if not name.startswith("mlprep") else name
    )
