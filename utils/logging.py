"""Structured logging setup for the quant platform.

Log level selection order: explicit ``level`` argument, then the environment
(``QUANT_LOG_LEVEL`` or ``LOG_LEVEL``, name or number, case-insensitive),
then ``INFO``. Environment overrides are read at process start only
(restart-only semantics); a running process never re-reads them.

File output uses a size-rotated handler (``RotatingFileHandler``) with
configurable ``max_bytes``/``backup_count`` and defaults of 10 MB x 5 files,
so an unattended daemon cannot fill the disk with one ever-growing log.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_logger: logging.Logger | None = None

DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5

_LEVEL_NAMES = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def resolve_level(level: int | str | None = None) -> int:
    """Resolve a log level from an argument or the environment.

    ``QUANT_LOG_LEVEL`` wins over ``LOG_LEVEL`` (docker-compose ships the
    latter). Unrecognized values fall back to INFO with a warning on stderr
    rather than raising at import time.
    """
    if isinstance(level, int):
        return level
    raw = level
    if raw is None:
        raw = os.environ.get("QUANT_LOG_LEVEL") or os.environ.get("LOG_LEVEL")
    if raw is None or str(raw).strip() == "":
        return logging.INFO
    text = str(raw).strip()
    if text.isdigit():
        return int(text)
    resolved = _LEVEL_NAMES.get(text.upper())
    if resolved is None:
        print(
            f"Warning: unknown log level {text!r} "
            f"(expected one of {sorted(_LEVEL_NAMES)}); using INFO",
            file=sys.stderr,
        )
        return logging.INFO
    return resolved


def setup_logging(
    level: int | str | None = None,
    log_file: str | Path | None = None,
    *,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> logging.Logger:
    """Configure and return the platform logger.

    Sets up both console (stderr) and optional rotating file output with
    consistent formatting including timestamps and module names.
    """
    global _logger

    resolved = resolve_level(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("quant_platform")
    logger.setLevel(resolved)
    logger.handlers.clear()
    logger.propagate = False

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(resolved)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Optional size-rotated file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=DEFAULT_MAX_BYTES if max_bytes is None else int(max_bytes),
            backupCount=(
                DEFAULT_BACKUP_COUNT if backup_count is None else int(backup_count)
            ),
            encoding="utf-8",
        )
        fh.setLevel(resolved)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _logger = logger
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for a specific module.

    Modules pass `__name__`, which is already ``quant_platform.<module>``;
    prefixing it unconditionally produced ``quant_platform.quant_platform.
    data.providers`` in every log line and made logger-name filtering
    unusable.
    """
    if name:
        if name == "quant_platform" or name.startswith("quant_platform."):
            return logging.getLogger(name)
        return logging.getLogger(f"quant_platform.{name}")
    return logging.getLogger("quant_platform")
