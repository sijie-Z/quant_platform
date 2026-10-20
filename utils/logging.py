"""Structured logging setup for the quant platform."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_logger: logging.Logger | None = None


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return the platform logger.

    Sets up both console (stderr) and optional file output with consistent
    formatting including timestamps and module names.
    """
    global _logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("quant_platform")
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
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
