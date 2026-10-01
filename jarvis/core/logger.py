"""
jarvis.core.logger
~~~~~~~~~~~~~~~~~~
Configures the Loguru global logger for JARVIS.

Import and call ``setup_logging()`` once at startup (done in ``main.py``).
All other modules simply use::

    from loguru import logger
    logger.info("My message")

The logger writes to stderr (coloured) and to a rotating file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from jarvis.core.config_loader import cfg


def setup_logging() -> None:
    """Configure Loguru sinks based on JARVIS config.

    Should be called exactly once, early in startup before any other
    module emits log messages.

    Returns:
        None
    """
    level: str = cfg.get("logging.level", default="INFO").upper()
    log_file: str = cfg.get("logging.file", default="logs/jarvis.log")
    rotation: str = cfg.get("logging.rotation", default="10 MB")
    retention: str = cfg.get("logging.retention", default="7 days")
    colorize: bool = cfg.get("logging.colorize", default=True)

    # Resolve log path relative to repo root
    log_path = Path(__file__).resolve().parent.parent.parent / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove default sink and add our own
    logger.remove()

    # Stderr sink (human-readable, coloured)
    fmt_stderr = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> – <level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=fmt_stderr,
        level=level,
        colorize=colorize,
        enqueue=True,
    )

    # File sink (full timestamps, no colour codes)
    fmt_file = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} – {message}"
    )
    logger.add(
        str(log_path),
        format=fmt_file,
        level="DEBUG",          # always capture debug in file
        rotation=rotation,
        retention=retention,
        enqueue=True,
        encoding="utf-8",
    )

    logger.info(f"JARVIS logging initialised  level={level}  file={log_path}")
