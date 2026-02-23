"""
Centralized logging configuration for the project.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")

All modules share the same formatting and handlers.
Console output uses colored level names; file output (when enabled)
is plain text with timestamps.
"""

import logging
import os
import sys

from src.config import OUTPUTS_DIR

# ──────────────────────────────────────────────
# FORMAT
# ──────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = os.path.join(OUTPUTS_DIR, "project.log")

_configured = False


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_file: str | None = None,
) -> None:
    """
    Configure the root logger with console and optional file handlers.

    Args:
        level: Logging level (default: INFO).
        log_to_file: Whether to also log to a file.
        log_file: Path to the log file. Defaults to outputs/project.log.
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (optional)
    if log_to_file:
        fpath = log_file or LOG_FILE
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        file_handler = logging.FileHandler(fpath, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name. Automatically sets up logging
    on first call.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured Logger instance.
    """
    setup_logging()
    return logging.getLogger(name)
