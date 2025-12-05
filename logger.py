import datetime
import logging
import sys
from pathlib import Path

class Logger:
    """
    Simple project-wide logger that supports both console and file outputs.
    """
    def __init__(self, log_file: Path = None, level: str = "INFO"):
        self.logger = logging.getLogger("RxFusion")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.propagate = False  # Avoid double logging

        # Formatter with timestamp
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Optional file handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def section(self, title: str):
        """Pretty section header"""
        self.logger.info("=" * 60)
        self.logger.info(f"🔹 {title}")
        self.logger.info("=" * 60)


# -----------------------------
# Convenience Singleton
# -----------------------------
# You can import this `log` function anywhere:
# from utils.logger import log

_logger_instance = None

def get_logger():
    global _logger_instance
    if _logger_instance is None:
        # Default: console-only logger
        _logger_instance = Logger()
    return _logger_instance

def log(message: str, level: str = "info"):
    """Simple global logging shortcut."""
    logger = get_logger()
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(message)
