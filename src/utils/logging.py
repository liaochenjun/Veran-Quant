from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(log_dir: Path, logger_name: str = "train") -> logging.Logger:
    """Create a logger that writes to both console and a timestamped file.

    The file lands under ``log_dir`` as ``<logger_name>_YYYYMMDD_HHMMSS.log``.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{logger_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
