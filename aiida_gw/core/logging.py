from __future__ import annotations

import logging
import sys
from pathlib import Path

from pydantic import BaseModel


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_path: str | None = None
    propagate: bool = False


_DEFAULT_CONFIG = LoggingConfig()


def setup_logging(config: LoggingConfig | None = None) -> None:
    cfg = config or _DEFAULT_CONFIG
    root_logger = logging.getLogger("aiida_gw")
    root_logger.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
    root_logger.propagate = cfg.propagate

    if root_logger.handlers:
        root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(cfg.format, datefmt=cfg.date_format)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if cfg.file_path:
        file_path = Path(cfg.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"aiida_gw.{name}")
