"""
Common logging module for the ConvFinQA system.
Provides consistent logging configuration across all modules.
"""

import logging
import os
import sys
from datetime import datetime

from src.main.config import get_config

CONFIG = get_config()

LOGS_DIR = CONFIG["log"]["log_dir"]
os.makedirs(LOGS_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d")
LOG_FILE = os.path.join(LOGS_DIR, f"convfinqa_{TIMESTAMP}.log")

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers = []

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


default_logger = setup_logger("convfinqa")
