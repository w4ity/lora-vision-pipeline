"""Small shared utilities: seeding, logging, markdown tables."""
from __future__ import annotations

import logging
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "lvp") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def md_table(headers: list[str], rows: list[list]) -> str:
    """Render a GitHub-flavored markdown table without extra dependencies."""
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)
