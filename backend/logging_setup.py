"""Logging configuration: console + rotating file + in-memory ring buffer.

The ring buffer backs the in-app Debug window. Verbosity comes from the config
file (``logging.level``) but can be overridden at startup via the
``CAMDETECT_LOG_LEVEL`` environment variable, and adjusted at runtime through
``POST /api/log-level``.
"""
from __future__ import annotations

import collections
import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional

from .config import Config

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class RingBufferHandler(logging.Handler):
    """Keeps the last N formatted log records in memory for the debug window."""

    def __init__(self, capacity: int = 2000):
        super().__init__()
        self.buffer: collections.deque = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return
        with self._lock:
            self._seq += 1
            self.buffer.append({
                "seq": self._seq,
                "ts": record.created,
                "level": record.levelname,
                "name": record.name,
                "msg": msg,
            })

    def tail(self, after_seq: int = 0, limit: int = 500) -> list[dict]:
        with self._lock:
            items = [r for r in self.buffer if r["seq"] > after_seq]
        return items[-limit:]


# Module-level singleton so the API can read it.
ring_handler: Optional[RingBufferHandler] = None


def resolve_level(cfg_level: str) -> int:
    """Resolve effective level from env override or config."""
    name = os.environ.get("CAMDETECT_LOG_LEVEL", cfg_level or "INFO").upper()
    return getattr(logging, name, logging.INFO)


def setup_logging(cfg: Config) -> RingBufferHandler:
    global ring_handler
    level = resolve_level(cfg.logging.level)

    root = logging.getLogger()
    root.setLevel(level)
    # Clear pre-existing handlers (uvicorn/basicConfig) to avoid duplicates.
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(_FMT)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    log_path = cfg.abspath(cfg.logging.file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fileh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3)
    fileh.setFormatter(fmt)
    root.addHandler(fileh)

    ring_handler = RingBufferHandler(cfg.logging.max_lines)
    ring_handler.setFormatter(fmt)
    root.addHandler(ring_handler)

    logging.getLogger("camdetect").info("Logging at %s -> %s",
                                        logging.getLevelName(level), log_path)
    return ring_handler


def set_level(name: str) -> str:
    level = getattr(logging, name.upper(), None)
    if level is None:
        return logging.getLevelName(logging.getLogger().level)
    logging.getLogger().setLevel(level)
    return logging.getLevelName(level)
