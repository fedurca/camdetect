"""Thread-safe runtime settings.

Mirrors the toggleable / tunable parts of :mod:`backend.config` so the UI can
turn detection types on/off and adjust CPU load (fps, resolution, audio window,
heavy models) live, without restarting the server. The pipeline reads a snapshot
each loop; the API writes partial updates.
"""
from __future__ import annotations

import copy
import threading
from typing import Any

from .config import Config


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge ``patch`` into ``base`` (in place) and return it."""
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


class Settings:
    """Holds the live, mutable runtime settings behind a lock."""

    def __init__(self, cfg: Config):
        d = cfg.detection
        a = cfg.audio
        self._data: dict[str, Any] = {
            "video": {
                "enabled": d.enabled,
                "fps": d.fps,
                "imgsz": d.imgsz,
                "confidence": d.confidence,
                "open_vocabulary": {
                    "enabled": d.open_vocabulary.enabled,
                    "confidence": d.open_vocabulary.confidence,
                    "prompts": list(d.open_vocabulary.prompts),
                },
            },
            "attributes": {
                "behavior": d.attributes.behavior,
                "age": d.attributes.age,
            },
            "audio": {
                "enabled": a.enabled,
                "events": a.events.enabled,
                "engine_2t4t": a.engine_2t4t,
                "window_s": a.window_s,
                "hop_s": a.hop_s,
            },
        }
        self._lock = threading.Lock()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            _deep_merge(self._data, patch)
            return copy.deepcopy(self._data)

    # Convenience accessors (each takes a fresh lock; cheap dict reads).
    def get(self, *path: str, default: Any = None) -> Any:
        with self._lock:
            node: Any = self._data
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    return default
                node = node[key]
            return copy.deepcopy(node)
