"""Application version and build (git commit) information."""
from __future__ import annotations

import os
import subprocess

from .config import PROJECT_ROOT

__version__ = "0.6.0"


def _git_commit() -> str:
    # explicit override (useful in containers / systemd where git may be absent)
    env = os.environ.get("CAMDETECT_COMMIT")
    if env:
        return env[:12]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, timeout=2)
        return out.decode().strip()
    except Exception:
        return "unknown"


def build_info() -> dict:
    return {"version": __version__, "commit": _git_commit()}
