"""Daily report: a textual summary of what happened plus a set of snapshots.

A background thread saves an annotated snapshot per camera every
``snapshot_interval_s`` into ``data/report/<YYYY-MM-DD>/`` (capped per day). The
text summary is generated on demand from the database (object counts by class,
event counts, busiest hour, vehicles with plates, recent transcripts).
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

CLASS_CS = {
    "person": "clovek", "car": "auto", "truck": "nakladni auto", "bus": "autobus",
    "motorcycle": "motorka", "bicycle": "kolo", "dog": "pes", "cat": "kocka",
    "bird": "ptak", "trash bin": "popelnice", "scooter": "kolobezka",
    "skates": "brusle", "drone": "dron",
}


def _day_bounds(date_str: str) -> tuple[float, float]:
    d = dt.datetime.strptime(date_str, "%Y-%m-%d")
    start = dt.datetime(d.year, d.month, d.day)
    return start.timestamp(), (start + dt.timedelta(days=1)).timestamp()


class ReportManager:
    def __init__(self, cfg, get_jpeg: Callable[[str], Optional[bytes]],
                 cameras: list, db=None):
        self.cfg = cfg
        self.get_jpeg = get_jpeg
        self.cameras = cameras
        self.db = db
        self.dir = cfg.abspath(cfg.report.dir)
        os.makedirs(self.dir, exist_ok=True)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- snapshots ---------------------------------------------------------
    def start(self) -> None:
        if not self.cfg.report.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="report")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _day_dir(self, date_str: str) -> str:
        return os.path.join(self.dir, date_str)

    def _loop(self) -> None:
        interval = max(10.0, float(self.cfg.report.snapshot_interval_s))
        while not self._stop.is_set():
            date_str = time.strftime("%Y-%m-%d")
            day = self._day_dir(date_str)
            os.makedirs(day, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            for cam in self.cameras:
                jpeg = self.get_jpeg(cam.id)
                if jpeg:
                    try:
                        with open(os.path.join(day, f"{cam.id}_{stamp}.jpg"), "wb") as fh:
                            fh.write(jpeg)
                    except Exception as exc:  # pragma: no cover
                        logger.debug("snapshot write failed: %s", exc)
            self._prune(day)
            self._stop.wait(interval)

    def _prune(self, day: str) -> None:
        try:
            files = sorted(os.listdir(day))
            cap = int(self.cfg.report.max_images_per_day)
            if len(files) > cap:
                for f in files[: len(files) - cap]:
                    os.remove(os.path.join(day, f))
        except Exception:  # pragma: no cover
            pass

    # -- report generation -------------------------------------------------
    def images(self, date_str: str) -> list[str]:
        day = self._day_dir(date_str)
        if not os.path.isdir(day):
            return []
        return sorted([f for f in os.listdir(day) if f.endswith(".jpg")])

    def text(self, date_str: str) -> str:
        lines = [f"Denni report {date_str}", ""]
        if self.db is None:
            return "\n".join(lines + ["(databaze vypnuta)"])
        t0, t1 = _day_bounds(date_str)
        s = self.db.summary(t0, t1)
        if not s["by_class"] and not s["n_events"]:
            return "\n".join(lines + ["Zadna aktivita."])

        objs = ", ".join(f"{CLASS_CS.get(k, k)}: {v}" for k, v in s["by_class"].items())
        lines.append(f"Detekovane objekty: {objs or 'zadne'}")
        lines.append(f"Udalosti celkem: {s['n_events']}")
        if s["by_kind"]:
            lines.append("  " + ", ".join(f"{k}: {v}" for k, v in s["by_kind"].items()))
        if s["by_hour"]:
            peak = max(s["by_hour"], key=s["by_hour"].get)
            lines.append(f"Nejrusnejsi hodina: {peak}:00 ({s['by_hour'][peak]} udalosti)")
        if s["vehicles"]:
            lines.append("")
            lines.append("Vozidla:")
            seen = set()
            for v in s["vehicles"]:
                key = v.get("plate") or f"{v.get('make')} {v.get('model')}"
                if key in seen:
                    continue
                seen.add(key)
                desc = " ".join(filter(None, [v.get("plate"), v.get("make"),
                                              v.get("model"), v.get("drivetrain")]))
                lines.append(f"  - {desc}")
        if s["transcripts"]:
            lines.append("")
            lines.append(f"Prepisy reci ({len(s['transcripts'])}):")
            for tr in s["transcripts"][:10]:
                who = tr.get("person") or tr.get("speaker") or "?"
                lines.append(f"  [{who}] {tr.get('text', '')}")
        lines.append("")
        lines.append(f"Snimku k dispozici: {len(self.images(date_str))}")
        return "\n".join(lines)

    def report(self, date_str: str) -> dict:
        return {"date": date_str, "text": self.text(date_str),
                "images": self.images(date_str)}

    def image_path(self, date_str: str, name: str) -> Optional[str]:
        if "/" in name or "\\" in name or "/" in date_str:
            return None
        p = os.path.join(self._day_dir(date_str), name)
        return p if os.path.exists(p) else None
