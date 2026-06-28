"""Local SQLite store for objects and events.

Two tables:
- ``objects``  - one row per physical object, merged across time/cameras by
  similarity (same class within a distance/time gap collapses into one row that
  accumulates first/last seen, max confidence, and a JSON ``attrs`` blob with
  the latest extras (behavior, age, plate, make/model, engine_type, ...)).
- ``events``   - an append-only log (object appeared/updated/left, audio events,
  plate reads, transcripts, ...).

The pipeline calls :meth:`Database.ingest` each fusion tick with the current
tracks; the store decides whether each maps to an existing object or a new one.
Writes happen on a single background thread so the pipeline never blocks on I/O.
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import sqlite3
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class TEXT NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    last_x REAL, last_y REAL,
    max_conf REAL,
    observations INTEGER DEFAULT 1,
    attrs TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    object_id INTEGER,
    cam TEXT,
    label TEXT,
    data TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_objects_last ON objects(last_seen);
"""


def _dist(ax, ay, bx, by) -> float:
    if None in (ax, ay, bx, by):
        return 1e9
    return math.hypot(ax - bx, ay - by)


class Database:
    def __init__(self, path: str, merge_distance_m: float = 3.0,
                 merge_time_s: float = 60.0, retention_days: int = 30):
        self.path = path
        self.merge_distance_m = merge_distance_m
        self.merge_time_s = merge_time_s
        self.retention_days = retention_days
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self._q: "queue.Queue" = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # maps live track id -> db object id (per process run)
        self._track_to_obj: dict[int, int] = {}
        self._conn: Optional[sqlite3.Connection] = None
        self._last_purge = 0.0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="db", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        while not self._stop.is_set():
            try:
                op = self._q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                op()
            except Exception as exc:  # pragma: no cover
                logger.warning("db op failed: %s", exc)

    def _submit(self, fn) -> None:
        try:
            self._q.put_nowait(fn)
        except queue.Full:  # pragma: no cover
            logger.warning("db queue full; dropping write")

    # -- ingest tracks -----------------------------------------------------
    def ingest(self, tracks: list[dict]) -> None:
        """Queue an upsert of the current fused tracks."""
        snapshot = [dict(t) for t in tracks]
        self._submit(lambda: self._ingest(snapshot))

    def _ingest(self, tracks: list[dict]) -> None:
        now = time.time()
        cur = self._conn.cursor()
        for t in tracks:
            obj_id = self._track_to_obj.get(t["id"])
            attrs = {k: t.get(k) for k in
                     ("behavior", "age", "engine_type", "plate", "make",
                      "model", "vehicle_age", "drivetrain", "cameras")
                     if t.get(k) is not None}
            if obj_id is None:
                obj_id = self._find_similar(cur, t, now)
            if obj_id is None:
                cur.execute(
                    "INSERT INTO objects(class, first_seen, last_seen, last_x, "
                    "last_y, max_conf, observations, attrs) "
                    "VALUES (?,?,?,?,?,?,1,?)",
                    (t["class"], now, now, t.get("x"), t.get("y"),
                     t.get("prob", 0.0), json.dumps(attrs)))
                obj_id = cur.lastrowid
                self._log(cur, now, "object_new", obj_id,
                          (t.get("cameras") or [None])[0], t["class"],
                          {"x": t.get("x"), "y": t.get("y")})
            else:
                cur.execute(
                    "UPDATE objects SET last_seen=?, last_x=?, last_y=?, "
                    "max_conf=MAX(max_conf, ?), observations=observations+1, "
                    "attrs=? WHERE id=?",
                    (now, t.get("x"), t.get("y"), t.get("prob", 0.0),
                     json.dumps(attrs), obj_id))
            self._track_to_obj[t["id"]] = obj_id
        self._conn.commit()
        self._maybe_purge(now)

    def _find_similar(self, cur, t: dict, now: float) -> Optional[int]:
        cur.execute(
            "SELECT id, last_x, last_y FROM objects WHERE class=? AND "
            "last_seen > ? ORDER BY last_seen DESC LIMIT 25",
            (t["class"], now - self.merge_time_s))
        for oid, lx, ly in cur.fetchall():
            if _dist(t.get("x"), t.get("y"), lx, ly) <= self.merge_distance_m:
                return oid
        return None

    # -- events ------------------------------------------------------------
    def log_event(self, kind: str, cam: Optional[str] = None,
                  label: Optional[str] = None, data: Optional[dict] = None,
                  object_id: Optional[int] = None) -> None:
        ts = time.time()
        self._submit(lambda: self._log_and_commit(ts, kind, object_id, cam, label, data))

    def _log_and_commit(self, ts, kind, object_id, cam, label, data) -> None:
        self._log(self._conn.cursor(), ts, kind, object_id, cam, label, data)
        self._conn.commit()

    def _log(self, cur, ts, kind, object_id, cam, label, data) -> None:
        cur.execute(
            "INSERT INTO events(ts, kind, object_id, cam, label, data) "
            "VALUES (?,?,?,?,?,?)",
            (ts, kind, object_id, cam, label,
             json.dumps(data) if data is not None else None))

    def _maybe_purge(self, now: float) -> None:
        if now - self._last_purge < 3600:
            return
        self._last_purge = now
        cutoff = now - self.retention_days * 86400
        c = self._conn.cursor()
        c.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
        c.execute("DELETE FROM objects WHERE last_seen < ?", (cutoff,))
        self._conn.commit()

    # -- queries (read on a short-lived connection; thread-safe) -----------
    def _read_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def list_objects(self, limit: int = 200, cls: Optional[str] = None) -> list[dict]:
        conn = self._read_conn()
        try:
            q = "SELECT * FROM objects"
            args: list[Any] = []
            if cls:
                q += " WHERE class=?"
                args.append(cls)
            q += " ORDER BY last_seen DESC LIMIT ?"
            args.append(limit)
            rows = conn.execute(q, args).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            d = dict(r)
            d["attrs"] = json.loads(d["attrs"]) if d.get("attrs") else {}
            out.append(d)
        return out

    def list_events(self, limit: int = 200, kind: Optional[str] = None) -> list[dict]:
        conn = self._read_conn()
        try:
            q = "SELECT * FROM events"
            args: list[Any] = []
            if kind:
                q += " WHERE kind=?"
                args.append(kind)
            q += " ORDER BY ts DESC LIMIT ?"
            args.append(limit)
            rows = conn.execute(q, args).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"]) if d.get("data") else None
            out.append(d)
        return out

    def stats(self) -> dict:
        conn = self._read_conn()
        try:
            nobj = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
            nev = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            by_class = {r[0]: r[1] for r in conn.execute(
                "SELECT class, COUNT(*) FROM objects GROUP BY class").fetchall()}
        finally:
            conn.close()
        return {"objects": nobj, "events": nev, "by_class": by_class}
