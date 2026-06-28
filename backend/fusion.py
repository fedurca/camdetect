"""Cross-camera fusion and tracking in world coordinates.

Per frame, each camera contributes localized detections (class, world X/Y,
confidence). :class:`Fusion` merges detections from different cameras that fall
within ``merge_distance_m`` of each other into single physical objects, then
associates those merged observations with persistent tracks (stable IDs) and
smooths their positions. Stale tracks are dropped after ``max_age_s``.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Optional

from .classes import height_for

# Speed thresholds (m/s) used to classify person behavior.
WALK_SPEED = 0.4
RUN_SPEED = 2.2
# A person present this long with little net displacement is "loitering".
LOITER_TIME_S = 15.0
LOITER_RADIUS_M = 2.0


@dataclass
class WorldDetection:
    """A single camera's detection projected onto the world ground plane."""

    camera_id: str
    class_id: int
    class_name: str
    confidence: float
    x: float
    y: float


@dataclass
class Track:
    id: int
    class_name: str
    class_id: int
    x: float
    y: float
    confidence: float
    height: float
    cameras: list[str] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    hits: int = 0
    # motion / attributes
    history: deque = field(default_factory=lambda: deque(maxlen=64))
    speed: float = 0.0
    behavior: Optional[str] = None
    age: Optional[str] = None          # experimental, from attributes module
    age_conf: float = 0.0
    engine_type: Optional[str] = None  # 2T / 4T / unknown (from audio)
    # vehicle attributes (cars)
    plate: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    vehicle_age: Optional[str] = None
    drivetrain: Optional[str] = None

    def update_motion(self, now: float) -> None:
        """Recompute speed and behavior from recent history."""
        self.history.append((now, self.x, self.y))
        # speed over a ~1s window
        ref = None
        for t, x, y in self.history:
            if now - t <= 1.0:
                ref = (t, x, y)
                break
        if ref is not None and now - ref[0] > 1e-3:
            self.speed = _dist(self.x, self.y, ref[1], ref[2]) / (now - ref[0])
        self.behavior = self._classify_behavior(now)

    def _classify_behavior(self, now: float) -> Optional[str]:
        if self.class_name == "person":
            if self.speed > RUN_SPEED:
                return "running"
            if self.speed > WALK_SPEED:
                return "walking"
            # standing for a while in a small area -> loitering
            if now - self.first_seen > LOITER_TIME_S:
                xs = [p[1] for p in self.history]
                ys = [p[2] for p in self.history]
                if xs and (max(xs) - min(xs)) < LOITER_RADIUS_M and \
                        (max(ys) - min(ys)) < LOITER_RADIUS_M:
                    return "loitering"
            return "standing"
        # vehicles / other: coarse moving vs stopped
        if self.class_name in ("car", "motorcycle", "truck", "bus", "bicycle",
                               "scooter"):
            return "moving" if self.speed > WALK_SPEED else "stopped"
        return None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "class": self.class_name,
            "class_id": self.class_id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "height": round(self.height, 2),
            "prob": round(self.confidence, 3),
            "cameras": self.cameras,
            "speed": round(self.speed, 2),
        }
        if self.behavior:
            d["behavior"] = self.behavior
        if self.age:
            d["age"] = self.age
            d["age_conf"] = round(self.age_conf, 2)
        if self.engine_type:
            d["engine_type"] = self.engine_type
        for k in ("plate", "make", "model", "vehicle_age", "drivetrain"):
            v = getattr(self, k)
            if v:
                d[k] = v
        return d


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


# Classes that are frequently confused between cameras (a silver car seen as
# car/truck/bus from different angles) are fused as one "vehicle" group so the
# same physical object does not show up multiple times.
CLASS_GROUPS = {
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
}
# Per-group merge radius multiplier (applied to the base merge distance).
GROUP_RADIUS_MULT = {"vehicle": 3.0, "person": 1.6}


def group_of(class_name: str) -> str:
    return CLASS_GROUPS.get(class_name, class_name)


@dataclass
class _Cluster:
    x: float
    y: float
    class_name: str
    class_id: int
    confidence: float
    cameras: list[str]
    group: str = ""


class Fusion:
    def __init__(self, merge_distance_m: float = 1.5, max_age_s: float = 2.0,
                 smoothing: float = 0.5, default_height_m: float = 1.7):
        self.merge_distance_m = merge_distance_m
        self.max_age_s = max_age_s
        self.smoothing = smoothing
        self.default_height_m = default_height_m
        self.tracks: dict[int, Track] = {}
        self._ids = count(1)

    def _radius(self, group: str) -> float:
        return self.merge_distance_m * GROUP_RADIUS_MULT.get(group, 1.0)

    # -- step 1: merge detections across cameras (by group, agglomerative) -
    def _cluster(self, detections: list[WorldDetection]) -> list[_Cluster]:
        # bucket detections by group
        groups: dict[str, list[WorldDetection]] = {}
        for det in detections:
            if math.isnan(det.x) or math.isnan(det.y):
                continue
            groups.setdefault(group_of(det.class_name), []).append(det)

        out: list[_Cluster] = []
        for grp, dets in groups.items():
            radius = self._radius(grp)
            members = [[d] for d in dets]  # start: each detection its own cluster
            cents = [(d.x, d.y) for d in dets]
            # single-linkage agglomerative: merge nearest clusters within radius
            merged = True
            while merged and len(members) > 1:
                merged = False
                bi = bj = -1
                bd = radius
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        d = _dist(*cents[i], *cents[j])
                        if d < bd:
                            bd, bi, bj = d, i, j
                if bi >= 0:
                    members[bi].extend(members[bj])
                    pts = members[bi]
                    cents[bi] = (sum(p.x for p in pts) / len(pts),
                                 sum(p.y for p in pts) / len(pts))
                    del members[bj]; del cents[bj]
                    merged = True

            for pts, (cx, cy) in zip(members, cents):
                # dominant class within the cluster by summed confidence
                by_cls: dict[str, float] = {}
                cid_for: dict[str, int] = {}
                for p in pts:
                    by_cls[p.class_name] = by_cls.get(p.class_name, 0.0) + p.confidence
                    cid_for[p.class_name] = p.class_id
                dom = max(by_cls, key=by_cls.get)
                out.append(_Cluster(
                    cx, cy, dom, cid_for[dom],
                    max(p.confidence for p in pts),
                    sorted({p.camera_id for p in pts}), grp))
        return out

    # -- step 2: associate clusters with tracks (by group) ----------------
    def update(self, detections: list[WorldDetection]) -> list[Track]:
        now = time.time()
        clusters = self._cluster(detections)

        unmatched = set(self.tracks.keys())
        for cluster in clusters:
            best_id: Optional[int] = None
            best_d = self._radius(cluster.group)
            for tid in unmatched:
                t = self.tracks[tid]
                if group_of(t.class_name) != cluster.group:
                    continue
                d = _dist(t.x, t.y, cluster.x, cluster.y)
                if d < best_d:
                    best_d = d
                    best_id = tid

            if best_id is not None:
                t = self.tracks[best_id]
                a = self.smoothing
                t.x = a * cluster.x + (1 - a) * t.x
                t.y = a * cluster.y + (1 - a) * t.y
                t.confidence = cluster.confidence
                # adopt the higher-confidence label within the group
                t.class_name = cluster.class_name
                t.class_id = cluster.class_id
                t.cameras = cluster.cameras
                t.last_update = now
                t.hits += 1
                t.update_motion(now)
                unmatched.discard(best_id)
            else:
                tid = next(self._ids)
                track = Track(
                    id=tid, class_name=cluster.class_name, class_id=cluster.class_id,
                    x=cluster.x, y=cluster.y, confidence=cluster.confidence,
                    height=height_for(cluster.class_name, self.default_height_m),
                    cameras=cluster.cameras, last_update=now, first_seen=now, hits=1,
                )
                track.update_motion(now)
                self.tracks[tid] = track

        self._dedupe_tracks()

        # -- step 3: drop stale tracks ------------------------------------
        for tid in list(self.tracks.keys()):
            if now - self.tracks[tid].last_update > self.max_age_s:
                del self.tracks[tid]

        return list(self.tracks.values())

    def _dedupe_tracks(self) -> None:
        """Collapse pre-existing tracks of the same group that drifted within
        the merge radius into a single track (keeps the longer-lived id)."""
        ids = sorted(self.tracks.keys())
        removed: set[int] = set()
        for i in range(len(ids)):
            if ids[i] in removed:
                continue
            a = self.tracks[ids[i]]
            for j in range(i + 1, len(ids)):
                if ids[j] in removed:
                    continue
                b = self.tracks[ids[j]]
                if group_of(a.class_name) != group_of(b.class_name):
                    continue
                if _dist(a.x, a.y, b.x, b.y) <= self._radius(group_of(a.class_name)):
                    # keep the one with more hits (older/stronger), drop other
                    keep, drop = (a, b) if a.hits >= b.hits else (b, a)
                    keep.cameras = sorted(set(keep.cameras) | set(drop.cameras))
                    keep.hits += drop.hits
                    if drop.confidence > keep.confidence:
                        keep.class_name = drop.class_name
                        keep.class_id = drop.class_id
                        keep.confidence = drop.confidence
                    removed.add(drop.id)
                    if drop.id == ids[i]:
                        break
        for rid in removed:
            self.tracks.pop(rid, None)

    def active(self) -> list[Track]:
        return list(self.tracks.values())
