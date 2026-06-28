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
from dataclasses import dataclass, field
from itertools import count
from typing import Optional

from .classes import height_for


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
    hits: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class": self.class_name,
            "class_id": self.class_id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "height": round(self.height, 2),
            "prob": round(self.confidence, 3),
            "cameras": self.cameras,
        }


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


@dataclass
class _Cluster:
    x: float
    y: float
    class_name: str
    class_id: int
    confidence: float
    cameras: list[str]


class Fusion:
    def __init__(self, merge_distance_m: float = 1.5, max_age_s: float = 2.0,
                 smoothing: float = 0.5, default_height_m: float = 1.7):
        self.merge_distance_m = merge_distance_m
        self.max_age_s = max_age_s
        self.smoothing = smoothing
        self.default_height_m = default_height_m
        self.tracks: dict[int, Track] = {}
        self._ids = count(1)

    # -- step 1: merge detections across cameras --------------------------
    def _cluster(self, detections: list[WorldDetection]) -> list[_Cluster]:
        clusters: list[_Cluster] = []
        members: list[list[WorldDetection]] = []
        for det in detections:
            if math.isnan(det.x) or math.isnan(det.y):
                continue
            placed = False
            for i, c in enumerate(clusters):
                # Only merge detections of the same class that are close.
                if c.class_name == det.class_name and \
                        _dist(c.x, c.y, det.x, det.y) <= self.merge_distance_m:
                    members[i].append(det)
                    placed = True
                    break
            if not placed:
                clusters.append(_Cluster(det.x, det.y, det.class_name,
                                         det.class_id, det.confidence, [det.camera_id]))
                members.append([det])

        # Recompute cluster centroids/aggregates from members.
        out: list[_Cluster] = []
        for mem in members:
            n = len(mem)
            cx = sum(m.x for m in mem) / n
            cy = sum(m.y for m in mem) / n
            conf = max(m.confidence for m in mem)
            cams = sorted({m.camera_id for m in mem})
            out.append(_Cluster(cx, cy, mem[0].class_name, mem[0].class_id, conf, cams))
        return out

    # -- step 2: associate clusters with tracks ---------------------------
    def update(self, detections: list[WorldDetection]) -> list[Track]:
        now = time.time()
        clusters = self._cluster(detections)

        unmatched = set(self.tracks.keys())
        for cluster in clusters:
            best_id: Optional[int] = None
            best_d = self.merge_distance_m * 1.5
            for tid in unmatched:
                t = self.tracks[tid]
                if t.class_name != cluster.class_name:
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
                t.cameras = cluster.cameras
                t.last_update = now
                t.hits += 1
                unmatched.discard(best_id)
            else:
                tid = next(self._ids)
                self.tracks[tid] = Track(
                    id=tid,
                    class_name=cluster.class_name,
                    class_id=cluster.class_id,
                    x=cluster.x,
                    y=cluster.y,
                    confidence=cluster.confidence,
                    height=height_for(cluster.class_name, self.default_height_m),
                    cameras=cluster.cameras,
                    last_update=now,
                    hits=1,
                )

        # -- step 3: drop stale tracks ------------------------------------
        for tid in list(self.tracks.keys()):
            if now - self.tracks[tid].last_update > self.max_age_s:
                del self.tracks[tid]

        return list(self.tracks.values())

    def active(self) -> list[Track]:
        return list(self.tracks.values())
