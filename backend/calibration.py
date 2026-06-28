"""Load and apply per-camera calibration.

A calibration file (``data/calibration/<cam>.json``) maps image pixels to the
shared world ground plane (meters) via a 3x3 homography, and records the
camera's world position/height plus optional PnP extrinsics. This module is the
read/apply side; the interactive creation tool lives in ``calibrate/``.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraCalibration:
    camera_id: str
    image_size: tuple[int, int]              # (w, h) the homography was built for
    homography: np.ndarray                    # 3x3 image px -> world meters
    world_position: tuple[float, float, float]
    intrinsics: Optional[np.ndarray] = None   # 3x3 K (scaled to image_size)
    extrinsics: Optional[dict] = None         # {"rvec": [...], "tvec": [...]}
    extra: dict = field(default_factory=dict)

    # -- IO ----------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibration":
        K = d.get("intrinsics")
        return cls(
            camera_id=d["camera_id"],
            image_size=tuple(d["image_size"]),  # type: ignore[arg-type]
            homography=np.array(d["homography"], dtype=np.float64),
            world_position=tuple(d.get("world_position", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
            intrinsics=np.array(K, dtype=np.float64) if K else None,
            extrinsics=d.get("extrinsics"),
            extra=d.get("extra", {}),
        )

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "image_size": list(self.image_size),
            "homography": self.homography.tolist(),
            "world_position": list(self.world_position),
            "intrinsics": self.intrinsics.tolist() if self.intrinsics is not None else None,
            "extrinsics": self.extrinsics,
            "extra": self.extra,
        }

    @classmethod
    def load(cls, path: str) -> "CameraCalibration":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    # -- projection --------------------------------------------------------
    def image_to_world(self, x: float, y: float,
                       frame_size: Optional[tuple[int, int]] = None) -> tuple[float, float]:
        """Map an image pixel (foot point) to world ground (X, Y) in meters.

        ``frame_size`` is the size of the frame the pixel came from; if it
        differs from the calibration ``image_size`` the point is rescaled first.
        """
        if frame_size is not None and frame_size != self.image_size:
            sx = self.image_size[0] / frame_size[0]
            sy = self.image_size[1] / frame_size[1]
            x, y = x * sx, y * sy
        p = self.homography @ np.array([x, y, 1.0])
        if abs(p[2]) < 1e-9:
            return (float("nan"), float("nan"))
        return (float(p[0] / p[2]), float(p[1] / p[2]))


class CalibrationStore:
    """Loads all available camera calibrations from a directory."""

    def __init__(self, directory: str):
        self.directory = directory
        self.calibrations: dict[str, CameraCalibration] = {}

    def load_all(self, camera_ids: list[str]) -> dict[str, CameraCalibration]:
        for cam_id in camera_ids:
            path = os.path.join(self.directory, f"{cam_id}.json")
            if os.path.exists(path):
                try:
                    self.calibrations[cam_id] = CameraCalibration.load(path)
                    logger.info("Loaded calibration for %s", cam_id)
                except Exception as exc:  # pragma: no cover
                    logger.error("Failed to load calibration %s: %s", path, exc)
            else:
                logger.warning("No calibration for %s (%s missing)", cam_id, path)
        return self.calibrations

    def get(self, cam_id: str) -> Optional[CameraCalibration]:
        return self.calibrations.get(cam_id)

    def __len__(self) -> int:
        return len(self.calibrations)
