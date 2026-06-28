"""Camera geometry helpers.

Builds the shared intrinsic matrix K from the UniFi G5 Bullet field-of-view and
resolution, and provides small helpers for the world frame and (optional)
solvePnP-based extrinsics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import IntrinsicsConfig


def camera_coverage(cameras, default_fov_deg: float) -> dict:
    """Per-camera ground coverage wedge.

    Returns ``{cam_id: {x, y, azimuth_deg, fov_deg, range_m}}`` where the look
    azimuth is taken from config when given, otherwise derived to point from the
    camera toward the centroid of all cameras (they all watch the shared area).
    Azimuth is degrees CCW from world +X (east).
    """
    if not cameras:
        return {}
    cx = sum(c.world_xy[0] for c in cameras) / len(cameras)
    cy = sum(c.world_xy[1] for c in cameras) / len(cameras)
    out: dict = {}
    for c in cameras:
        x, y = c.world_xy
        if c.azimuth_deg is not None:
            az = float(c.azimuth_deg)
        else:
            az = math.degrees(math.atan2(cy - y, cx - x))
        out[c.id] = {
            "x": x, "y": y,
            "azimuth_deg": az,
            "fov_deg": float(c.fov_deg) if c.fov_deg else float(default_fov_deg),
            "range_m": float(c.range_m),
        }
    return out


def in_coverage(cov: dict, X: float, Y: float, fov_margin_deg: float = 12.0,
                range_margin: float = 1.3) -> bool:
    """True if world point (X, Y) is inside the camera's coverage wedge."""
    dx, dy = X - cov["x"], Y - cov["y"]
    dist = math.hypot(dx, dy)
    if dist > cov["range_m"] * range_margin:
        return False
    if dist < 1e-3:
        return True
    ang = math.degrees(math.atan2(dy, dx))
    diff = abs((ang - cov["azimuth_deg"] + 180.0) % 360.0 - 180.0)
    return diff <= cov["fov_deg"] / 2.0 + fov_margin_deg


def build_intrinsics(intr: IntrinsicsConfig) -> np.ndarray:
    """Return the 3x3 camera matrix K derived from FOV + resolution.

    fx = (W/2) / tan(HFOV/2), fy = (H/2) / tan(VFOV/2), principal point centered.
    """
    w, h = intr.width, intr.height
    fx = (w / 2.0) / math.tan(math.radians(intr.fov_horizontal_deg) / 2.0)
    fy = (h / 2.0) / math.tan(math.radians(intr.fov_vertical_deg) / 2.0)
    cx = w / 2.0
    cy = h / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def scale_intrinsics(K: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    """Scale K when an image is resized from ``from_size`` to ``to_size`` (w, h)."""
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


@dataclass
class Extrinsics:
    """Camera pose in the world frame (rotation + translation)."""

    rvec: np.ndarray  # (3,1) Rodrigues rotation
    tvec: np.ndarray  # (3,1) translation
    position: np.ndarray  # (3,) camera center in world coords

    @property
    def R(self) -> np.ndarray:
        import cv2

        R, _ = cv2.Rodrigues(self.rvec)
        return R


def solve_extrinsics(
    K: np.ndarray,
    image_points: np.ndarray,
    world_points: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Optional[Extrinsics]:
    """Estimate camera pose from >=4 image<->world 3D correspondences via PnP.

    ``image_points`` is (N,2) pixels, ``world_points`` is (N,3) meters.
    Returns ``None`` if the solve fails or there are too few points.
    """
    import cv2

    if image_points.shape[0] < 4 or image_points.shape[0] != world_points.shape[0]:
        return None
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        world_points.astype(np.float64),
        image_points.astype(np.float64),
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    position = (-R.T @ tvec).reshape(3)
    return Extrinsics(rvec=rvec, tvec=tvec, position=position)
