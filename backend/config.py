"""Configuration loading for camdetect.

Loads ``config.yaml`` from the project root into lightweight dataclasses so the
rest of the backend gets typed, attribute access instead of dict spelunking.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class DetectionConfig:
    device: str = "auto"
    model: str = "yolo11n.pt"
    imgsz: int = 960
    confidence: float = 0.35
    fps: float = 3.0
    classes: Optional[list[int]] = None


@dataclass
class IntrinsicsConfig:
    width: int = 2688
    height: int = 1512
    fov_horizontal_deg: float = 84.4
    fov_vertical_deg: float = 45.4


@dataclass
class WorldConfig:
    reference_edge_m: float = 14.6
    ground_elevation_m: float = 0.0


@dataclass
class CameraConfig:
    id: str
    url: str
    device: Optional[str] = None
    world_xy: tuple[float, float] = (0.0, 0.0)
    height_m: float = 3.0


@dataclass
class FusionConfig:
    merge_distance_m: float = 1.5
    max_age_s: float = 2.0
    smoothing: float = 0.5
    default_height_m: float = 1.7


@dataclass
class CalibrationConfig:
    dir: str = "data/calibration"


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    intrinsics: IntrinsicsConfig = field(default_factory=IntrinsicsConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    path: str = DEFAULT_CONFIG_PATH

    @property
    def calibration_dir(self) -> str:
        d = self.calibration.dir
        if not os.path.isabs(d):
            d = os.path.join(PROJECT_ROOT, d)
        return d

    def camera(self, cam_id: str) -> Optional[CameraConfig]:
        for cam in self.cameras:
            if cam.id == cam_id:
                return cam
        return None


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Config:
    """Load and parse the YAML config file into a :class:`Config`."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    server = ServerConfig(**(raw.get("server") or {}))
    detection = DetectionConfig(**(raw.get("detection") or {}))
    intrinsics = IntrinsicsConfig(**(raw.get("intrinsics") or {}))
    world = WorldConfig(**(raw.get("world") or {}))
    fusion = FusionConfig(**(raw.get("fusion") or {}))
    calibration = CalibrationConfig(**(raw.get("calibration") or {}))

    cameras: list[CameraConfig] = []
    for cam in raw.get("cameras") or []:
        cameras.append(
            CameraConfig(
                id=cam["id"],
                url=cam["url"],
                device=cam.get("device"),
                world_xy=tuple(cam.get("world_xy", (0.0, 0.0))),  # type: ignore[arg-type]
                height_m=float(cam.get("height_m", 3.0)),
            )
        )

    return Config(
        server=server,
        detection=detection,
        intrinsics=intrinsics,
        world=world,
        cameras=cameras,
        fusion=fusion,
        calibration=calibration,
        path=path,
    )
