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
class OpenVocabConfig:
    """Open-vocabulary (YOLO-World) detector for classes not in COCO."""

    enabled: bool = False
    model: str = "models/yolov8s-world.pt"
    confidence: float = 0.10
    # Text prompts; mapped to canonical class names in classes.py.
    prompts: list[str] = field(default_factory=lambda: [
        "trash bin", "garbage can", "kick scooter", "roller skates",
        "inline skates", "drone", "quadcopter",
    ])


@dataclass
class AttributesConfig:
    """Per-person attribute estimation."""

    behavior: bool = True   # standing/walking/running/loitering (cheap)
    age: bool = False       # experimental, needs a visible face (off on CPU)


@dataclass
class DetectionConfig:
    enabled: bool = True
    device: str = "auto"
    model: str = "yolo11n.pt"
    imgsz: int = 960
    confidence: float = 0.35
    fps: float = 3.0
    classes: Optional[list[int]] = None
    open_vocabulary: OpenVocabConfig = field(default_factory=OpenVocabConfig)
    attributes: AttributesConfig = field(default_factory=AttributesConfig)


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
class AudioEventsConfig:
    """Sound-event classification (heavy model, off by default on CPU)."""

    enabled: bool = False
    model: str = "panns_cnn14"  # informational; classifier is pluggable


@dataclass
class AudioConfig:
    enabled: bool = True
    sample_rate: int = 16000
    window_s: float = 2.0      # analysis window length
    hop_s: float = 0.5         # how often we recompute
    fft_size: int = 1024
    hop_size: int = 256
    mel_bands: int = 96
    fmax: int = 8000
    spectrogram_width: int = 320
    spectrogram_height: int = 160
    engine_2t4t: bool = True   # experimental 2-stroke/4-stroke heuristic
    events: AudioEventsConfig = field(default_factory=AudioEventsConfig)


@dataclass
class DatabaseConfig:
    enabled: bool = True
    path: str = "data/camdetect.sqlite"
    # Two observations of the same class within this distance and time gap are
    # treated as the same physical object (merged).
    merge_distance_m: float = 3.0
    merge_time_s: float = 60.0
    retention_days: int = 30


@dataclass
class VehiclesConfig:
    """License-plate reading + make/model/age/drivetrain (experimental)."""

    enabled: bool = False
    plates: bool = True       # ANPR (needs OCR backend; experimental)
    make_model: bool = True   # make/model/age/drivetrain (needs a classifier)
    ocr_backend: str = "easyocr"  # informational; pluggable


@dataclass
class TranscriptionConfig:
    """Audio recording + Czech transcription + speaker diarization."""

    enabled: bool = False
    language: str = "cs"
    model: str = "small"      # faster-whisper model size
    diarization: bool = False
    record: bool = False
    record_dir: str = "data/audio"
    segment_s: float = 6.0    # transcription chunk length


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "data/logs/camdetect.log"
    max_lines: int = 2000     # in-memory ring buffer for the debug window


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
    audio: AudioConfig = field(default_factory=AudioConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    vehicles: VehiclesConfig = field(default_factory=VehiclesConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    path: str = DEFAULT_CONFIG_PATH

    def abspath(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

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

    det_raw = dict(raw.get("detection") or {})
    open_vocab = OpenVocabConfig(**(det_raw.pop("open_vocabulary", None) or {}))
    attributes = AttributesConfig(**(det_raw.pop("attributes", None) or {}))
    detection = DetectionConfig(open_vocabulary=open_vocab, attributes=attributes,
                                **det_raw)

    intrinsics = IntrinsicsConfig(**(raw.get("intrinsics") or {}))
    world = WorldConfig(**(raw.get("world") or {}))
    fusion = FusionConfig(**(raw.get("fusion") or {}))

    audio_raw = dict(raw.get("audio") or {})
    audio_events = AudioEventsConfig(**(audio_raw.pop("events", None) or {}))
    audio = AudioConfig(events=audio_events, **audio_raw)

    database = DatabaseConfig(**(raw.get("database") or {}))
    vehicles = VehiclesConfig(**(raw.get("vehicles") or {}))
    transcription = TranscriptionConfig(**(raw.get("transcription") or {}))
    logging_cfg = LoggingConfig(**(raw.get("logging") or {}))

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
        audio=audio,
        database=database,
        vehicles=vehicles,
        transcription=transcription,
        logging=logging_cfg,
        calibration=calibration,
        path=path,
    )
