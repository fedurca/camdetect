"""Detection + localization + fusion orchestration.

The pipeline produces two things the web UI consumes:
- annotated JPEG frames per camera (for the MJPEG stream panels), and
- a fused list of world-space objects (for the 3D scene + minimap).

Two sources are supported:
- ``live``: real RTSP capture + YOLO detection + homography localization.
- ``demo``: synthetic objects moving across the courtyard, projected into each
  camera view via the inverse homography and fed to fusion. Lets the whole UI
  run with no cameras and no GPU.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

from .calibration import CalibrationStore
from .cameras import CameraManager
from .classes import color_for
from .config import PROJECT_ROOT, Config
from .detector import Detection, Detector, resolve_device
from .fusion import Fusion, WorldDetection

logger = logging.getLogger(__name__)


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


def _encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def draw_detection(frame: np.ndarray, label: str, conf: float,
                   bbox: tuple[float, float, float, float],
                   rgb: tuple[int, int, int]) -> None:
    """Draw a labelled, colored detection box on ``frame`` (in place)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    color = _bgr(rgb)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)


class Pipeline:
    def __init__(self, cfg: Config, mode: str = "live"):
        self.cfg = cfg
        self.mode = mode
        self.calib = CalibrationStore(cfg.calibration_dir)
        self.calib.load_all([c.id for c in cfg.cameras])
        self.fusion = Fusion(
            merge_distance_m=cfg.fusion.merge_distance_m,
            max_age_s=cfg.fusion.max_age_s,
            smoothing=cfg.fusion.smoothing,
            default_height_m=cfg.fusion.default_height_m,
        )

        self._annotated: dict[str, bytes] = {}
        self._world_dets: dict[str, list[WorldDetection]] = {c.id: [] for c in cfg.cameras}
        self._objects: list[dict] = []
        self._status: dict[str, bool] = {c.id: False for c in cfg.cameras}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        if mode == "live":
            self.detectors = self._build_detectors()
            self.manager = CameraManager(cfg.cameras)
        else:
            self.base_frames = self._load_demo_frames()
            self._inv_homography = self._build_inverse_homographies()

    # -- setup -------------------------------------------------------------
    def _build_detectors(self) -> dict[str, Detector]:
        """One Detector per unique resolved device, shared across cameras."""
        by_device: dict[str, Detector] = {}
        mapping: dict[str, Detector] = {}
        for cam in self.cfg.cameras:
            dev = resolve_device(cam.device or self.cfg.detection.device)
            if dev not in by_device:
                by_device[dev] = Detector(self.cfg.detection, device=dev)
            mapping[cam.id] = by_device[dev]
        return mapping

    def _demo_image_size(self) -> tuple[int, int]:
        return (1920, 1080)

    def _load_demo_frames(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        w, h = self._demo_image_size()
        for cam in self.cfg.cameras:
            digit = "".join(ch for ch in cam.id if ch.isdigit()) or cam.id
            path = os.path.join(PROJECT_ROOT, f"{digit}.png")
            img = cv2.imread(path)
            if img is None:
                img = np.full((h, w, 3), 40, dtype=np.uint8)
            else:
                img = cv2.resize(img, (w, h))
            frames[cam.id] = img
        return frames

    def _build_inverse_homographies(self) -> dict[str, Optional[np.ndarray]]:
        inv: dict[str, Optional[np.ndarray]] = {}
        for cam in self.cfg.cameras:
            c = self.calib.get(cam.id)
            if c is None:
                inv[cam.id] = None
                continue
            try:
                inv[cam.id] = np.linalg.inv(c.homography)
            except np.linalg.LinAlgError:
                inv[cam.id] = None
        return inv

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self.mode == "live":
            self.manager.start()
            for cam in self.cfg.cameras:
                t = threading.Thread(target=self._camera_loop, args=(cam.id,),
                                     name=f"detect-{cam.id}", daemon=True)
                t.start()
                self._threads.append(t)
            tf = threading.Thread(target=self._fusion_loop, name="fusion", daemon=True)
            tf.start()
            self._threads.append(tf)
        else:
            t = threading.Thread(target=self._demo_loop, name="demo", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        if self.mode == "live":
            self.manager.stop()
        for t in self._threads:
            t.join(timeout=3.0)

    # -- live path ---------------------------------------------------------
    def _camera_loop(self, cam_id: str) -> None:
        stream = self.manager.get(cam_id)
        detector = self.detectors[cam_id]
        calib = self.calib.get(cam_id)
        interval = 1.0 / max(self.cfg.detection.fps, 0.1)
        last_seq = -1
        while not self._stop.is_set():
            start = time.time()
            frame, seq, _ = stream.read() if stream else (None, 0, 0.0)
            self._status[cam_id] = bool(stream and stream.connected)
            if frame is None or seq == last_seq:
                self._stop.wait(0.05)
                continue
            last_seq = seq
            h, w = frame.shape[:2]
            dets = detector.detect(frame)
            world: list[WorldDetection] = []
            for d in dets:
                draw_detection(frame, d.class_name, d.confidence, d.bbox,
                               color_for(d.class_name))
                if calib is not None:
                    fx, fy = d.foot_point
                    X, Y = calib.image_to_world(fx, fy, (w, h))
                    world.append(WorldDetection(cam_id, d.class_id, d.class_name,
                                                d.confidence, X, Y))
            with self._lock:
                self._annotated[cam_id] = _encode_jpeg(frame)
                self._world_dets[cam_id] = world
            elapsed = time.time() - start
            if elapsed < interval:
                self._stop.wait(interval - elapsed)

    def _fusion_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                dets: list[WorldDetection] = []
                for d in self._world_dets.values():
                    dets.extend(d)
            tracks = self.fusion.update(dets)
            with self._lock:
                self._objects = [t.to_dict() for t in tracks]
            self._stop.wait(0.1)

    # -- demo path ---------------------------------------------------------
    def _synthetic_objects(self, t: float) -> list[tuple[str, int, float, float, float]]:
        """Return synthetic (class, class_id, X, Y, conf) moving over the courtyard."""
        objs = []
        # A car driving back and forth along the road (y ~ 1.5).
        cx = 2.0 + (math.sin(t * 0.4) * 0.5 + 0.5) * 10.0
        objs.append(("car", 2, cx, 1.8, 0.86 + 0.05 * math.sin(t)))
        # A person walking a circle around the courtyard center.
        px = 7.0 + 2.5 * math.cos(t * 0.6)
        py = 4.5 + 2.5 * math.sin(t * 0.6)
        objs.append(("person", 0, px, py, 0.78))
        # A parked truck.
        objs.append(("truck", 7, 11.0, 6.5, 0.69))
        return objs

    def _demo_loop(self) -> None:
        fps = 12.0
        interval = 1.0 / fps
        t0 = time.time()
        w, h = self._demo_image_size()
        for cid in self._status:
            self._status[cid] = True
        while not self._stop.is_set():
            t = time.time() - t0
            objs = self._synthetic_objects(t)
            world_by_cam: dict[str, list[WorldDetection]] = {c.id: [] for c in self.cfg.cameras}
            frames = {cid: img.copy() for cid, img in self.base_frames.items()}

            for cam in self.cfg.cameras:
                Hinv = self._inv_homography.get(cam.id)
                if Hinv is None:
                    continue
                for (name, cid, X, Y, conf) in objs:
                    p = Hinv @ np.array([X, Y, 1.0])
                    if abs(p[2]) < 1e-9:
                        continue
                    u, v = p[0] / p[2], p[1] / p[2]
                    if not (0 <= u < w and 0 <= v < h):
                        continue  # object not visible to this camera
                    box_h = {"person": 90, "car": 70, "truck": 110}.get(name, 80)
                    box_w = {"person": 40, "car": 130, "truck": 150}.get(name, 70)
                    bbox = (u - box_w / 2, v - box_h, u + box_w / 2, v)
                    draw_detection(frames[cam.id], name, conf, bbox, color_for(name))
                    # small per-camera noise to exercise fusion
                    nx = X + np.random.uniform(-0.2, 0.2)
                    ny = Y + np.random.uniform(-0.2, 0.2)
                    world_by_cam[cam.id].append(
                        WorldDetection(cam.id, cid, name, conf, nx, ny))

            dets = [d for lst in world_by_cam.values() for d in lst]
            tracks = self.fusion.update(dets)
            with self._lock:
                for cid, fr in frames.items():
                    self._annotated[cid] = _encode_jpeg(fr)
                self._objects = [t.to_dict() for t in tracks]
            self._stop.wait(interval)

    # -- accessors ---------------------------------------------------------
    def latest_jpeg(self, cam_id: str) -> Optional[bytes]:
        with self._lock:
            return self._annotated.get(cam_id)

    def get_state(self) -> dict:
        with self._lock:
            return {
                "objects": list(self._objects),
                "cameras": {c.id: self._status.get(c.id, False) for c in self.cfg.cameras},
                "ts": time.time(),
            }
