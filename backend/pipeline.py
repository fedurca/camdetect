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

from .audio import AudioManager
from .calibration import CalibrationStore
from .cameras import CameraManager
from .classes import color_for, label_cs
from .config import PROJECT_ROOT, Config
from .db import Database
from .detector import (Detection, Detector, OpenVocabDetector, cuda_devices,
                       merge_detections, resolve_device)
from .fusion import Fusion, WorldDetection
from .geometry import camera_coverage, in_coverage
from .recorder import RecordingManager
from .report import ReportManager
from .settings import Settings
from .transcribe import TranscriptionManager
from .vehicles import VehicleAnalyzer, VehicleInfo

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {"car", "truck", "bus"}


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
    def __init__(self, cfg: Config, mode: str = "live",
                 settings: Optional[Settings] = None):
        self.cfg = cfg
        self.mode = mode
        self.settings = settings or Settings(cfg)
        self.calib = CalibrationStore(cfg.calibration_dir)
        self.calib.load_all([c.id for c in cfg.cameras])
        self.fusion = Fusion(
            merge_distance_m=cfg.fusion.merge_distance_m,
            max_age_s=cfg.fusion.max_age_s,
            smoothing=cfg.fusion.smoothing,
            default_height_m=cfg.fusion.default_height_m,
            min_cameras=cfg.fusion.min_cameras,
            min_hits=cfg.fusion.min_hits,
            confirm_window_s=cfg.fusion.confirm_window_s,
        )

        self._annotated: dict[str, bytes] = {}
        self._world_dets: dict[str, list[WorldDetection]] = {c.id: [] for c in cfg.cameras}
        self._objects: list[dict] = []
        self._status: dict[str, bool] = {c.id: False for c in cfg.cameras}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        # Open-vocab detectors are built lazily on first use (heavy weights).
        self._open_vocab: dict[str, Optional[OpenVocabDetector]] = {}
        self._open_vocab_prompts: list[str] = []

        # Audio subsystem (runs in both live and demo).
        self.audio = AudioManager(cfg.cameras, cfg.audio, self.settings, mode=mode)

        # Local database (objects + events).
        self.db: Optional[Database] = None
        if cfg.database.enabled:
            self.db = Database(cfg.abspath(cfg.database.path),
                               merge_distance_m=cfg.database.merge_distance_m,
                               merge_time_s=cfg.database.merge_time_s,
                               retention_days=cfg.database.retention_days)

        # Speaker->person linking (acoustic localization across the 3 mics).
        self._cam_pos = {c.id: (c.world_xy[0], c.world_xy[1]) for c in cfg.cameras}
        self._speech_by_obj: dict[int, dict] = {}

        # Camera coverage wedges (from the UniFi coverage map) - used to reject
        # detections projected outside a camera's real field of view.
        self._coverage = camera_coverage(cfg.cameras, cfg.intrinsics.fov_horizontal_deg)

        # Transcription (uses the audio ring buffers); links speakers to people.
        self.transcription = TranscriptionManager(
            self.audio, cfg.transcription, self.settings, mode=mode, db=self.db,
            locate_speaker=self._locate_speaker)

        # Vehicle analyzer (ANPR + make/model) and recent observations buffer.
        self.vehicle = VehicleAnalyzer(cfg.vehicles.ocr_backend)
        self._vehicle_obs: list[tuple[float, float, VehicleInfo, float]] = []

        # Video sample recorder (MKV + events subtitle track).
        self.recorder = RecordingManager(
            cfg, mode, get_jpeg=self.latest_jpeg, get_state=self.get_state,
            url_for=lambda cid: (self.cfg.camera(cid).url if self.cfg.camera(cid) else None))

        # Daily report (snapshots + text summary).
        self.report = ReportManager(cfg, get_jpeg=self.latest_jpeg,
                                    cameras=cfg.cameras, db=self.db)

        if mode == "live":
            self.detectors = self._build_detectors()
            self.manager = CameraManager(cfg.cameras)
        else:
            self.base_frames = self._load_demo_frames()
            self._inv_homography = self._build_inverse_homographies()

    # -- setup -------------------------------------------------------------
    def _device_for(self, cam, gpus: list[str], idx: int) -> str:
        """Resolve a camera's device, spreading cameras across all GPUs.

        Explicit per-camera ``device`` wins. Otherwise, when the global device
        is ``auto`` and multiple CUDA GPUs exist, cameras are assigned
        round-robin (cuda:0, cuda:1, ...) so the load is split across both GPUs.
        """
        if cam.device:
            return cam.device
        if self.cfg.detection.device == "auto" and gpus:
            return gpus[idx % len(gpus)]
        return resolve_device(self.cfg.detection.device)

    def _build_detectors(self) -> dict[str, Detector]:
        """One Detector per unique resolved device, shared across cameras."""
        gpus = cuda_devices()
        if gpus:
            logger.info("CUDA GPUs available: %s", ", ".join(gpus))
        else:
            logger.info("No CUDA GPU - running detection on CPU")
        by_device: dict[str, Detector] = {}
        mapping: dict[str, Detector] = {}
        for idx, cam in enumerate(self.cfg.cameras):
            dev = self._device_for(cam, gpus, idx)
            if dev not in by_device:
                by_device[dev] = Detector(self.cfg.detection, device=dev)
            mapping[cam.id] = by_device[dev]
            logger.info("Camera %s -> %s", cam.id, dev)
        return mapping

    def _get_open_vocab(self, device: str, prompts: list[str]) -> Optional[OpenVocabDetector]:
        """Lazily build/cached YOLO-World detector per device; refresh prompts."""
        det = self._open_vocab.get(device)
        if det is None:
            try:
                det = OpenVocabDetector(self.cfg.detection.open_vocabulary,
                                        device=device)
            except Exception as exc:
                logger.error("open-vocab load failed: %s", exc)
                self._open_vocab[device] = None
                return None
            self._open_vocab[device] = det
        if prompts and prompts != self._open_vocab_prompts:
            try:
                det.set_prompts(prompts)
                self._open_vocab_prompts = list(prompts)
            except Exception as exc:  # pragma: no cover
                logger.warning("set_prompts failed: %s", exc)
        return det

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
        if self.db is not None:
            self.db.start()
        self.audio.start()
        self.transcription.start()
        self.report.start()
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
        self.report.stop()
        self.recorder.stop_all()
        self.transcription.stop()
        self.audio.stop()
        if self.db is not None:
            self.db.stop()
        if self.mode == "live":
            self.manager.stop()
        for t in self._threads:
            t.join(timeout=3.0)

    # -- live path ---------------------------------------------------------
    def _camera_loop(self, cam_id: str) -> None:
        stream = self.manager.get(cam_id)
        detector = self.detectors[cam_id]
        calib = self.calib.get(cam_id)
        last_seq = -1
        while not self._stop.is_set():
            start = time.time()
            vs = self.settings.get("video", default={}) or {}
            enabled = vs.get("enabled", True)
            fps = float(vs.get("fps", self.cfg.detection.fps))
            interval = 1.0 / max(fps, 0.1)

            frame, seq, _ = stream.read() if stream else (None, 0, 0.0)
            self._status[cam_id] = bool(stream and stream.connected)

            if not enabled:
                # detection off: still show the raw frame, clear world dets
                if frame is not None and seq != last_seq:
                    last_seq = seq
                    with self._lock:
                        self._annotated[cam_id] = _encode_jpeg(frame)
                        self._world_dets[cam_id] = []
                self._stop.wait(0.1)
                continue

            if frame is None or seq == last_seq:
                self._stop.wait(0.05)
                continue
            last_seq = seq
            h, w = frame.shape[:2]

            imgsz = int(vs.get("imgsz", self.cfg.detection.imgsz))
            conf = float(vs.get("confidence", self.cfg.detection.confidence))
            detector.cfg.imgsz = imgsz
            detector.cfg.confidence = conf
            dets = detector.detect(frame)
            dets = self._maybe_open_vocab(detector.device, frame, imgsz, dets, vs)

            veh = self.settings.get("vehicles", default={}) or {}
            world: list[WorldDetection] = []
            for d in dets:
                draw_detection(frame, label_cs(d.class_name), d.confidence, d.bbox,
                               color_for(d.class_name))
                if calib is not None:
                    fx, fy = d.foot_point
                    X, Y = calib.image_to_world(fx, fy, (w, h))
                    cov = self._coverage.get(cam_id)
                    # Reject projections that fall outside this camera's real
                    # coverage wedge (usually a homography/labeling error) -
                    # improves cross-camera fusion precision.
                    if cov is not None and not in_coverage(cov, X, Y):
                        continue
                    world.append(WorldDetection(cam_id, d.class_id, d.class_name,
                                                d.confidence, X, Y))
                    if veh.get("enabled") and d.class_name in VEHICLE_CLASSES:
                        self._analyze_vehicle(frame, d, X, Y, veh)
            with self._lock:
                self._annotated[cam_id] = _encode_jpeg(frame)
                self._world_dets[cam_id] = world
            elapsed = time.time() - start
            if elapsed < interval:
                self._stop.wait(interval - elapsed)

    def _maybe_open_vocab(self, device: str, frame: np.ndarray, imgsz: int,
                          dets: list[Detection], vs: dict) -> list[Detection]:
        ov = vs.get("open_vocabulary", {}) or {}
        if not ov.get("enabled"):
            return dets
        prompts = ov.get("prompts") or self.cfg.detection.open_vocabulary.prompts
        det = self._get_open_vocab(device, prompts)
        if det is None:
            return dets
        try:
            extra = det.detect(frame, imgsz, ov.get("confidence"))
            return merge_detections(dets, extra)
        except Exception as exc:  # pragma: no cover
            logger.warning("open-vocab detect failed: %s", exc)
            return dets

    def _fusion_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                dets: list[WorldDetection] = []
                for d in self._world_dets.values():
                    dets.extend(d)
            self.fusion.min_cameras = int(self.settings.get(
                "video", "min_cameras", default=self.cfg.fusion.min_cameras))
            tracks = self.fusion.update(dets)
            self._attach_engine_type(tracks)
            if (self.settings.get("vehicles", "enabled", default=False)):
                self._apply_vehicles(tracks)
            objs = [t.to_dict() for t in tracks]
            self._attach_speech(objs)
            with self._lock:
                self._objects = objs
            if self.db is not None and self.settings.get("database", "enabled", default=True):
                self.db.ingest(objs)
            self._stop.wait(0.1)

    def _analyze_vehicle(self, frame: np.ndarray, d: Detection, X: float, Y: float,
                         veh: dict) -> None:
        """Run ANPR / make-model on a car crop; buffer the result by world pos."""
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return
        info = VehicleInfo()
        try:
            if veh.get("plates") and self.vehicle.plates_available:
                pr = self.vehicle.read_plate(crop)
                if pr:
                    info.plate, info.plate_conf = pr
            if veh.get("make_model"):
                mm = self.vehicle.estimate_make_model(crop)
                info.make, info.model = mm.make, mm.model
                info.vehicle_age, info.drivetrain = mm.vehicle_age, mm.drivetrain
        except Exception as exc:  # pragma: no cover
            logger.debug("vehicle analyze failed: %s", exc)
            return
        if info.to_attrs():
            self._vehicle_obs.append((X, Y, info, time.time()))
            self._vehicle_obs = self._vehicle_obs[-50:]

    def _apply_vehicles(self, tracks) -> None:
        """Attach buffered vehicle info to the nearest car track."""
        now = time.time()
        obs = [o for o in self._vehicle_obs if now - o[3] < 5.0]
        for t in tracks:
            if t.class_name not in VEHICLE_CLASSES:
                continue
            best = None
            best_d = 4.0
            for (ox, oy, info, _ts) in obs:
                dd = math.hypot(t.x - ox, t.y - oy)
                if dd < best_d:
                    best_d, best = dd, info
            if best:
                t.plate = best.plate or t.plate
                t.make = best.make or t.make
                t.model = best.model or t.model
                t.vehicle_age = best.vehicle_age or t.vehicle_age
                t.drivetrain = best.drivetrain or t.drivetrain

    @staticmethod
    def _apply_demo_vehicles(tracks) -> None:
        """Synthetic, stable vehicle attributes per car track (demo only)."""
        makes = [("Skoda", "Octavia", "~2019", "diesel"),
                 ("Skoda", "Fabia", "~2015", "benzin"),
                 ("Tesla", "Model 3", "~2022", "elektro"),
                 ("VW", "Golf", "~2017", "benzin")]
        for t in tracks:
            if t.class_name in VEHICLE_CLASSES:
                mk, md, ag, dt = makes[t.id % len(makes)]
                t.make, t.model, t.vehicle_age, t.drivetrain = mk, md, ag, dt
                if not t.plate:
                    t.plate = f"{t.id % 9 + 1}{chr(65 + t.id % 26)}{t.id % 9} {1000 + t.id % 9000}"

    # -- speaker -> person linking (variant 2: acoustic localization) ------
    def _locate_speaker(self, cam_id: str, segment: dict) -> Optional[dict]:
        """Match a transcript segment to the most likely person track.

        Uses the speech-band loudness measured at each of the three camera
        microphones together with the known camera world positions: for each
        candidate person we predict the per-camera loudness from inverse-square
        distance and pick the person whose predicted pattern best matches the
        observed one (cosine similarity). Also returns an energy-weighted source
        position estimate. No sample-level sync needed - it is a level-pattern
        localization, which is robust over independent RTSP audio streams.
        """
        audio = self.audio.results()
        cams = [c.id for c in self.cfg.cameras]
        e_obs = np.array([float(audio.get(c, {}).get("speech_level", 0.0)) for c in cams])
        if e_obs.sum() <= 1e-6:
            return None

        with self._lock:
            persons = [o for o in self._objects if o.get("class") == "person"]
        if not persons:
            return None

        # energy-weighted source position estimate (for display/logging)
        w = e_obs / e_obs.sum()
        sx = sum(w[i] * self._cam_pos[c][0] for i, c in enumerate(cams))
        sy = sum(w[i] * self._cam_pos[c][1] for i, c in enumerate(cams))

        def cosine(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0

        best, best_score = None, -1.0
        for p in persons:
            pred = np.array([1.0 / (((p["x"] - self._cam_pos[c][0]) ** 2 +
                                     (p["y"] - self._cam_pos[c][1]) ** 2) + 1.0)
                             for c in cams])
            score = cosine(e_obs, pred)
            if score > best_score:
                best_score, best = score, p

        if best is None or (len(persons) > 1 and best_score < 0.5):
            return None

        track_id = best["id"]
        self._speech_by_obj[track_id] = {
            "text": segment.get("text", ""),
            "speaker": segment.get("speaker", "S1"),
            "ts": time.time(),
            "score": round(best_score, 2),
        }
        return {"track_id": track_id, "score": round(best_score, 2),
                "x": round(sx, 2), "y": round(sy, 2)}

    def _attach_speech(self, objs: list[dict], max_age_s: float = 6.0) -> None:
        """Annotate object dicts with the latest transcript linked to them."""
        now = time.time()
        for o in objs:
            sp = self._speech_by_obj.get(o["id"])
            if sp and now - sp["ts"] <= max_age_s:
                o["speech"] = sp["text"]
                o["speaker"] = sp["speaker"]

    def _attach_engine_type(self, tracks) -> None:
        """Tag motorcycle/car tracks with the 2T/4T type from audio, if any."""
        engine = None
        for res in self.audio.results().values():
            et = res.get("engine_type")
            if et in ("2T", "4T"):
                engine = et
                break
        if engine is None:
            return
        for t in tracks:
            if t.class_name in ("motorcycle", "car"):
                t.engine_type = engine

    # -- demo path ---------------------------------------------------------
    def _synthetic_objects(self, t: float, include_open_vocab: bool
                           ) -> list[tuple[str, int, float, float, float]]:
        """Return synthetic (class, class_id, X, Y, conf) over the courtyard."""
        objs = []
        # A car driving back and forth along the road (y ~ 1.5).
        cx = 2.0 + (math.sin(t * 0.4) * 0.5 + 0.5) * 10.0
        objs.append(("car", 2, cx, 1.8, 0.86 + 0.05 * math.sin(t)))
        # A person walking a circle around the courtyard center.
        px = 7.0 + 2.5 * math.cos(t * 0.6)
        py = 4.5 + 2.5 * math.sin(t * 0.6)
        objs.append(("person", 0, px, py, 0.78))
        # A motorcycle crossing (engine_type comes from the audio analyzer).
        mx = 12.0 - (math.sin(t * 0.5) * 0.5 + 0.5) * 9.0
        objs.append(("motorcycle", 3, mx, 2.6, 0.71))
        # A dog trotting near the person.
        objs.append(("dog", 16, px + 1.2, py + 0.6, 0.6))
        if include_open_vocab:
            objs.append(("trash bin", 1000, 10.8, 6.6, 0.55))   # static
            sx = 3.0 + (math.cos(t * 0.7) * 0.5 + 0.5) * 6.0
            objs.append(("scooter", 1001, sx, 5.5, 0.5))
            objs.append(("drone", 1003, 7.0 + 3 * math.cos(t * 0.9), 1.0, 0.45))
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
            vs = self.settings.get("video", default={}) or {}
            attrs = self.settings.get("attributes", default={}) or {}
            video_on = vs.get("enabled", True)
            ov_on = (vs.get("open_vocabulary", {}) or {}).get("enabled", False)

            frames = {cid: img.copy() for cid, img in self.base_frames.items()}
            world_by_cam: dict[str, list[WorldDetection]] = {c.id: [] for c in self.cfg.cameras}

            if video_on:
                objs = self._synthetic_objects(t, include_open_vocab=ov_on)
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
                        box_h = {"person": 90, "car": 70, "truck": 110,
                                 "motorcycle": 80, "drone": 40}.get(name, 70)
                        box_w = {"person": 40, "car": 130, "truck": 150,
                                 "motorcycle": 60, "trash bin": 50}.get(name, 70)
                        bbox = (u - box_w / 2, v - box_h, u + box_w / 2, v)
                        draw_detection(frames[cam.id], label_cs(name), conf, bbox,
                                       color_for(name))
                        nx = X + np.random.uniform(-0.2, 0.2)
                        ny = Y + np.random.uniform(-0.2, 0.2)
                        world_by_cam[cam.id].append(
                            WorldDetection(cam.id, cid, name, conf, nx, ny))

            dets = [d for lst in world_by_cam.values() for d in lst]
            self.fusion.min_cameras = int(self.settings.get(
                "video", "min_cameras", default=self.cfg.fusion.min_cameras))
            tracks = self.fusion.update(dets)
            self._attach_engine_type(tracks)
            if attrs.get("age"):
                self._apply_demo_age(tracks)
            if (self.settings.get("vehicles", "enabled", default=False)):
                self._apply_demo_vehicles(tracks)
            objs = [tr.to_dict() for tr in tracks]
            self._attach_speech(objs)
            with self._lock:
                for cid, fr in frames.items():
                    self._annotated[cid] = _encode_jpeg(fr)
                self._objects = objs
            if self.db is not None and self.settings.get("database", "enabled", default=True):
                self.db.ingest(objs)
            self._stop.wait(interval)

    @staticmethod
    def _apply_demo_age(tracks) -> None:
        """Populate a synthetic age bucket on person tracks (demo only)."""
        buckets = ["child", "teen", "adult", "senior"]
        for tr in tracks:
            if tr.class_name == "person":
                tr.age = buckets[tr.id % len(buckets)]
                tr.age_conf = 0.55

    # -- accessors ---------------------------------------------------------
    def latest_jpeg(self, cam_id: str) -> Optional[bytes]:
        with self._lock:
            return self._annotated.get(cam_id)

    def spectrogram_jpeg(self, cam_id: str) -> Optional[bytes]:
        return self.audio.spectrogram_jpeg(cam_id)

    def get_state(self) -> dict:
        with self._lock:
            objects = list(self._objects)
            cameras = {c.id: self._status.get(c.id, False) for c in self.cfg.cameras}
        return {
            "objects": objects,
            "cameras": cameras,
            "audio": self.audio.results(),
            "transcripts": self.transcription.results(),
            "ts": time.time(),
        }

    def benchmark(self, frames: int = 12) -> dict:
        """Report compute device and a quick inference FPS estimate.

        Reuses an existing detector (never builds a second model in the request
        thread) and runs under the detector's lock so it can't race the camera
        threads - which previously caused a segfault.
        """
        info: dict = {"cuda": False, "device": "cpu", "gpus": []}
        try:
            import torch
            info["torch"] = torch.__version__
            if torch.cuda.is_available():
                info["cuda"] = True
                info["device"] = "cuda:0"
                info["gpus"] = [torch.cuda.get_device_name(i)
                                for i in range(torch.cuda.device_count())]
        except Exception:  # pragma: no cover
            pass

        # Only benchmark an already-loaded detector to avoid loading a second
        # model concurrently (a common crash source). In demo there is none.
        det = None
        if getattr(self, "detectors", None):
            det = self.detectors.get(self.cfg.cameras[0].id)
        if det is None:
            info["note"] = ("detekce neni aktivni (demo/vypnuto) - spust v 'live'"
                            " rezimu pro mereni modelu")
            return info

        try:
            imgsz = int(self.settings.get("video", "imgsz", default=self.cfg.detection.imgsz))
            det.cfg.imgsz = imgsz
            dummy = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            det.detect(dummy)  # warmup (locked internally)
            t0 = time.time()
            for _ in range(max(1, frames)):
                det.detect(dummy)
            dt = time.time() - t0
            fps = frames / dt if dt > 0 else 0.0
            info["model"] = self.cfg.detection.model
            info["imgsz"] = imgsz
            info["fps_single"] = round(fps, 2)
            info["latency_ms"] = round(dt / frames * 1000, 1)
            info["suggested_fps"] = round(min(fps / max(1, len(self.cfg.cameras)), 30), 1)
        except Exception as exc:  # pragma: no cover
            info["error"] = str(exc)
        return info
