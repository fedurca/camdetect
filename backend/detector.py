"""YOLO object detection wrapper.

Wraps ultralytics YOLO with:
- device auto-detection (CUDA when available, else CPU) and per-camera override,
- downscaled inference (the ``imgsz`` long-side handles big 2688x1512 frames),
- class filtering aligned with the cameras' people/vehicle/animal detections.

The same model instance can serve multiple cameras on CPU. On the future
multi-GPU rig, construct one :class:`Detector` per device (see ``device``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .classes import COCO_NAMES, canonical_name
from .config import DetectionConfig, OpenVocabConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single 2D detection in image pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    # bbox in pixels of the ORIGINAL frame: (x1, y1, x2, y2)
    bbox: tuple[float, float, float, float]

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center of the bbox - where the object meets the ground."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def resolve_device(requested: str) -> str:
    """Map config device strings to an ultralytics device id."""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


class Detector:
    """Thin wrapper around an ultralytics YOLO model."""

    def __init__(self, cfg: DetectionConfig, device: Optional[str] = None):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = resolve_device(device or cfg.device)
        logger.info("Loading model %s on %s", cfg.model, self.device)
        self.model = YOLO(cfg.model)
        # Names provided by the model (fallback to our COCO map).
        self._names = getattr(self.model, "names", None) or COCO_NAMES

    def name_for(self, class_id: int) -> str:
        name = None
        if isinstance(self._names, dict):
            name = self._names.get(class_id)
        return name or COCO_NAMES.get(class_id, str(class_id))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame. Returns detections in original px."""
        results = self.model.predict(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.confidence,
            classes=self.cfg.classes,
            device=self.device,
            verbose=False,
        )
        return self._parse(results)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Batched detection (used on GPU). Returns one list per input frame."""
        if not frames:
            return []
        results = self.model.predict(
            frames,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.confidence,
            classes=self.cfg.classes,
            device=self.device,
            verbose=False,
        )
        return [self._parse([r]) for r in results]

    def _parse(self, results) -> list[Detection]:
        out: list[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, clss):
                out.append(
                    Detection(
                        class_id=int(cid),
                        class_name=self.name_for(int(cid)),
                        confidence=float(conf),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
        return out


class OpenVocabDetector:
    """YOLO-World open-vocabulary detector driven by text prompts.

    Detects arbitrary classes (e.g. trash bin, scooter, drone) without training.
    Heavier than the COCO model, so it is opt-in.
    """

    def __init__(self, cfg: OpenVocabConfig, device: Optional[str] = None):
        from ultralytics import YOLOWorld

        self.cfg = cfg
        self.device = resolve_device(device or "auto")
        logger.info("Loading open-vocab model %s on %s", cfg.model, self.device)
        self.model = YOLOWorld(cfg.model)
        self.set_prompts(cfg.prompts)

    def set_prompts(self, prompts: list[str]) -> None:
        self.prompts = list(prompts)
        # YOLO-World maps class index -> prompt text.
        self.model.set_classes(self.prompts)

    def detect(self, frame: np.ndarray, imgsz: int,
               confidence: Optional[float] = None) -> list[Detection]:
        conf = self.cfg.confidence if confidence is None else confidence
        results = self.model.predict(
            frame, imgsz=imgsz, conf=conf, device=self.device, verbose=False,
        )
        out: list[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, clss):
                prompt = self.prompts[int(cid)] if int(cid) < len(self.prompts) else str(cid)
                out.append(
                    Detection(
                        class_id=1000 + int(cid),  # offset to avoid COCO id clash
                        class_name=canonical_name(prompt),
                        confidence=float(c),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
        return out


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def merge_detections(primary: list[Detection], extra: list[Detection],
                     iou_threshold: float = 0.6) -> list[Detection]:
    """Combine two detection lists, dropping ``extra`` boxes that strongly
    overlap a ``primary`` box (keeps the COCO result in case of duplicates)."""
    merged = list(primary)
    for e in extra:
        if any(_iou(e.bbox, p.bbox) > iou_threshold for p in primary):
            continue
        merged.append(e)
    return merged
