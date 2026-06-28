"""Vehicle analysis: license plate reading + make/model/age/drivetrain.

This is experimental and modular:

- License plate (ANPR): a real solution detects the plate region then runs OCR.
  We optionally use ``easyocr`` if installed (CPU-capable but slow); the plate
  region is taken as the lower-center of the car bbox. Czech plates match the
  pattern ``\\d[A-Z]\\d \\d{4}`` (e.g. ``1AB 2345``).
- Make / model / age / drivetrain: a real solution uses a fine-grained vehicle
  classifier (e.g. a CNN trained on a make/model dataset) plus a plate->registry
  lookup for age/drivetrain. No such model ships here, so live results are
  ``unknown`` unless a model is plugged in.

In demo mode the pipeline injects synthetic, stable values per track so the UI
and database paths are exercised without any heavy models.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CZ_PLATE_RE = re.compile(r"^[0-9][A-Z][0-9]\s?[A-Z0-9]{4}$")


@dataclass
class VehicleInfo:
    plate: Optional[str] = None
    plate_conf: float = 0.0
    make: Optional[str] = None
    model: Optional[str] = None
    vehicle_age: Optional[str] = None     # e.g. "~2018" or "10+ let"
    drivetrain: Optional[str] = None      # benzin / diesel / elektro / hybrid

    def to_attrs(self) -> dict:
        d = {}
        if self.plate:
            d["plate"] = self.plate
        if self.make:
            d["make"] = self.make
        if self.model:
            d["model"] = self.model
        if self.vehicle_age:
            d["vehicle_age"] = self.vehicle_age
        if self.drivetrain:
            d["drivetrain"] = self.drivetrain
        return d


class VehicleAnalyzer:
    """Reads plates / estimates make-model. Backends are optional + lazy."""

    def __init__(self, ocr_backend: str = "easyocr"):
        self.ocr_backend = ocr_backend
        self._ocr = None
        self._ocr_tried = False

    def _ensure_ocr(self):
        if self._ocr_tried:
            return self._ocr
        self._ocr_tried = True
        try:  # pragma: no cover - depends on optional dependency
            if self.ocr_backend == "easyocr":
                import easyocr
                self._ocr = easyocr.Reader(["en"], gpu=False)
                logger.info("easyocr ANPR backend ready")
        except Exception as exc:  # pragma: no cover
            logger.warning("ANPR backend unavailable (%s); plates disabled", exc)
            self._ocr = None
        return self._ocr

    @property
    def plates_available(self) -> bool:
        return self._ensure_ocr() is not None

    def read_plate(self, car_crop: np.ndarray) -> Optional[tuple[str, float]]:
        """OCR the plate from a car crop. Returns (plate, conf) or None."""
        ocr = self._ensure_ocr()
        if ocr is None or car_crop.size == 0:
            return None
        try:  # pragma: no cover - exercised only with easyocr installed
            h = car_crop.shape[0]
            roi = car_crop[int(h * 0.55):, :]   # plates sit low on the vehicle
            results = ocr.readtext(roi)
            best = None
            for _, text, conf in results:
                t = text.upper().replace("-", "").strip()
                t = re.sub(r"[^A-Z0-9 ]", "", t)
                if CZ_PLATE_RE.match(t.replace(" ", "")) or (6 <= len(t) <= 8):
                    if best is None or conf > best[1]:
                        best = (t, float(conf))
            return best
        except Exception as exc:  # pragma: no cover
            logger.debug("plate OCR failed: %s", exc)
            return None

    def estimate_make_model(self, car_crop: np.ndarray) -> VehicleInfo:
        """Placeholder for a fine-grained classifier. Returns unknowns until a
        model is plugged in (see module docstring)."""
        return VehicleInfo()
