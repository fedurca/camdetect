"""Experimental per-person attribute estimation (age).

Behavior (standing/walking/running/loitering) is derived purely from motion in
:mod:`backend.fusion` and is reliable. Age estimation, by contrast, needs a
clearly visible face - rarely the case for these overhead, wide-angle cameras -
so it is OFF by default and treated as experimental.

This module is a plug point: if an optional age model is available it is used;
otherwise :meth:`AgeEstimator.estimate` returns ``None`` (no guessing). The demo
mode populates synthetic ages so the UI path is exercised without a model.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Coarse age buckets used for display.
AGE_BUCKETS = ["child", "teen", "adult", "senior"]


class AgeEstimator:
    """Estimates a coarse age bucket from a person crop, if a model is present."""

    def __init__(self) -> None:
        self._model = None
        self._tried = False

    def _ensure_model(self) -> None:
        if self._tried:
            return
        self._tried = True
        # Optional: plug a real model here (e.g. an ONNX age net on face crops).
        # We intentionally do not pull a heavy dependency by default.
        try:  # pragma: no cover - only if user installs a model
            import os

            from .config import PROJECT_ROOT

            path = os.path.join(PROJECT_ROOT, "models", "age.onnx")
            if os.path.exists(path):
                import cv2

                self._model = cv2.dnn.readNetFromONNX(path)
                logger.info("Age model loaded from %s", path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Age model unavailable: %s", exc)
            self._model = None

    @property
    def available(self) -> bool:
        self._ensure_model()
        return self._model is not None

    def estimate(self, person_crop: np.ndarray) -> Optional[tuple[str, float]]:
        """Return (age_bucket, confidence) or None when no model is available."""
        self._ensure_model()
        if self._model is None or person_crop.size == 0:
            return None
        try:  # pragma: no cover - depends on the optional model's I/O
            blob = __import__("cv2").dnn.blobFromImage(
                person_crop, scalefactor=1 / 255.0, size=(224, 224), swapRB=True)
            self._model.setInput(blob)
            out = self._model.forward().ravel()
            idx = int(np.argmax(out))
            conf = float(out[idx] / (out.sum() + 1e-9))
            bucket = AGE_BUCKETS[min(idx, len(AGE_BUCKETS) - 1)]
            return bucket, conf
        except Exception as exc:  # pragma: no cover
            logger.debug("age estimate failed: %s", exc)
            return None
