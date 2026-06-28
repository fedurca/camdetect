"""Threaded RTSP capture.

Each :class:`CameraStream` runs a background thread that continuously decodes
its RTSP feed and keeps only the most recent frame (so consumers never process
stale buffered frames). Connection drops trigger automatic reconnection with
backoff.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

from .config import CameraConfig

logger = logging.getLogger(__name__)

# Force FFmpeg to use TCP for RTSP - far more reliable than the UDP default,
# especially for UniFi Protect streams.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000"
)


class CameraStream:
    """Background RTSP reader exposing the latest decoded frame."""

    def __init__(self, cam: CameraConfig, reconnect_delay: float = 2.0):
        self.cam = cam
        self.reconnect_delay = reconnect_delay

        self._frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._seq: int = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "CameraStream":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name=f"capture-{self.cam.id}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # -- state -------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected

    def read(self) -> tuple[Optional[np.ndarray], int, float]:
        """Return (frame_copy_or_None, sequence, timestamp)."""
        with self._lock:
            if self._frame is None:
                return None, self._seq, 0.0
            return self._frame.copy(), self._seq, self._frame_ts

    # -- worker ------------------------------------------------------------
    def _open(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.cam.url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - not all backends support it
            pass
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _run(self) -> None:
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self._connected = False
                logger.warning("[%s] cannot open stream, retrying", self.cam.id)
                self._stop.wait(self.reconnect_delay)
                continue

            self._connected = True
            logger.info("[%s] stream connected", self.cam.id)
            fail = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    fail += 1
                    if fail > 30:
                        logger.warning("[%s] read failures, reconnecting", self.cam.id)
                        break
                    time.sleep(0.05)
                    continue
                fail = 0
                with self._lock:
                    self._frame = frame
                    self._frame_ts = time.time()
                    self._seq += 1

            cap.release()
            self._connected = False
            if not self._stop.is_set():
                self._stop.wait(self.reconnect_delay)


class CameraManager:
    """Owns one :class:`CameraStream` per configured camera."""

    def __init__(self, cameras: list[CameraConfig]):
        self.streams: dict[str, CameraStream] = {
            cam.id: CameraStream(cam) for cam in cameras
        }

    def start(self) -> None:
        for s in self.streams.values():
            s.start()

    def stop(self) -> None:
        for s in self.streams.values():
            s.stop()

    def get(self, cam_id: str) -> Optional[CameraStream]:
        return self.streams.get(cam_id)

    def status(self) -> dict[str, bool]:
        return {cid: s.connected for cid, s in self.streams.items()}
