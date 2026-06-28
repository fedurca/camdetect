"""Quick connectivity test for the configured RTSP streams.

Usage:
    python -m backend.check_streams
    python -m backend.check_streams --timeout 10 --save

Opens each camera, waits for the first decoded frame, and reports the
resolution. With ``--save`` it writes a snapshot per camera to
``data/snapshots/`` (handy for calibration when the feeds are reachable).
"""
from __future__ import annotations

import argparse
import os
import time

import cv2

from .cameras import CameraManager
from .config import PROJECT_ROOT, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Test RTSP connectivity")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="seconds to wait for a first frame per camera")
    parser.add_argument("--save", action="store_true",
                        help="save a snapshot per camera to data/snapshots/")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    manager = CameraManager(cfg.cameras)
    manager.start()

    snap_dir = os.path.join(PROJECT_ROOT, "data", "snapshots")
    if args.save:
        os.makedirs(snap_dir, exist_ok=True)

    print(f"Testing {len(cfg.cameras)} camera(s), timeout {args.timeout}s each...\n")
    all_ok = True
    try:
        for cam in cfg.cameras:
            stream = manager.get(cam.id)
            assert stream is not None
            deadline = time.time() + args.timeout
            frame = None
            while time.time() < deadline:
                frame, seq, _ = stream.read()
                if frame is not None and seq > 0:
                    break
                time.sleep(0.2)

            if frame is None:
                all_ok = False
                print(f"  [FAIL] {cam.id:6s} {cam.url}  - no frame")
                continue

            h, w = frame.shape[:2]
            print(f"  [ OK ] {cam.id:6s} {cam.url}  - {w}x{h}")
            if args.save:
                path = os.path.join(snap_dir, f"{cam.id}.jpg")
                cv2.imwrite(path, frame)
                print(f"         saved snapshot -> {path}")
    finally:
        manager.stop()

    print("\nAll streams reachable." if all_ok else "\nSome streams failed.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
