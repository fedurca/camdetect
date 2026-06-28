"""Interactive (and file-based) camera calibration.

Goal: produce ``data/calibration/<cam>.json`` for each camera - a homography
mapping image pixels to the shared world ground plane (meters), plus the
camera's world position and optional PnP extrinsics.

Two ways to use it:

1. Interactive (run locally, needs a display):
       python -m calibrate.calibrate --camera cam2 --image data/snapshots/cam2.jpg \
           --map calibrate/satellite.png
   Click >=4 matching points: first in the camera image, then the same ground
   spot on the satellite map. Press 'u' to undo, 's' to save, 'q' to quit.
   The satellite map is calibrated to meters once via --map-scale (m per pixel)
   or by clicking the reference edge (the 14.6 m camera-triangle top edge).

2. From a correspondences file (no display needed, scriptable/testable):
       python -m calibrate.calibrate --from-points calibrate/points.example.json

See ``calibrate/points.example.json`` for the file schema.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np

from backend.calibration import CameraCalibration
from backend.config import PROJECT_ROOT, load_config
from backend.geometry import build_intrinsics, scale_intrinsics, solve_extrinsics


def compute_homography(image_pts: np.ndarray, world_pts: np.ndarray) -> np.ndarray:
    """Homography mapping image pixels -> world meters (X, Y)."""
    import cv2

    H, _ = cv2.findHomography(image_pts, world_pts, method=cv2.RANSAC)
    if H is None:
        raise RuntimeError("findHomography failed - check correspondences")
    return H


def reprojection_error(H: np.ndarray, image_pts: np.ndarray, world_pts: np.ndarray) -> float:
    pts = np.hstack([image_pts, np.ones((len(image_pts), 1))])
    proj = (H @ pts.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return float(np.sqrt(((proj - world_pts) ** 2).sum(axis=1)).mean())


def build_calibration(
    camera_id: str,
    image_size: tuple[int, int],
    correspondences: list[dict],
    world_position: tuple[float, float, float],
    K_full: np.ndarray,
    full_size: tuple[int, int],
    camera_marks: Optional[list[dict]] = None,
) -> CameraCalibration:
    """Create a :class:`CameraCalibration` from ground correspondences."""
    image_pts = np.array([c["image"] for c in correspondences], dtype=np.float64)
    world_pts = np.array([c["world"] for c in correspondences], dtype=np.float64)
    if len(image_pts) < 4:
        raise ValueError(f"{camera_id}: need >=4 correspondences, got {len(image_pts)}")

    H = compute_homography(image_pts, world_pts)
    err = reprojection_error(H, image_pts, world_pts)

    K = scale_intrinsics(K_full, full_size, image_size)

    extrinsics = None
    # Optional PnP: combine ground correspondences (Z=0) with camera-mark 3D
    # points (each = another camera's world position at its mounting height).
    pnp_img = [c["image"] for c in correspondences]
    pnp_world = [[c["world"][0], c["world"][1], 0.0] for c in correspondences]
    for m in camera_marks or []:
        pnp_img.append(m["image"])
        pnp_world.append(m["world"])  # already (X, Y, Z)
    if len(pnp_img) >= 4:
        ext = solve_extrinsics(K, np.array(pnp_img, dtype=np.float64),
                               np.array(pnp_world, dtype=np.float64))
        if ext is not None:
            extrinsics = {
                "rvec": ext.rvec.reshape(3).tolist(),
                "tvec": ext.tvec.reshape(3).tolist(),
                "position": ext.position.tolist(),
            }

    print(f"  {camera_id}: H reprojection error = {err:.3f} m "
          f"({len(image_pts)} pts){' + PnP' if extrinsics else ''}")

    return CameraCalibration(
        camera_id=camera_id,
        image_size=image_size,
        homography=H,
        world_position=world_position,
        intrinsics=K,
        extrinsics=extrinsics,
        extra={"reprojection_error_m": err},
    )


def run_from_points(points_path: str, cfg) -> None:
    with open(points_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    K_full = build_intrinsics(cfg.intrinsics)
    full_size = (cfg.intrinsics.width, cfg.intrinsics.height)
    out_dir = cfg.calibration_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Building calibrations from {points_path}")
    for cam_id, spec in data.get("cameras", {}).items():
        calib = build_calibration(
            camera_id=cam_id,
            image_size=tuple(spec["image_size"]),
            correspondences=spec["correspondences"],
            world_position=tuple(spec.get("world_position", (0.0, 0.0, 0.0))),
            K_full=K_full,
            full_size=full_size,
            camera_marks=spec.get("camera_marks"),
        )
        path = os.path.join(out_dir, f"{cam_id}.json")
        calib.save(path)
        print(f"  saved -> {path}")


def run_interactive(args, cfg) -> None:  # pragma: no cover - needs a display
    import cv2

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read camera image {args.image}")
    smap = cv2.imread(args.map)
    if smap is None:
        raise SystemExit(f"cannot read satellite map {args.map}")

    image_size = (img.shape[1], img.shape[0])
    K_full = build_intrinsics(cfg.intrinsics)
    full_size = (cfg.intrinsics.width, cfg.intrinsics.height)

    state = {"img_pts": [], "map_pts": [], "stage": "image"}

    def on_image(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["stage"] == "image":
            state["img_pts"].append((x, y))
            state["stage"] = "map"

    def on_map(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["stage"] == "map":
            state["map_pts"].append((x, y))
            state["stage"] = "image"

    cv2.namedWindow("camera")
    cv2.namedWindow("map")
    cv2.setMouseCallback("camera", on_image)
    cv2.setMouseCallback("map", on_map)

    scale = args.map_scale  # meters per map pixel
    print("Click matching points (camera first, then map). u=undo s=save q=quit")
    while True:
        ci = img.copy()
        for i, p in enumerate(state["img_pts"]):
            cv2.circle(ci, p, 6, (0, 0, 255), -1)
            cv2.putText(ci, str(i + 1), (p[0] + 6, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        mi = smap.copy()
        for i, p in enumerate(state["map_pts"]):
            cv2.circle(mi, p, 6, (0, 255, 0), -1)
            cv2.putText(mi, str(i + 1), (p[0] + 6, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        cv2.imshow("camera", ci)
        cv2.imshow("map", mi)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u"):
            if state["stage"] == "image" and state["map_pts"]:
                state["map_pts"].pop()
                state["img_pts"].pop()
            elif state["img_pts"]:
                state["img_pts"].pop()
            state["stage"] = "image"
        if key == ord("s"):
            n = min(len(state["img_pts"]), len(state["map_pts"]))
            if n < 4:
                print("need >=4 pairs")
                continue
            corr = [
                {"image": list(state["img_pts"][i]),
                 "world": [state["map_pts"][i][0] * scale, state["map_pts"][i][1] * scale]}
                for i in range(n)
            ]
            calib = build_calibration(
                args.camera, image_size, corr,
                tuple(args.world_position), K_full, full_size,
            )
            path = os.path.join(cfg.calibration_dir, f"{args.camera}.json")
            calib.save(path)
            print(f"saved -> {path}")
    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="camdetect calibration")
    parser.add_argument("--from-points", help="build all calibrations from a JSON file")
    parser.add_argument("--camera", help="camera id (interactive mode)")
    parser.add_argument("--image", help="camera snapshot path (interactive mode)")
    parser.add_argument("--map", help="satellite map path (interactive mode)")
    parser.add_argument("--map-scale", type=float, default=0.05,
                        help="meters per map pixel (interactive mode)")
    parser.add_argument("--world-position", type=float, nargs=3, default=[0, 0, 3.0],
                        help="camera world position X Y Z meters (interactive mode)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()

    if args.from_points:
        run_from_points(args.from_points, cfg)
        return 0
    if args.camera and args.image and args.map:
        run_interactive(args, cfg)
        return 0
    parser.error("provide --from-points OR (--camera --image --map)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
