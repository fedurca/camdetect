# camdetect

Multi-camera 3D object detection and visualization for three UniFi G5 Bullet
cameras overlooking a shared courtyard. It pulls the RTSP streams, detects
people / vehicles / animals with YOLO, localizes each detection on a shared
metric ground plane, fuses detections seen by multiple cameras into single
tracked objects, and shows everything in a browser:

- three live video panels with detection overlays (class, track ID, probability),
- a 4th panel with a 3D scene of the detected objects on the courtyard ground,
  color-coded by class, plus camera positions and a top-down minimap.

```
 cam2 ─┐
 cam3 ─┼─▶ YOLO ─▶ image→ground homography ─▶ fusion/tracking ─▶ WebSocket ─▶ 3D UI
 cam4 ─┘                                                        └▶ MJPEG ─▶ video panels
```

## How localization works

Each camera is calibrated with a homography mapping image pixels to the shared
world ground plane (meters). The world frame and scale come from the Google
Earth top-down view (the three cameras sit at the vertices of a triangle whose
top edge measures 14.6 m). A detection's foot point (bottom-center of its box)
is mapped to world `(X, Y)`; detections from different cameras that land close
together are merged into one physical object. Camera intrinsics `K` are derived
analytically from the identical G5 Bullet FOV (84.4° × 45.4°) and 2K resolution
(2688 × 1512), enabling correct 3D frustums and optional `solvePnP` extrinsics.

## Requirements

- Python 3.11+ (tested on 3.12)
- The local machine. CPU-only works (small model, low FPS); a CUDA GPU is used
  automatically when available. Multi-GPU is config-driven (see below).
- Network access to the cameras at `rtsp://10.24.0.1:7447/...`.

## Quick start

```bash
# 1. Try it with no cameras and no GPU (synthetic moving objects):
./run.sh demo
# open http://localhost:8000

# 2. Check the real cameras are reachable (and save snapshots for calibration):
./run.sh check

# 3. Run live:
./run.sh live
```

`run.sh` creates `.venv` and installs `requirements.txt` on first run. If the
file isn't executable, run it with `bash run.sh demo`. There's also a `Makefile`
with the same shortcuts: `make setup`, `make demo`, `make check`, `make run`.

## Running on your machine (`~/camdetect`, Ubuntu, CPU)

First time, get the latest `main`. If you have local untracked copies of the
camera images (`2.png 3.png 4.png`), a plain `git pull` is blocked
("untracked working tree files would be overwritten"). They're already tracked
in the repo and identical, so just clear the local copies and pull:

```bash
cd ~/camdetect
rm -f 2.png 3.png 4.png        # or: git stash -u
git pull origin main
```

Then run it (CPU-only is fine; first run installs deps and downloads the small
model, which can take a few minutes):

```bash
./run.sh demo        # or: make demo   — no cameras needed, http://localhost:8000
./run.sh check       # confirm the 3 RTSP cameras are reachable on your LAN
./run.sh live        # or: make run    — real cameras
```

Viewing the UI when you're SSH'd into the machine — either open the LAN address
directly (`http://<machine-ip>:8000`) or forward the port over SSH from your
laptop:

```bash
ssh -L 8000:localhost:8000 fedurca@tpd
# then browse to http://localhost:8000 on your laptop
```

To keep it running after you disconnect, use tmux (`tmux new -s camdetect`,
run `./run.sh live`, detach with Ctrl-b d) or install it as a service.

## Configuration

Everything lives in [`config.yaml`](config.yaml):

- `cameras` — RTSP URLs, per-camera world position / height, optional per-camera
  `device` (e.g. `cuda:0`).
- `detection` — model, device (`auto`/`cpu`/`cuda:0`), `imgsz`, confidence, target
  `fps`, and the COCO `classes` to keep (people, vehicles, animals).
- `intrinsics` — camera resolution and FOV (used to build `K`).
- `world` — the 14.6 m reference edge and ground elevation.
- `fusion` — merge distance, track max age, smoothing, default object height.

### CPU now, multi-GPU later

The current default is CPU-friendly: `model: models/yolo11n.pt`, `imgsz: 960`,
`fps: 3`. When you move to the multi-NVIDIA box, no code changes are needed:

- set `detection.model` to a larger model (e.g. `yolo11m.pt` / `yolo11l.pt`),
- raise `detection.fps`,
- optionally pin cameras to GPUs via each camera's `device: "cuda:N"`
  (the pipeline builds one detector per distinct device).

## Calibration

Calibration files live in `data/calibration/<cam>.json`. Example calibrations
are included so demo mode works out of the box — replace them with real ones.

Interactive (run locally, needs a display):

```bash
# Grab snapshots first:
./run.sh check               # writes data/snapshots/<cam>.jpg
# Click >=4 matching ground points in the camera image and the satellite map:
python -m calibrate.calibrate --camera cam2 \
    --image data/snapshots/cam2.jpg --map calibrate/satellite.png \
    --map-scale 0.05 --world-position 0 0 3
```

Non-interactive (scriptable): edit
[`calibrate/points.example.json`](calibrate/points.example.json) with your
correspondences and run:

```bash
python -m calibrate.calibrate --from-points calibrate/points.example.json
```

Tips for accurate world coordinates:

- Export a top-down satellite crop of the courtyard from Google Earth.
- Pick ground features visible in multiple cameras (paving corners, road edges).
- The red rectangles in the original camera images mark where the *other* two
  cameras are — add them as `camera_marks` (image pixel + that camera's world
  X/Y/Z) to refine the PnP extrinsics.

## Project layout

```
config.yaml              # all settings
run.sh                   # launcher (live | demo | check)
backend/
  app.py                 # FastAPI: UI, MJPEG streams, /api/*, WebSocket
  pipeline.py            # capture -> detect -> localize -> fuse orchestration
  cameras.py             # threaded RTSP capture (TCP, auto-reconnect)
  detector.py            # YOLO wrapper, device auto-detect
  calibration.py         # load/apply homographies (image -> world)
  geometry.py            # intrinsics K, scaling, solvePnP extrinsics
  fusion.py              # cross-camera merge + tracking
  classes.py             # class names, colors, heights
  config.py              # config loader
  check_streams.py       # RTSP connectivity test
calibrate/
  calibrate.py           # interactive / file-based calibration tool
  points.example.json    # correspondence file schema + example
data/calibration/        # per-camera calibration JSON
frontend/
  index.html style.css   # 2x2 grid UI
  main.js                # Three.js scene + WebSocket + minimap
  colors.js              # shared class -> color map
models/                  # YOLO weights (auto-downloaded, gitignored)
```

## API

- `GET /` — the web UI.
- `GET /api/config` — cameras, world frame, intrinsics, class colors, mode.
- `GET /api/state` — current fused objects + camera status (JSON snapshot).
- `GET /stream/{cam}` — annotated MJPEG video stream.
- `WS /ws` — pushes `{objects: [...], cameras: {...}}` ~10×/s.

## Validation

```bash
./run.sh check                       # all 3 RTSP streams decode
python -m calibrate.calibrate --from-points calibrate/points.example.json
```

Sanity check: place an object at a measured spot on the courtyard; its world
`(X, Y)` should match, and cameras that both see it should agree within a small
tolerance (the calibration tool prints the homography reprojection error in m).
