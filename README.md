# camdetect

Multi-camera 3D object detection and visualization for three UniFi G5 Bullet
cameras overlooking a shared courtyard. It pulls the RTSP streams, detects
people / vehicles / animals with YOLO, localizes each detection on a shared
metric ground plane, fuses detections seen by multiple cameras into single
tracked objects, and shows everything in a browser:

- three live video panels with detection overlays (class, track ID, probability) -
  hideable, hidden by default,
- a 3D scene of the detected objects on the courtyard ground, color-coded by class,
- a dedicated top-down (plan) view,
- an audio panel with a live spectrogram, frequency analysis, sound events, and
  a live Czech transcript, and
- tabs for object/event **History**, a **Benchmark** (GPU/CPU + tuning), and a
  **Debug** log viewer.

Both video and audio detection are independently toggleable and tunable (CPU
load / frame rate) from the in-app settings drawer. Detected objects and events
are logged to a local SQLite database and browsable in the History tab.

```
 cam2 ─┬─ video ─▶ YOLO (+ YOLO-World) ─▶ ground homography ─▶ fusion/tracking ─┐
 cam3 ─┤                                                                        ├▶ WS ─▶ 3D + top-down
 cam4 ─┘─ audio ─▶ spectrogram + events + 2T/4T ───────────────────────────────┘     └▶ MJPEG panels
```

## Detection (video)

- Known classes via a COCO model (`yolo11n`): person, car, motorcycle, bicycle,
  dog, cat, bird, ...
- Rare classes via an optional open-vocabulary detector (YOLO-World) driven by
  text prompts: trash bin (popelnice), kick scooter (koloberka), roller skates
  (brusle), drone (dron). Heavy on CPU, so OFF by default; recommended for the GPU rig.
- Per-person attributes: behavior from motion (standing/walking/running/loitering)
  and an experimental, off-by-default age estimate (needs a visible face; these
  overhead cameras rarely provide one, so treat as experimental).

## Detection (audio)

Audio is part of each camera's RTSP stream. The audio subsystem computes a live
log spectrogram and frequency analysis (dominant frequency, low/mid/high band
energy, level), and derives sound events with a light heuristic classifier
(engine, drone, bark, speech, loud). An experimental 2-stroke vs 4-stroke engine
classifier inspects the low-frequency harmonic structure and tags nearby
motorcycle/car tracks. A heavy event model can be enabled later (GPU).

## Audio recording, transcription and diarization (Czech)

Approach (see `backend/transcribe.py`):

- **Transcription** uses `faster-whisper` (CTranslate2 Whisper) with
  `language="cs"`. Pick a model size (`tiny`/`base`/`small`/`medium`) to trade
  accuracy for speed; it runs on CPU but is much faster on GPU. Loaded lazily and
  OFF by default.
- **Diarization** (who spoke when): the accurate route is `pyannote.audio` (needs
  a HuggingFace token, heavy). When it is not installed we fall back to a light
  spectral-centroid heuristic that still tags S1/S2 turns so the UI shows speaker
  separation.
- **Recording**: optional rolling per-camera WAV files (`data/audio/<cam>.wav`).

Transcripts appear live in the audio panel and are logged to the database. In
demo mode synthetic Czech segments are emitted so the path works without models.

To enable the real models locally: `pip install -r requirements-optional.txt`
(faster-whisper, optionally pyannote.audio) and turn on transcription in the
settings drawer. Recommended on the GPU box.

### Linking a speaker to a detected person (acoustic localization)

Each transcript segment is tied to the most likely **person** detected in video
(`backend/pipeline.py::_locate_speaker`). Because all three cameras have a
microphone and we know their world positions, we localize the speaker by
**level-pattern matching**: for every person track we predict the per-camera
speech loudness from inverse-square distance and pick the person whose predicted
pattern best matches the speech-band energy actually measured at the three mics
(cosine similarity), plus an energy-weighted source-position estimate. This needs
no sample-level audio sync, so it is robust over the three independent RTSP audio
streams. The linked person id is shown in the transcript (`osoba #N`), on the
person's 3D label, and stored with each `transcript` event in the database.

### How to test it on your machine

Demo (no cameras/models needed) - verify the whole path quickly:

```bash
./run.sh demo            # http://localhost:8000
# In the UI: open "Nastaveni", enable "Prepis (CZ)".
# Audio panel shows Czech segments tagged "osoba #N"; the person's 3D label
# shows the spoken text. Open the "Historie" tab -> transcript events are
# labelled with the linked person and a localization score.
```

Live (real cameras, Czech speech):

```bash
pip install -r requirements-optional.txt     # faster-whisper (+ optional pyannote)
./run.sh live
# Enable "Prepis (CZ)" (and "Diarizace mluvcich" for speaker turns).
# Speak near one camera while a person is in view of the cameras: the transcript
# should attach to that person (osoba #N). Walking closer to a different camera
# shifts the localization. Check the result in the Historie tab and in
# data/camdetect.sqlite (events table, kind='transcript', data.loc + data.track_id).
```

Notes on accuracy: localization is level-based, so it works best when speakers
are separated in space and the mics' gains are comparable. With a single person
in view the segment is attributed to them directly. For sample-accurate TDOA
triangulation you would need synchronized audio capture (future work).

## Vehicle analysis: plates, make/model, age, drivetrain

On car/truck/bus detections (`backend/vehicles.py`, experimental, OFF by default):

- **License plate (ANPR)**: optional `easyocr` backend reads the plate region
  (lower part of the vehicle box); Czech plates match `\d[A-Z]\d \d{4}`.
- **Make / model / age / drivetrain**: a real solution uses a fine-grained
  vehicle classifier plus a plate->registry lookup; no such model ships here, so
  live values are `unknown` until one is plugged in.

In demo mode stable synthetic values are attached per car so the UI/DB show the
feature. Results are merged onto the fused track and persisted to the database.

## Video sample recording (MKV + events in subtitles)

`backend/recorder.py` records short MKV samples on demand from the **Historie**
tab ("Nahrat vzorek"):

- video uses **native encoding** - in live mode the camera's RTSP stream is
  stream-copied (`-c copy`, no re-encode); in demo/fallback the annotated frames
  are encoded with FFV1 (lossless, MKV-native),
- the **detected events** captured during the clip are muxed into the same
  container as an **SRT subtitle track** (one cue per second), so the data
  travels inside the file.

API: `POST /api/record/start {cam,duration}`, `POST /api/record/stop {cam}`,
`GET /api/recordings`, `GET /recordings/{name}`. Files land in
`data/recordings/` (gitignored). Inspect with
`ffprobe file.mkv` (shows a `subrip` subtitle stream) or play in mpv/VLC with
subtitles enabled.

## Camera coverage (from the UniFi coverage map)

Each camera has a ground **coverage wedge** defined by its world position, look
azimuth and field of view (`config.yaml` per-camera `azimuth_deg`/`fov_deg`/
`range_m`; `azimuth_deg: null` auto-aims at the camera-cluster centroid). It is
used two ways:

- **Visualization**: the wedges are drawn in the top-down view and as flat
  sectors on the 3D ground, mirroring the UniFi Protect coverage plan.
- **Precision**: a detection whose ground projection falls outside the camera's
  coverage wedge is rejected (it is almost always a homography/labeling error),
  which removes ghost detections and tightens cross-camera fusion.

Set `azimuth_deg`/`range_m` per camera from the UniFi coverage map for the most
accurate wedges (cam2 faces the garage gate, cam3 is to its left, cam4 to its
right).

## Duplicate-detection handling

The same physical object seen from several cameras (often with a flipped label,
e.g. a silver car read as car/truck/bus) used to appear multiple times. Fusion
now groups confusable vehicle classes together, uses a class-aware merge radius
(larger for vehicles), clusters detections agglomeratively, and de-duplicates
tracks that drift within the merge radius - collapsing them into one object with
the highest-confidence label.

## Object/event history (SQLite)

`backend/db.py` logs to a local SQLite DB (`data/camdetect.sqlite`):

- `objects` - one row per physical object, merged across time/cameras by
  similarity (same class within `database.merge_distance_m` and
  `database.merge_time_s`), accumulating first/last seen, max confidence,
  observation count and a JSON `attrs` blob (behavior, age, plate, make/model,
  engine type, ...).
- `events` - append-only log (object appeared, audio events, transcripts, ...).

Browse it in the **History** tab; query it via `/api/history/objects`,
`/api/history/events`, `/api/history/stats`. Rows older than
`database.retention_days` are purged.

## Benchmark and startup config

The **Benchmark** tab reports whether you are on **GPU or CPU** (and the GPU
name), measures inference latency/FPS for the current model, and suggests a
per-camera FPS. Tune the detection settings in the drawer, then
**"Ulozit jako vychozi"** persists them to `data/startup.json`, which is loaded
on top of `config.yaml` at the next launch (so the program starts with your
tuned config). Delete that file to revert to the committed defaults.

## Debug log

The **Debug** tab has two panes: the **system log** (also written to
`data/logs/camdetect.log`, rotating) and a **detection-events log** fed from the
database. Verbosity is set in `config.yaml` (`logging.level`), can be overridden
at startup via `CAMDETECT_LOG_LEVEL`, and changed live from the tab
(`POST /api/log-level`).

## UI notes

- The audio panel shows a **single combined frequency plot** with all three
  cameras' band energies overlaid (grouped bars, one color per camera); the
  spectrogram tabs still pick a single camera.
- The layout is **mobile friendly**: on narrow screens the panels stack
  vertically, the tab bar scrolls, and the audio sub-panels reflow to one column.

## One config of record

All persistent parameters - camera RTSP URLs, their mutual world positions,
detection/audio/db/logging defaults - live in a single committed
[`config.yaml`](config.yaml). Only machine-specific runtime tweaks land in the
gitignored `data/startup.json` overlay.

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
- `ffmpeg` on PATH (used to pull the audio track from each RTSP stream).
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

The web UI is always published on **all network interfaces** (`0.0.0.0`), so it
is reachable from other machines on the LAN. On startup the server prints the
exact URLs (e.g. `http://192.168.x.y:8000`); you can also list the host IPs with
`hostname -I`.

## Testing on the NVIDIA multi-GPU PC

1. Get the code and create the environment on the GPU machine:

```bash
git clone https://github.com/fedurca/camdetect && cd camdetect   # or: git pull
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# CUDA-enabled PyTorch (pick the index matching your driver/CUDA):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional: Czech STT + ANPR
```

2. Confirm the GPU is visible:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

3. Turn the heavy features on for GPU in [`config.yaml`](config.yaml):

- `detection.device: "auto"` (auto-selects CUDA) and a larger `model:`
  (e.g. `yolo11m.pt` or `yolo11l.pt`), higher `fps`;
- enable `detection.open_vocabulary.enabled`, `detection.attributes.age`,
  `vehicles.enabled`, `audio.events.enabled`, `transcription.enabled` as desired;
- multi-GPU: pin cameras to GPUs with each camera's `device: "cuda:0"` /
  `"cuda:1"` (the pipeline builds one detector per distinct device).

4. Run it (binds to all interfaces):

```bash
./run.sh live          # or: ./run.sh demo  to try without cameras
# startup prints: http://<this-pc-ip>:8000
```

5. Open it from any other PC on the network at `http://<gpu-pc-ip>:8000`.
   If it is not reachable, open the port in the firewall:

```bash
sudo ufw allow 8000/tcp        # Ubuntu firewall (if enabled)
```

6. Verify GPU usage in the app: open the **Benchmark** tab and click
   "Spustit benchmark" - it should report `GPU (NVIDIA ...)`, a low latency and
   high FPS. Tune the detection settings in **Nastaveni** and click
   "Ulozit jako vychozi" so the program starts with that config next time
   (saved to `data/startup.json`). Watch live load with `nvidia-smi -l 1`.

Notes: the 3D view loads three.js from a CDN, so the browser you view from needs
internet access. The camera RTSP streams must be reachable from the GPU PC's
network.

## Configuration

Everything lives in [`config.yaml`](config.yaml):

- `cameras` — RTSP URLs, per-camera world position / height, optional per-camera
  `device` (e.g. `cuda:0`).
- `detection` — `enabled`, model, device (`auto`/`cpu`/`cuda:0`), `imgsz`,
  confidence, target `fps`, the COCO `classes`, `open_vocabulary` (YOLO-World
  model + prompts), and `attributes` (behavior, age).
- `audio` — `enabled`, sample rate, analysis `window_s`/`hop_s`, spectrogram size,
  `engine_2t4t`, and the heavy `events` model toggle.
- `intrinsics` — camera resolution and FOV (used to build `K`).
- `world` — the 14.6 m reference edge and ground elevation.
- `fusion` — merge distance, track max age, smoothing, default object height.

Most of these are also adjustable live from the in-app settings drawer (gear in
the top bar); changes are POSTed to `/api/settings`, applied immediately, and
remembered in the browser via `localStorage`.

### Runtime controls (settings drawer)

- Camera previews: on/off (hidden by default).
- Video detection: on/off, FPS, resolution, confidence, open-vocabulary on/off + prompts.
- Person attributes: behavior on/off, age on/off (experimental).
- Audio detection: on/off, event model on/off, 2T/4T on/off, window/hop.

### CPU now, multi-GPU later

Defaults are CPU-friendly: COCO `yolo11n` at `imgsz 960`, `fps 3`, open-vocabulary
OFF, audio spectrogram + frequency analysis + 2T/4T ON (cheap), audio event model
OFF, age OFF. When you move to the multi-NVIDIA box, no code changes are needed:

- set `detection.model` to a larger model (e.g. `yolo11m.pt` / `yolo11l.pt`),
- raise `detection.fps`, enable `open_vocabulary`, enable `audio.events` and `age`,
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
  pipeline.py            # capture -> detect -> localize -> fuse + benchmark
  cameras.py             # threaded RTSP capture (TCP, auto-reconnect)
  detector.py            # YOLO + YOLO-World wrappers, device auto-detect
  audio.py               # audio capture, spectrogram, events, 2T/4T
  transcribe.py          # Czech transcription + diarization (optional)
  vehicles.py            # ANPR + make/model/age/drivetrain (optional)
  attributes.py          # experimental age estimation (optional)
  db.py                  # SQLite object/event store + similarity merge
  settings.py            # thread-safe runtime settings + startup persistence
  logging_setup.py       # file + ring-buffer logging for the debug window
  calibration.py         # load/apply homographies (image -> world)
  geometry.py            # intrinsics K, scaling, solvePnP extrinsics
  fusion.py              # cross-camera merge + tracking + behavior
  classes.py             # class names, colors, heights, Czech labels
  config.py              # config loader
  check_streams.py       # RTSP connectivity test
calibrate/
  calibrate.py           # interactive / file-based calibration tool
  points.example.json    # correspondence file schema + example
data/calibration/        # per-camera calibration JSON
frontend/
  index.html style.css   # settings drawer + cameras + 3D + top-down + audio
  main.js                # Three.js scene, top-down, audio panel, settings
  colors.js              # shared class -> color map
models/                  # YOLO weights (auto-downloaded, gitignored)
```

## API

- `GET /` — the web UI.
- `GET /api/config` — cameras, world frame, intrinsics, class colors, Czech labels, features, mode.
- `GET /api/settings` / `POST /api/settings` — read / live-update runtime settings.
- `POST /api/settings/save-startup` — persist current settings as startup defaults.
- `GET /api/state` — fused objects (+ behavior/age/engine/vehicle), camera status, audio, transcripts.
- `GET /stream/{cam}` — annotated MJPEG video stream.
- `GET /audio/{cam}/spectrogram` — MJPEG spectrogram stream.
- `GET /api/history/objects|events|stats` — browse the SQLite store.
- `GET /api/benchmark` — device (GPU/CPU) + inference FPS estimate.
- `GET /api/logs` / `POST /api/log-level` — debug log tail / set verbosity.
- `WS /ws` — pushes `{objects, cameras, audio, transcripts}` ~10×/s.

## Validation

```bash
./run.sh check                       # all 3 RTSP streams decode
python -m calibrate.calibrate --from-points calibrate/points.example.json
```

Sanity check: place an object at a measured spot on the courtyard; its world
`(X, Y)` should match, and cameras that both see it should agree within a small
tolerance (the calibration tool prints the homography reprojection error in m).
