"""FastAPI application: serves the UI, MJPEG streams, config, and a WebSocket.

Run with:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000

Environment:
    CAMDETECT_MODE = live | demo   (default: live)
    CAMDETECT_CONFIG = path to config.yaml (optional)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import logging_setup
from .classes import CLASS_COLORS, CLASS_LABELS_CS
from .config import PROJECT_ROOT, load_config
from .geometry import build_intrinsics
from .logging_setup import set_level, setup_logging
from .pipeline import Pipeline
from .settings import Settings

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

CONFIG_PATH = os.environ.get("CAMDETECT_CONFIG")
MODE = os.environ.get("CAMDETECT_MODE", "live").lower()

cfg = load_config(CONFIG_PATH) if CONFIG_PATH else load_config()
setup_logging(cfg)
logger = logging.getLogger("camdetect")

settings = Settings(cfg)
STARTUP_PATH = cfg.abspath("data/startup.json")
settings.load_startup(STARTUP_PATH)
pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting pipeline in '%s' mode", MODE)
    pipeline = Pipeline(cfg, mode=MODE, settings=settings)
    pipeline.start()
    try:
        yield
    finally:
        logger.info("Stopping pipeline")
        if pipeline is not None:
            pipeline.stop()


app = FastAPI(title="camdetect", lifespan=lifespan)


@app.get("/api/config")
def api_config() -> JSONResponse:
    """Scene/config data for the frontend (cameras, world frame, colors)."""
    K = build_intrinsics(cfg.intrinsics)
    return JSONResponse({
        "cameras": [
            {
                "id": c.id,
                "world_xy": list(c.world_xy),
                "height_m": c.height_m,
            }
            for c in cfg.cameras
        ],
        "world": {
            "reference_edge_m": cfg.world.reference_edge_m,
            "ground_elevation_m": cfg.world.ground_elevation_m,
        },
        "intrinsics": {
            "width": cfg.intrinsics.width,
            "height": cfg.intrinsics.height,
            "fov_horizontal_deg": cfg.intrinsics.fov_horizontal_deg,
            "fov_vertical_deg": cfg.intrinsics.fov_vertical_deg,
            "K": K.tolist(),
        },
        "fusion": {
            "default_height_m": cfg.fusion.default_height_m,
        },
        "colors": {name: list(rgb) for name, rgb in CLASS_COLORS.items()},
        "labels": CLASS_LABELS_CS,
        "mode": MODE,
        "features": {
            "database": cfg.database.enabled,
            "vehicles": cfg.vehicles.enabled,
            "transcription": cfg.transcription.enabled,
        },
    })


@app.get("/api/settings")
def api_get_settings() -> JSONResponse:
    return JSONResponse(settings.snapshot())


@app.post("/api/settings")
async def api_set_settings(request: Request) -> JSONResponse:
    try:
        patch = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(patch, dict):
        return JSONResponse({"error": "expected an object"}, status_code=400)
    return JSONResponse(settings.update(patch))


@app.post("/api/settings/save-startup")
def api_save_startup() -> JSONResponse:
    """Persist current settings as the startup defaults for next launch."""
    settings.save_startup(STARTUP_PATH)
    return JSONResponse({"saved": True, "path": STARTUP_PATH})


@app.get("/api/logs")
def api_logs(after: int = 0, limit: int = 500) -> JSONResponse:
    rh = logging_setup.ring_handler
    items = rh.tail(after_seq=after, limit=limit) if rh else []
    return JSONResponse({"logs": items, "level": logging.getLevelName(
        logging.getLogger().level)})


@app.post("/api/log-level")
async def api_log_level(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    level = set_level(str(body.get("level", "INFO")))
    return JSONResponse({"level": level})


@app.get("/api/benchmark")
def api_benchmark() -> JSONResponse:
    if pipeline is None:
        return JSONResponse({"error": "pipeline not ready"}, status_code=503)
    return JSONResponse(pipeline.benchmark())


@app.get("/api/history/objects")
def api_history_objects(limit: int = 200, cls: str | None = None) -> JSONResponse:
    if pipeline is None or pipeline.db is None:
        return JSONResponse({"objects": []})
    return JSONResponse({"objects": pipeline.db.list_objects(limit=limit, cls=cls)})


@app.get("/api/history/events")
def api_history_events(limit: int = 200, kind: str | None = None) -> JSONResponse:
    if pipeline is None or pipeline.db is None:
        return JSONResponse({"events": []})
    return JSONResponse({"events": pipeline.db.list_events(limit=limit, kind=kind)})


@app.get("/api/history/stats")
def api_history_stats() -> JSONResponse:
    if pipeline is None or pipeline.db is None:
        return JSONResponse({"objects": 0, "events": 0, "by_class": {}})
    return JSONResponse(pipeline.db.stats())


@app.post("/api/record/start")
async def api_record_start(request: Request) -> JSONResponse:
    if pipeline is None:
        return JSONResponse({"error": "pipeline not ready"}, status_code=503)
    body = await request.json()
    cam = str(body.get("cam", ""))
    if cfg.camera(cam) is None:
        return JSONResponse({"error": f"unknown camera {cam}"}, status_code=404)
    duration = float(body.get("duration", 15.0))
    return JSONResponse(pipeline.recorder.start(cam, duration))


@app.post("/api/record/stop")
async def api_record_stop(request: Request) -> JSONResponse:
    if pipeline is None:
        return JSONResponse({"error": "pipeline not ready"}, status_code=503)
    body = await request.json()
    return JSONResponse(pipeline.recorder.stop(str(body.get("cam", ""))))


@app.get("/api/recordings")
def api_recordings() -> JSONResponse:
    if pipeline is None:
        return JSONResponse({"recordings": []})
    return JSONResponse({"recordings": pipeline.recorder.list_recordings()})


@app.get("/recordings/{name}")
def get_recording(name: str):
    if pipeline is None or "/" in name or "\\" in name:
        return JSONResponse({"error": "bad request"}, status_code=400)
    path = os.path.join(pipeline.recorder.dir, name)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="video/x-matroska", filename=name)


@app.get("/api/state")
def api_state() -> JSONResponse:
    if pipeline is None:
        return JSONResponse({"objects": [], "cameras": {}})
    return JSONResponse(pipeline.get_state())


def _mjpeg_generator(source, fps: float = 15.0):
    """Yield multipart JPEG frames from a callable returning latest bytes."""
    import time
    boundary = b"--frame"
    while True:
        jpeg = source() if pipeline else None
        if jpeg:
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                   + jpeg + b"\r\n")
        time.sleep(1 / fps)


@app.get("/stream/{cam_id}")
def stream(cam_id: str):
    if cfg.camera(cam_id) is None:
        return JSONResponse({"error": f"unknown camera {cam_id}"}, status_code=404)
    return StreamingResponse(
        _mjpeg_generator(lambda: pipeline.latest_jpeg(cam_id)),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/audio/{cam_id}/spectrogram")
def audio_spectrogram(cam_id: str):
    if cfg.camera(cam_id) is None:
        return JSONResponse({"error": f"unknown camera {cam_id}"}, status_code=404)
    return StreamingResponse(
        _mjpeg_generator(lambda: pipeline.spectrogram_jpeg(cam_id), fps=4.0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            state = pipeline.get_state() if pipeline else {"objects": [], "cameras": {}}
            await websocket.send_text(json.dumps(state))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # pragma: no cover
        logger.debug("ws closed: %s", exc)


# Static frontend mounted last so API/stream/ws routes take precedence.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
