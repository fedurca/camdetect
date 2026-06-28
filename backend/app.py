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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .classes import CLASS_COLORS
from .config import PROJECT_ROOT, load_config
from .geometry import build_intrinsics
from .pipeline import Pipeline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("camdetect")

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

CONFIG_PATH = os.environ.get("CAMDETECT_CONFIG")
MODE = os.environ.get("CAMDETECT_MODE", "live").lower()

cfg = load_config(CONFIG_PATH) if CONFIG_PATH else load_config()
pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting pipeline in '%s' mode", MODE)
    pipeline = Pipeline(cfg, mode=MODE)
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
        "mode": MODE,
    })


@app.get("/api/state")
def api_state() -> JSONResponse:
    if pipeline is None:
        return JSONResponse({"objects": [], "cameras": {}})
    return JSONResponse(pipeline.get_state())


def _mjpeg_generator(cam_id: str):
    boundary = b"--frame"
    while True:
        jpeg = pipeline.latest_jpeg(cam_id) if pipeline else None
        if jpeg:
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                   + jpeg + b"\r\n")
        # ~15 fps cap for the browser <img>; backend produces less on CPU.
        import time
        time.sleep(1 / 15)


@app.get("/stream/{cam_id}")
def stream(cam_id: str):
    if cfg.camera(cam_id) is None:
        return JSONResponse({"error": f"unknown camera {cam_id}"}, status_code=404)
    return StreamingResponse(
        _mjpeg_generator(cam_id),
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
