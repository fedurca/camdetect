"""Video sample recording to MKV with detected events as a subtitle track.

Each recording produces a Matroska (`.mkv`) file:
- the video is muxed with **native encoding** - in live mode the camera's RTSP
  stream is copied without re-encoding (`-c copy`); in demo / fallback mode the
  annotated frames are encoded with FFV1 (a lossless, MKV-native codec),
- the **detected events** captured during the clip are written as an SRT
  **subtitle track** inside the same container (one cue per second listing the
  objects detected on that camera), so the data travels with the video.

Recording is on-demand via the API; one clip per camera at a time.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(cues: list[tuple[float, float, str]]) -> str:
    out = []
    for i, (start, end, text) in enumerate(cues, 1):
        out.append(f"{i}\n{_ts(start)} --> {_ts(end)}\n{text or '-'}\n")
    return "\n".join(out)


class RecordingManager:
    def __init__(self, cfg, mode: str,
                 get_jpeg: Callable[[str], Optional[bytes]],
                 get_state: Callable[[], dict],
                 url_for: Callable[[str], Optional[str]]):
        self.cfg = cfg
        self.mode = mode
        self.get_jpeg = get_jpeg
        self.get_state = get_state
        self.url_for = url_for
        self.dir = cfg.abspath("data/recordings")
        os.makedirs(self.dir, exist_ok=True)
        self._active: dict[str, dict] = {}     # cam -> {stop, thread, file}
        self._lock = threading.Lock()
        self._has_ffmpeg = shutil.which("ffmpeg") is not None

    # -- public API --------------------------------------------------------
    def start(self, cam: str, duration: float = 15.0) -> dict:
        if not self._has_ffmpeg:
            return {"error": "ffmpeg not available"}
        with self._lock:
            if cam in self._active:
                return {"error": f"already recording {cam}"}
            stamp = time.strftime("%Y%m%d-%H%M%S")
            name = f"{cam}_{stamp}.mkv"
            path = os.path.join(self.dir, name)
            stop = threading.Event()
            th = threading.Thread(target=self._record, args=(cam, duration, path, stop),
                                  daemon=True, name=f"rec-{cam}")
            self._active[cam] = {"stop": stop, "thread": th, "file": name}
            th.start()
        logger.info("Recording %s for %.0fs -> %s", cam, duration, name)
        return {"recording": True, "cam": cam, "file": name, "duration": duration}

    def stop(self, cam: str) -> dict:
        with self._lock:
            rec = self._active.get(cam)
        if not rec:
            return {"error": f"not recording {cam}"}
        rec["stop"].set()
        return {"stopping": True, "cam": cam}

    def stop_all(self) -> None:
        with self._lock:
            recs = list(self._active.values())
        for r in recs:
            r["stop"].set()
        for r in recs:
            r["thread"].join(timeout=5.0)

    def list_recordings(self) -> list[dict]:
        out = []
        for f in sorted(os.listdir(self.dir), reverse=True):
            if not f.endswith(".mkv"):
                continue
            p = os.path.join(self.dir, f)
            out.append({"file": f, "size": os.path.getsize(p),
                        "mtime": os.path.getmtime(p),
                        "recording": any(a["file"] == f for a in self._active.values())})
        return out

    # -- worker ------------------------------------------------------------
    def _collect_event(self, cam: str) -> str:
        try:
            state = self.get_state()
        except Exception:
            return ""
        parts = []
        for o in state.get("objects", []):
            if cam in (o.get("cameras") or []) or not o.get("cameras"):
                lbl = f"{o.get('class')} #{o.get('id')} {o.get('prob', 0):.2f}"
                for k in ("behavior", "plate", "engine_type"):
                    if o.get(k):
                        lbl += f" {o[k]}"
                parts.append(lbl)
        au = state.get("audio", {}).get(cam, {})
        for ev in au.get("events", []):
            parts.append(f"audio:{ev.get('type')}")
        return "; ".join(parts)

    def _record(self, cam: str, duration: float, path: str, stop: threading.Event) -> None:
        tmp = path + ".video.mkv"
        srt = path + ".events.srt"
        start = time.time()
        cues: list[tuple[float, float, str]] = []
        try:
            url = self.url_for(cam)
            if self.mode == "live" and url:
                self._record_copy(url, duration, tmp, stop, cam, cues, start)
            else:
                self._record_frames(cam, duration, tmp, stop, cues, start)

            # write subtitles and mux them into the final mkv
            with open(srt, "w", encoding="utf-8") as fh:
                fh.write(build_srt(cues) or "1\n00:00:00,000 --> 00:00:01,000\n-\n")
            if os.path.exists(tmp):
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", srt,
                     "-map", "0", "-map", "1", "-c", "copy", "-c:s", "srt",
                     "-metadata:s:s:0", "title=detected_events", path],
                    check=False)
        except Exception as exc:  # pragma: no cover
            logger.warning("recording %s failed: %s", cam, exc)
        finally:
            for f in (tmp, srt):
                try:
                    os.remove(f)
                except OSError:
                    pass
            with self._lock:
                self._active.pop(cam, None)
            logger.info("Recording %s finished -> %s", cam, os.path.basename(path))

    def _record_copy(self, url, duration, tmp, stop, cam, cues, start) -> None:
        """Live: copy the native RTSP stream (no re-encode)."""
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-rtsp_transport", "tcp",
             "-i", url, "-t", str(duration), "-c", "copy", "-map", "0", tmp],
            stdin=subprocess.DEVNULL)
        last = 0
        while proc.poll() is None and not stop.is_set():
            t = time.time() - start
            if int(t) > last:
                last = int(t)
                cues.append((last - 1, last, self._collect_event(cam)))
            if t > duration:
                break
            time.sleep(0.1)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _record_frames(self, cam, duration, tmp, stop, cues, start) -> None:
        """Demo/fallback: encode annotated frames to FFV1 (native MKV codec)."""
        fps = 10
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe",
             "-framerate", str(fps), "-i", "-", "-c:v", "ffv1", "-pix_fmt",
             "yuv420p", tmp],
            stdin=subprocess.PIPE)
        last_sec = -1
        while not stop.is_set():
            t = time.time() - start
            if t > duration:
                break
            jpeg = self.get_jpeg(cam)
            if jpeg:
                try:
                    proc.stdin.write(jpeg)
                except BrokenPipeError:  # pragma: no cover
                    break
            if int(t) > last_sec:
                last_sec = int(t)
                cues.append((last_sec, last_sec + 1, self._collect_event(cam)))
            time.sleep(1.0 / fps)
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
