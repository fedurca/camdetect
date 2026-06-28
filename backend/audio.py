"""Audio detection subsystem.

Each camera's RTSP stream carries audio. For every camera we:
- capture the audio (ffmpeg -> mono PCM) into a ring buffer (live mode) or
  synthesize it (demo mode),
- compute a log power spectrogram + frequency analysis (dominant frequency,
  low/mid/high band energies, level),
- derive sound events with a light heuristic classifier (optional heavy model
  can be plugged in), and
- run an experimental 2-stroke / 4-stroke engine classifier.

Everything is numpy-only (no heavy DSP deps) so it stays CPU-cheap.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import AudioConfig, CameraConfig
from .settings import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
class AudioSource:
    """Base audio source producing mono float32 samples in [-1, 1]."""

    connected: bool = False

    def start(self) -> None:  # pragma: no cover - interface
        ...

    def stop(self) -> None:  # pragma: no cover - interface
        ...

    def read(self) -> np.ndarray:
        """Return any new samples decoded since the last call (may be empty)."""
        return np.empty(0, dtype=np.float32)


class FfmpegAudioSource(AudioSource):
    """Pulls the audio track from an RTSP stream via ffmpeg as s16le PCM."""

    def __init__(self, url: str, sample_rate: int):
        self.url = url
        self.sample_rate = sample_rate
        self._proc: Optional[subprocess.Popen] = None
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="audio-ffmpeg")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        if not shutil.which("ffmpeg"):
            logger.error("ffmpeg not found; audio capture disabled")
            return
        while not self._stop.is_set():
            cmd = [
                "ffmpeg", "-nostdin", "-loglevel", "quiet",
                "-rtsp_transport", "tcp", "-i", self.url,
                "-vn", "-ac", "1", "-ar", str(self.sample_rate),
                "-f", "s16le", "-",
            ]
            try:
                self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                               stderr=subprocess.DEVNULL)
            except Exception as exc:  # pragma: no cover
                logger.warning("audio ffmpeg launch failed: %s", exc)
                self._stop.wait(2.0)
                continue
            self.connected = True
            try:
                while not self._stop.is_set():
                    chunk = self._proc.stdout.read(4096)
                    if not chunk:
                        break
                    with self._lock:
                        self._buf.extend(chunk)
            finally:
                self.connected = False
                try:
                    self._proc.kill()
                except Exception:
                    pass
            if not self._stop.is_set():
                self._stop.wait(2.0)

    def read(self) -> np.ndarray:
        with self._lock:
            if len(self._buf) < 2:
                return np.empty(0, dtype=np.float32)
            n = len(self._buf) // 2
            raw = bytes(self._buf[: n * 2])
            del self._buf[: n * 2]
        ints = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return ints


class SyntheticAudioSource(AudioSource):
    """Generates a lively test signal: an engine rumble that comes and goes,
    periodic dog barks, and background noise. Lets the audio UI run with no
    cameras."""

    def __init__(self, sample_rate: int, seed: int = 0):
        self.sample_rate = sample_rate
        self.connected = True
        self._t = 0.0
        self._last = time.time()
        self._rng = np.random.default_rng(seed)
        # engine fundamental; <0 marks a 4-stroke-like (strong half order)
        self._engine_f0 = 90.0 + seed * 12.0
        self._four_stroke = (seed % 2 == 0)

    def start(self) -> None:
        self._last = time.time()

    def stop(self) -> None:
        self.connected = False

    def read(self) -> np.ndarray:
        now = time.time()
        dt = now - self._last
        self._last = now
        n = int(dt * self.sample_rate)
        if n <= 0:
            return np.empty(0, dtype=np.float32)
        t = self._t + np.arange(n) / self.sample_rate
        self._t += n / self.sample_rate
        sig = 0.01 * self._rng.standard_normal(n).astype(np.float32)

        # Engine rumble for ~5s every ~12s.
        if (self._t % 12.0) < 5.0:
            f0 = self._engine_f0
            sig += 0.25 * np.sin(2 * np.pi * f0 * t)
            sig += 0.15 * np.sin(2 * np.pi * 2 * f0 * t)
            sig += 0.10 * np.sin(2 * np.pi * 3 * f0 * t)
            if self._four_stroke:
                # strong half-order component (lumpy 4-stroke note)
                sig += 0.18 * np.sin(2 * np.pi * (f0 / 2) * t)

        # Periodic dog bark bursts (~ every 7s).
        if (self._t % 7.0) < 0.25:
            burst = self._rng.standard_normal(n).astype(np.float32)
            sig += 0.4 * burst * np.hanning(max(n, 1)).astype(np.float32)

        return sig.astype(np.float32)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@dataclass
class AudioResult:
    cam_id: str
    level: float = 0.0
    dominant_freq: float = 0.0
    bands: dict = field(default_factory=lambda: {"low": 0.0, "mid": 0.0, "high": 0.0})
    events: list = field(default_factory=list)   # [{"type":str,"conf":float}]
    engine_type: Optional[str] = None             # 2T / 4T / unknown
    connected: bool = False

    def to_dict(self) -> dict:
        return {
            "cam": self.cam_id,
            "level": round(self.level, 3),
            "dominant_freq": round(self.dominant_freq, 1),
            "bands": {k: round(v, 3) for k, v in self.bands.items()},
            "events": self.events,
            "engine_type": self.engine_type,
            "connected": self.connected,
        }


class AudioAnalyzer:
    """Maintains a per-camera ring buffer and computes spectrogram + features."""

    def __init__(self, cam_id: str, cfg: AudioConfig):
        self.cam_id = cam_id
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.window = int(cfg.window_s * self.sr)
        self._ring = np.zeros(self.window, dtype=np.float32)
        self._filled = 0

    def push(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        if samples.size >= self.window:
            self._ring = samples[-self.window:].astype(np.float32)
            self._filled = self.window
            return
        n = samples.size
        self._ring = np.roll(self._ring, -n)
        self._ring[-n:] = samples
        self._filled = min(self.window, self._filled + n)

    # -- spectrogram -------------------------------------------------------
    def _stft_power(self) -> np.ndarray:
        """Return log power spectrogram, shape (n_bins, n_frames)."""
        fft = self.cfg.fft_size
        hop = self.cfg.hop_size
        x = self._ring
        if x.size < fft:
            return np.zeros((fft // 2 + 1, 1), dtype=np.float32)
        win = np.hanning(fft).astype(np.float32)
        frames = 1 + (x.size - fft) // hop
        cols = []
        for i in range(frames):
            seg = x[i * hop: i * hop + fft] * win
            spec = np.abs(np.fft.rfft(seg))
            cols.append(spec)
        mag = np.stack(cols, axis=1)  # (bins, frames)
        return np.log1p(mag)

    def render_spectrogram(self) -> np.ndarray:
        """Return a colorized spectrogram image (H, W, 3) BGR."""
        logp = self._stft_power()
        # normalize
        m = logp.max()
        norm = (logp / m * 255.0).astype(np.uint8) if m > 1e-6 else \
            np.zeros_like(logp, dtype=np.uint8)
        # low freq at bottom
        norm = np.flipud(norm)
        img = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
        img = cv2.resize(img, (self.cfg.spectrogram_width, self.cfg.spectrogram_height),
                         interpolation=cv2.INTER_LINEAR)
        return img

    # -- features ----------------------------------------------------------
    def analyze(self, do_events: bool, do_engine: bool, connected: bool) -> AudioResult:
        res = AudioResult(cam_id=self.cam_id, connected=connected)
        x = self._ring
        if self._filled < self.cfg.fft_size:
            return res

        res.level = float(np.sqrt(np.mean(x ** 2)))

        spec = np.abs(np.fft.rfft(x * np.hanning(x.size).astype(np.float32)))
        freqs = np.fft.rfftfreq(x.size, 1.0 / self.sr)
        if spec.sum() > 1e-9:
            res.dominant_freq = float(freqs[int(np.argmax(spec))])
        total = spec.sum() + 1e-9
        low = spec[(freqs < 250)].sum() / total
        mid = spec[(freqs >= 250) & (freqs < 2000)].sum() / total
        high = spec[(freqs >= 2000)].sum() / total
        res.bands = {"low": float(low), "mid": float(mid), "high": float(high)}

        if do_events:
            res.events = self._classify_events(res, spec, freqs)
        if do_engine:
            res.engine_type = self._engine_type(res, x)
        return res

    def _classify_events(self, res: AudioResult, spec: np.ndarray,
                         freqs: np.ndarray) -> list:
        """Light heuristic sound-event classifier (no heavy model)."""
        events = []
        if res.level < 0.02:
            return events
        low, mid, high = res.bands["low"], res.bands["mid"], res.bands["high"]
        # tonality: peak / mean ratio
        tonality = float(spec.max() / (spec.mean() + 1e-9))
        if low > 0.45 and res.dominant_freq < 250:
            events.append({"type": "engine", "conf": round(min(1.0, low + 0.2), 2)})
        if high > 0.4 and tonality > 8 and res.dominant_freq > 1500:
            events.append({"type": "drone", "conf": round(min(1.0, high), 2)})
        if mid > 0.4 and res.level > 0.15 and tonality < 8:
            events.append({"type": "bark", "conf": round(min(1.0, res.level * 2), 2)})
        if 0.2 < mid < 0.55 and 80 < res.dominant_freq < 400:
            events.append({"type": "speech", "conf": 0.4})
        if not events and res.level > 0.25:
            events.append({"type": "loud", "conf": round(min(1.0, res.level), 2)})
        return events

    def _engine_type(self, res: AudioResult, x: np.ndarray) -> Optional[str]:
        """Experimental 2T vs 4T from low-frequency harmonic structure.

        4-stroke engines exhibit a strong half-order (sub-harmonic) component
        relative to the firing fundamental; 2-stroke fire every revolution and
        lack that lumpy half-order. We estimate the low fundamental via
        autocorrelation, then compare energy at f0 vs f0/2.
        """
        if res.bands["low"] < 0.4 or res.level < 0.05:
            return None
        # low-pass-ish: work on the signal as-is, autocorrelation for period
        sig = x - x.mean()
        ac = np.correlate(sig, sig, mode="full")[sig.size - 1:]
        if ac[0] <= 0:
            return None
        ac /= ac[0]
        # search plausible engine fundamentals 30..200 Hz
        min_lag = int(self.sr / 200)
        max_lag = int(self.sr / 30)
        seg = ac[min_lag:max_lag]
        if seg.size == 0:
            return "unknown"
        lag = min_lag + int(np.argmax(seg))
        f0 = self.sr / lag
        # energy at f0 vs f0/2 from the magnitude spectrum
        spec = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
        freqs = np.fft.rfftfreq(sig.size, 1.0 / self.sr)

        def energy_at(f):
            if f <= 0:
                return 0.0
            idx = np.argmin(np.abs(freqs - f))
            return float(spec[max(0, idx - 1): idx + 2].sum())

        e_f0 = energy_at(f0)
        e_half = energy_at(f0 / 2)
        if e_f0 <= 1e-9:
            return "unknown"
        ratio = e_half / e_f0
        return "4T" if ratio > 0.4 else "2T"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class AudioManager:
    """Owns per-camera audio sources + analyzers and a worker per camera."""

    def __init__(self, cameras: list[CameraConfig], cfg: AudioConfig,
                 settings: Settings, mode: str = "live"):
        self.cfg = cfg
        self.settings = settings
        self.mode = mode
        self.cameras = cameras
        self.analyzers: dict[str, AudioAnalyzer] = {
            c.id: AudioAnalyzer(c.id, cfg) for c in cameras
        }
        self.sources: dict[str, AudioSource] = {}
        for i, c in enumerate(cameras):
            if mode == "live":
                self.sources[c.id] = FfmpegAudioSource(c.url, cfg.sample_rate)
            else:
                self.sources[c.id] = SyntheticAudioSource(cfg.sample_rate, seed=i)

        self._spectrograms: dict[str, bytes] = {}
        self._results: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._running: dict[str, bool] = {c.id: False for c in cameras}

    def start(self) -> None:
        for cam in self.cameras:
            t = threading.Thread(target=self._loop, args=(cam.id,), daemon=True,
                                 name=f"audio-{cam.id}")
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for s in self.sources.values():
            s.stop()
        for t in self._threads:
            t.join(timeout=2.0)

    def _loop(self, cam_id: str) -> None:
        source = self.sources[cam_id]
        analyzer = self.analyzers[cam_id]
        while not self._stop.is_set():
            enabled = bool(self.settings.get("audio", "enabled", default=True))
            if not enabled:
                if self._running[cam_id]:
                    source.stop()
                    self._running[cam_id] = False
                self._stop.wait(0.3)
                continue
            if not self._running[cam_id]:
                source.start()
                self._running[cam_id] = True

            samples = source.read()
            analyzer.push(samples)

            hop = float(self.settings.get("audio", "hop_s", default=self.cfg.hop_s))
            do_events = bool(self.settings.get("audio", "events", default=False))
            do_engine = bool(self.settings.get("audio", "engine_2t4t", default=True))
            result = analyzer.analyze(do_events, do_engine, source.connected)
            spec = analyzer.render_spectrogram()
            ok, buf = cv2.imencode(".jpg", spec)
            with self._lock:
                self._results[cam_id] = result.to_dict()
                if ok:
                    self._spectrograms[cam_id] = buf.tobytes()
            self._stop.wait(max(0.1, hop))

    # -- accessors ---------------------------------------------------------
    def spectrogram_jpeg(self, cam_id: str) -> Optional[bytes]:
        with self._lock:
            return self._spectrograms.get(cam_id)

    def results(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._results)
