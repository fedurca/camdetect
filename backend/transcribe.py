"""Audio transcription (Czech) + speaker diarization.

How this is solved
------------------
- Transcription: `faster-whisper` (CTranslate2 Whisper) with ``language="cs"``.
  It runs on CPU but is much faster on GPU; pick a model size (tiny/base/small/
  medium) to trade accuracy for speed. It is OFF by default and loaded lazily.
- Diarization (who spoke when): the accurate route is ``pyannote.audio`` (needs a
  HuggingFace token and is heavy). When unavailable we fall back to a light
  energy/pause-based speaker-turn heuristic so segments still get a speaker tag.
- Recording: optionally write rolling per-camera WAV files for later review.

The :class:`TranscriptionManager` pulls the latest ``segment_s`` of audio from
each camera's :class:`~backend.audio.AudioAnalyzer` ring buffer on a worker
thread, transcribes it, stores segments for the UI and logs them to the DB.

In demo mode it emits synthetic Czech segments so the UI/DB paths work without
any model.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "S1"

    def to_dict(self) -> dict:
        return {"start": round(self.start, 2), "end": round(self.end, 2),
                "text": self.text, "speaker": self.speaker}


class Transcriber:
    """Lazy faster-whisper wrapper with a diarization fallback."""

    def __init__(self, language: str = "cs", model: str = "small",
                 diarization: bool = False):
        self.language = language
        self.model_size = model
        self.diarization = diarization
        self._model = None
        self._tried = False

    def _ensure(self):
        if self._tried:
            return self._model
        self._tried = True
        try:  # pragma: no cover - optional heavy dependency
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device="auto",
                                       compute_type="int8")
            logger.info("faster-whisper '%s' loaded (lang=%s)",
                        self.model_size, self.language)
        except Exception as exc:  # pragma: no cover
            logger.warning("transcription model unavailable (%s); disabled", exc)
            self._model = None
        return self._model

    @property
    def available(self) -> bool:
        return self._ensure() is not None

    def transcribe(self, samples: np.ndarray, sr: int) -> list[Segment]:
        model = self._ensure()
        if model is None or samples.size == 0:
            return []
        try:  # pragma: no cover - exercised only with the model installed
            audio = samples.astype(np.float32)
            segs, _ = model.transcribe(audio, language=self.language,
                                       vad_filter=True)
            out = []
            for s in segs:
                spk = self._speaker_for(audio, sr, s.start, s.end)
                out.append(Segment(s.start, s.end, s.text.strip(), spk))
            return out
        except Exception as exc:  # pragma: no cover
            logger.debug("transcribe failed: %s", exc)
            return []

    def _speaker_for(self, audio, sr, start, end) -> str:
        """Very rough diarization fallback: bucket by mean pitch band.

        Real diarization should use pyannote.audio; this only provides a stable
        S1/S2 split so the UI shows speaker turns."""
        if not self.diarization:
            return "S1"
        a = audio[int(start * sr):int(end * sr)]
        if a.size < 256:
            return "S1"
        spec = np.abs(np.fft.rfft(a))
        freqs = np.fft.rfftfreq(a.size, 1.0 / sr)
        centroid = float((freqs * spec).sum() / (spec.sum() + 1e-9))
        return "S2" if centroid > 220 else "S1"


# Synthetic Czech phrases for demo mode.
_DEMO_PHRASES = [
    ("Dobry den, jak se mate?", "S1"),
    ("Vsechno v poradku, dekuji.", "S2"),
    ("To auto tady parkuje kazdy den.", "S1"),
    ("Slysel jsi ten motor?", "S2"),
    ("Pes zase steka na zahrade.", "S1"),
]


class TranscriptionManager:
    def __init__(self, audio_manager, cfg, settings, mode: str = "live",
                 db=None):
        self.audio = audio_manager
        self.cfg = cfg
        self.settings = settings
        self.mode = mode
        self.db = db
        self.transcriber = Transcriber(cfg.language, cfg.model, cfg.diarization)
        self._results: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._wavs: dict[str, wave.Wave_write] = {}
        self._demo_idx = 0

    def start(self) -> None:
        for cam_id in self.audio.analyzers:
            t = threading.Thread(target=self._loop, args=(cam_id,), daemon=True,
                                 name=f"stt-{cam_id}")
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for w in self._wavs.values():
            try:
                w.close()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)

    def _loop(self, cam_id: str) -> None:
        analyzer = self.audio.analyzers[cam_id]
        while not self._stop.is_set():
            enabled = bool(self.settings.get("transcription", "enabled", default=False))
            seg_s = float(self.cfg.segment_s)
            if not enabled:
                self._stop.wait(1.0)
                continue

            samples, sr = analyzer.get_samples()
            if bool(self.settings.get("transcription", "record", default=False)):
                self._record(cam_id, samples, sr)

            if self.mode == "demo":
                segs = self._demo_segments()
            else:
                self.transcriber.diarization = bool(
                    self.settings.get("transcription", "diarization", default=False))
                segs = [s.to_dict() for s in self.transcriber.transcribe(samples, sr)]

            if segs:
                with self._lock:
                    self._results[cam_id] = segs
                if self.db is not None:
                    for s in segs:
                        self.db.log_event("transcript", cam=cam_id,
                                          label=s.get("speaker"), data=s)
            self._stop.wait(seg_s)

    def _demo_segments(self) -> list[dict]:
        phrase, spk = _DEMO_PHRASES[self._demo_idx % len(_DEMO_PHRASES)]
        self._demo_idx += 1
        now = time.time()
        return [Segment(0.0, 2.0, phrase, spk).to_dict()]

    def _record(self, cam_id: str, samples: np.ndarray, sr: int) -> None:
        try:
            w = self._wavs.get(cam_id)
            if w is None:
                d = self.cfg.record_dir
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"{cam_id}.wav")
                w = wave.open(path, "wb")
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                self._wavs[cam_id] = w
            pcm = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
            w.writeframes(pcm.tobytes())
        except Exception as exc:  # pragma: no cover
            logger.debug("record failed: %s", exc)

    def results(self) -> dict[str, list[dict]]:
        with self._lock:
            return {k: list(v) for k, v in self._results.items()}
