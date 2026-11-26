"""apptor-tiny-tts (apptor/apptor-tiny-tts): German-only TTS, voices F2 / M2, 24 kHz, onnxruntime.

The model repo ships its own runtime (speak.py + tinyvoca/ front end, G2P model instead of eSpeak);
download_models.py fetches a pinned revision into $VOCA_MODELS_DIR/apptor-tiny-tts.
Voice ids here: apptor-de-f2, apptor-de-m2.

MultiTTS puts tiny-voca and apptor behind the one interface server.py and voca_wyoming.py use.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np

from .engine import MODELS, NothingToSay

APPTOR_DIR = Path(os.environ.get("APPTOR_MODEL_DIR", MODELS / "apptor-tiny-tts"))
_LOG = logging.getLogger("voca.apptor")


class ApptorTTS:
    sample_rate = 24000
    PREFIX = "apptor-de-"

    def __init__(self, model_dir: Path = APPTOR_DIR, vocos: bool = True, threads: int = 4):
        sys.path.insert(0, str(model_dir))              # for its tinyvoca package
        spec = importlib.util.spec_from_file_location("apptor_speak", model_dir / "speak.py")
        speak = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(speak)
        self.speaker = speak.Speaker(model_dir / "apptor-tiny-tts-de.onnx", vocos=vocos, threads=threads)
        if self.speaker.meta["sample_rate"] != self.sample_rate:
            raise ValueError("apptor-tiny-tts: unexpected sample rate")
        self.model_voice = {self.PREFIX + v.lower(): v for v in self.speaker.voices}   # apptor-de-f2 -> F2
        self.voices = list(self.model_voice)
        self.lock = threading.Lock()

    def voice_language(self, voice: str) -> str:
        return "de"

    def synthesize(self, text: str, voice: str, language: str | None = None, speed: float = 1.0) -> np.ndarray:
        if voice not in self.model_voice:
            raise ValueError(f"unknown voice {voice!r}")
        with self.lock:
            try:
                out = self.speaker.speak(text, voice=self.model_voice[voice], speed=speed)
            except ValueError as exc:                   # "the text has nothing to pronounce"
                raise NothingToSay(str(exc)) from exc
        if out.size == 0:
            raise NothingToSay("nothing in the text could be pronounced")
        return out.astype(np.float32)


class MultiTTS:
    """tiny-voca plus (optionally) apptor-tiny-tts; dispatches on the voice id."""
    sample_rate = 24000

    def __init__(self, voca, apptor=None):
        self.engines = {"tiny-voca": voca}
        if apptor is not None:
            self.engines["apptor-tiny-tts"] = apptor
        self._by_voice = {v: e for e in self.engines.values() for v in e.voices}
        self.voices = list(self._by_voice)

    def engine_name(self, voice: str) -> str:
        return next(n for n, e in self.engines.items() if voice in e.voices)

    def voice_language(self, voice: str) -> str:
        return self._by_voice[voice].voice_language(voice)

    def synthesize(self, text: str, voice: str, language: str | None = None, speed: float = 1.0) -> np.ndarray:
        if voice not in self._by_voice:
            raise ValueError(f"unknown voice {voice!r}")
        return self._by_voice[voice].synthesize(text, voice, language, speed)


def load_apptor():
    """ApptorTTS, or None when disabled (APPTOR=0) or the model files are missing."""
    if os.environ.get("APPTOR", "1") == "0":
        return None
    if not (APPTOR_DIR / "apptor-tiny-tts-de.onnx").exists():
        _LOG.warning("apptor-tiny-tts not found in %s (run download_models.py); its voices are disabled", APPTOR_DIR)
        return None
    return ApptorTTS(vocos=os.environ.get("APPTOR_VOCOS", "1") != "0",
                     threads=int(os.environ.get("APPTOR_THREADS", "4")))
