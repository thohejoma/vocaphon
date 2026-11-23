"""OpenAI-compatible speech server.

  POST /v1/audio/speech           TTS  tiny-voca (Fraunhofer-IEM/tiny-voca-de-en), German + English
  POST /v1/audio/transcriptions   STT  Phonon-2.5 (FermionResearch/Phonon-2.5), German + English (+23 more)
  GET  /v1/models                 model list
  GET  /v1/audio/voices           voice list (non-standard helper)

Optional Wyoming server for Home Assistant Assist: set WYOMING_URI (e.g. tcp://0.0.0.0:10200).
All models and caches live under ./models (override with VOCA_MODELS_DIR).
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = Path(os.environ.setdefault("VOCA_MODELS_DIR", str(ROOT / "models")))
os.environ.setdefault("HF_HOME", str(MODELS / "hf"))
os.environ.setdefault("FERMION_CACHE_DIR", str(MODELS / "fermion"))

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from voca_tts.engine import NothingToSay, VocaTTS, detect_language

STT_MODEL = os.environ.get("STT_MODEL", "FermionResearch/Phonon-2.5")
TTS_MODEL_NAMES = {"tiny-voca", "tts-1", "tts-1-hd", "gpt-4o-mini-tts"}
STT_MODEL_NAMES = {"phonon-2.5", "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"}

# OpenAI voice names -> voice index within the chosen language ("f03" / "m01").
OPENAI_VOICES = {
    "nova": "f00", "shimmer": "f01", "alloy": "f02", "coral": "f03", "sage": "f04",
    "echo": "m00", "onyx": "m01", "fable": "m02", "ash": "m03", "ballad": "m04", "verse": "m05",
}

tts = VocaTTS()
_stt = None


_stt_lock = threading.Lock()


def stt():
    global _stt
    with _stt_lock:
        if _stt is None:
            import fermion
            _stt = fermion.load_speech(STT_MODEL, quiet=False)
    return _stt


@asynccontextmanager
async def lifespan(_app):
    if os.environ.get("STT_LAZY") != "1":
        stt()
    yield


app = FastAPI(title="voca speech server", lifespan=lifespan)


# ------------------------------------------------------------------ models
@app.get("/v1/models")
def models():
    now = int(time.time())
    ids = ["tiny-voca", "phonon-2.5", "tts-1", "tts-1-hd", "whisper-1"]
    return {"object": "list", "data": [{"id": i, "object": "model", "created": now, "owned_by": "local"} for i in ids]}


@app.get("/v1/audio/voices")
def voices():
    return {"openai_aliases": OPENAI_VOICES, "voices": [
        {"id": v, "language": tts.voice_language(v), "gender": "female" if v.split("-")[-1][0] == "f" else "male"}
        for v in tts.voices]}


# ------------------------------------------------------------------ TTS
class SpeechRequest(BaseModel):
    model: str = "tiny-voca"
    input: str = Field(..., max_length=20000)
    voice: str = "nova"
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    language: str | None = None          # extension: "de" / "en"; default = from voice id or auto-detect
    instructions: str | None = None      # accepted for compatibility, ignored


def _resolve_voice(voice: str, language: str | None, text: str) -> tuple[str, str]:
    voice = voice.strip()
    if voice in tts.voices:
        return voice, (language or tts.voice_language(voice))
    lang = (language or detect_language(text)).lower()[:2]
    if lang not in ("de", "en"):
        raise HTTPException(400, f"unsupported language {language!r}; use 'de' or 'en'")
    key = voice.lower()
    if key in OPENAI_VOICES:
        return f"fresh-{lang}-{OPENAI_VOICES[key]}", lang
    if len(key) == 3 and key[0] in "fm" and key[1:].isdigit():       # shorthand like "f07"
        cand = f"fresh-{lang}-{key}"
        if cand in tts.voices:
            return cand, lang
    raise HTTPException(400, f"unknown voice {voice!r}; see GET /v1/audio/voices")


def _encode(wave: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if fmt == "pcm":
        return (np.clip(wave, -1, 1) * 32767).astype("<i2").tobytes(), "audio/pcm"
    if fmt == "wav":
        sf.write(buf, wave, sr, format="WAV", subtype="PCM_16"); return buf.getvalue(), "audio/wav"
    if fmt == "flac":
        sf.write(buf, wave, sr, format="FLAC"); return buf.getvalue(), "audio/flac"
    if fmt == "mp3":
        sf.write(buf, wave, sr, format="MP3"); return buf.getvalue(), "audio/mpeg"
    if fmt == "opus":
        sf.write(buf, wave, sr, format="OGG", subtype="OPUS"); return buf.getvalue(), "audio/ogg"
    if fmt == "aac":
        if not shutil.which("ffmpeg"):
            raise HTTPException(400, "aac needs ffmpeg on PATH")
        pcm = (np.clip(wave, -1, 1) * 32767).astype("<i2").tobytes()
        out = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-",
                              "-c:a", "aac", "-f", "adts", "-"], input=pcm, capture_output=True, check=True).stdout
        return out, "audio/aac"
    raise HTTPException(400, f"unsupported response_format {fmt!r} (mp3, opus, aac, flac, wav, pcm)")


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input.strip():
        raise HTTPException(400, "input is empty")
    voice, lang = _resolve_voice(req.voice, req.language, req.input)
    try:
        wave = tts.synthesize(req.input, voice, language=lang, speed=req.speed)
    except NothingToSay as exc:
        raise HTTPException(400, str(exc))
    data, mime = _encode(wave, tts.sample_rate, req.response_format.lower())
    return Response(data, media_type=mime, headers={"X-Voice": voice, "X-Language": lang})


# ------------------------------------------------------------------ STT
def _load_audio_16k(raw: bytes, filename: str) -> np.ndarray:
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        return audio
    except Exception:
        pass
    if not shutil.which("ffmpeg"):
        raise HTTPException(400, f"could not decode {filename!r} (install ffmpeg for m4a/webm/aac)")
    with tempfile.NamedTemporaryFile(suffix=Path(filename or "a").suffix) as f:
        f.write(raw); f.flush()
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f.name, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                           capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise HTTPException(400, f"could not decode {filename!r}: {p.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def _ts(t: float, sep: str) -> str:
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("phonon-2.5"),
    language: str | None = Form(None),       # Phonon-2.5 detects the language itself; echoed back
    prompt: str | None = Form(None),         # used as hotwords (comma/space separated terms)
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: list[str] | None = Form(None, alias="timestamp_granularities[]"),
):
    audio = _load_audio_16k(file.file.read(), file.filename or "")
    if audio.size == 0:
        raise HTTPException(400, "empty audio")
    hotwords = None
    if prompt:
        hotwords = [w.strip() for w in prompt.replace("\n", ",").split(",") if w.strip()][:25] or None
    r = stt().transcribe_array_detailed(audio, hotwords=hotwords)
    fmt = response_format.lower()
    if fmt == "text":
        return PlainTextResponse(r.text)
    if fmt in ("srt", "vtt"):
        sep = "," if fmt == "srt" else "."
        lines = [] if fmt == "srt" else ["WEBVTT", ""]
        for i, s in enumerate(r.segments, 1):
            if fmt == "srt":
                lines.append(str(i))
            lines += [f"{_ts(s['start'], sep)} --> {_ts(s['end'], sep)}", s["text"].strip(), ""]
        return PlainTextResponse("\n".join(lines), media_type="text/vtt" if fmt == "vtt" else "text/plain")
    if fmt == "verbose_json":
        from fermion._speech.cues import openai_segments
        body = {"task": "transcribe", "language": language or "", "duration": round(r.audio_seconds, 3),
                "text": r.text, "segments": openai_segments(r.segments)}
        if r.timed and (not timestamp_granularities or "word" in timestamp_granularities):
            body["words"] = [{"word": w["text"], "start": w["start"], "end": w["end"]} for w in r.words]
        return JSONResponse(body)
    if fmt == "json":
        return {"text": r.text, "usage": {"type": "duration", "seconds": int(np.ceil(r.audio_seconds))}}
    raise HTTPException(400, f"unsupported response_format {response_format!r}")


@app.get("/health")
def health():
    return {"status": "ok", "tts_voices": len(tts.voices), "stt_loaded": _stt is not None}


async def _main():
    import asyncio
    import logging
    import uvicorn
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="%(levelname)s %(name)s: %(message)s")
    server = uvicorn.Server(uvicorn.Config(app, host=os.environ.get("HOST", "0.0.0.0"),
                                           port=int(os.environ.get("PORT", "8000"))))
    tasks = [server.serve()]
    if os.environ.get("WYOMING_URI"):
        from voca_wyoming import run_wyoming
        tasks.append(run_wyoming(os.environ["WYOMING_URI"], tts, stt))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
