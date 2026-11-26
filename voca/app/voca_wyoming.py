"""Wyoming protocol server (Home Assistant Assist): tiny-voca + apptor-tiny-tts TTS, Phonon-2.5 STT.

Runs in the same process as the OpenAI API (see server.py) and shares its models.
"""
from __future__ import annotations

import asyncio
import logging
import os

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

from voca_tts.engine import NothingToSay, detect_language

_LOG = logging.getLogger("voca.wyoming")
LANG_TAGS = {"de": ["de", "de-DE", "de-AT", "de-CH"], "en": ["en", "en-US", "en-GB"]}
CHUNK_SAMPLES = 2048
VERSION = "1.0.0"


def build_info(tts) -> Info:
    tts_attr = Attribution(name="Fraunhofer IEM / tiny-voca", url="https://huggingface.co/Fraunhofer-IEM/tiny-voca-de-en")
    stt_attr = Attribution(name="Fermion Research", url="https://huggingface.co/FermionResearch/Phonon-2.5")
    apptor_attr = Attribution(name="apptor / apptor-tiny-tts", url="https://huggingface.co/apptor/apptor-tiny-tts")
    attrs = {"tiny-voca": tts_attr, "apptor-tiny-tts": apptor_attr}
    descriptions = {"tiny-voca": "tiny-voca German/English TTS", "apptor-tiny-tts": "apptor-tiny-tts German TTS"}

    def program(name, engine):
        voices = [TtsVoice(name=v, description=v, attribution=attrs[name], installed=True, version=VERSION,
                           languages=LANG_TAGS[engine.voice_language(v)]) for v in engine.voices]
        return TtsProgram(name=name, description=descriptions[name], attribution=attrs[name],
                          installed=True, version=VERSION, voices=voices)
    return Info(
        asr=[AsrProgram(name="phonon", description="Phonon-2.5 speech recognition", attribution=stt_attr,
                        installed=True, version=VERSION,
                        models=[AsrModel(name="phonon-2.5", description="Phonon-2.5", attribution=stt_attr,
                                         installed=True, version=VERSION,
                                         languages=LANG_TAGS["de"] + LANG_TAGS["en"])])],
        tts=[program(name, engine) for name, engine in tts.engines.items()],
    )


class VocaHandler(AsyncEventHandler):
    def __init__(self, *args, tts, stt_getter, default_voices: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.tts, self.stt_getter, self.default_voices = tts, stt_getter, default_voices
        self.info_event = build_info(tts).event()
        self.converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.audio = bytearray()
        self.language = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info_event)
        elif Synthesize.is_type(event.type):
            await self._synthesize(Synthesize.from_event(event))
        elif Transcribe.is_type(event.type):
            self.language = Transcribe.from_event(event).language
        elif AudioStart.is_type(event.type):
            self.audio = bytearray()
        elif AudioChunk.is_type(event.type):
            self.audio += self.converter.convert(AudioChunk.from_event(event)).audio
        elif AudioStop.is_type(event.type):
            await self._transcribe()
        return True

    async def _synthesize(self, req: Synthesize):
        text = req.text.strip()
        name = req.voice.name if req.voice and req.voice.name else None
        lang = (req.voice.language if req.voice and req.voice.language else None) or None
        lang = lang[:2].lower() if lang else None
        if name not in self.tts.voices:
            lang = lang if lang in ("de", "en") else detect_language(text)
            name = self.default_voices[lang]
        rate = self.tts.sample_rate
        await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        try:
            wave = await asyncio.to_thread(self.tts.synthesize, text, name, None, 1.0)
            pcm = (np.clip(wave, -1, 1) * 32767).astype("<i2").tobytes()
        except NothingToSay:
            pcm = b""
        except Exception:
            _LOG.exception("synthesis failed")
            pcm = b""
        step = CHUNK_SAMPLES * 2
        for i in range(0, len(pcm), step):
            await self.write_event(AudioChunk(rate=rate, width=2, channels=1, audio=pcm[i:i + step]).event())
        await self.write_event(AudioStop().event())
        _LOG.debug("tts %s: %r", name, text[:60])

    async def _transcribe(self):
        audio = np.frombuffer(bytes(self.audio), dtype="<i2").astype(np.float32) / 32768.0
        self.audio = bytearray()
        text = ""
        if audio.size:
            stt = await asyncio.to_thread(self.stt_getter)
            result = await asyncio.to_thread(stt.transcribe_array_detailed, audio)
            text = result.text.strip()
        _LOG.debug("stt: %r", text)
        await self.write_event(Transcript(text=text, language=self.language).event())


async def run_wyoming(uri: str, tts, stt_getter):
    default_voices = {"de": os.environ.get("WYOMING_VOICE_DE", "fresh-de-f00"),
                      "en": os.environ.get("WYOMING_VOICE_EN", "fresh-en-f00")}
    for lang, v in default_voices.items():
        if v not in tts.voices:
            raise SystemExit(f"default {lang} voice {v!r} does not exist")
    server = AsyncServer.from_uri(uri)
    _LOG.info("Wyoming server on %s", uri)
    await server.run(lambda *a, **k: VocaHandler(*a, tts=tts, stt_getter=stt_getter,
                                                default_voices=default_voices, **k))
