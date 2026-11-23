"""tiny-voca German/English TTS: model loading, text frontends, synthesis.

Model: Fraunhofer-IEM/tiny-voca-de-en (voca64: 32 German + 32 English voices).
The model consumes phoneme ids from the 88-symbol inventory in `text.py`
(ids 0-75 are identical to upstream tiny-voca, 76-87 are German phones).

Frontends
  en: dizione, the official tiny-voca English phonemizer (same ids, used as is).
  de: espeak-ng IPA (bundled via espeakng-loader), mapped onto the inventory.
      The model's own German frontend ("Phony") is unpublished, so this mapping
      is a best-effort approximation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .config import Config
from .model import TinyVoca, UtteranceTooLong
from .text import SYMBOL_TO_ID, VOWEL_SET

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("VOCA_MODELS_DIR", ROOT / "models"))
MODEL_DIR = Path(os.environ.get("VOCA_TTS_MODEL_DIR", MODELS / "tiny-voca-de-en/voca64"))
os.environ.setdefault("DIZIONE_DATA", str(MODELS / "dizione/phonemizer"))

PUNCT = {",", ".", "?", "!", ";", ":", "—", "…"}
SP, BOS, EOS = SYMBOL_TO_ID["<sp>"], SYMBOL_TO_ID["<bos>"], SYMBOL_TO_ID["<eos>"]


class NothingToSay(ValueError):
    pass


# ------------------------------------------------------------------ German frontend
# espeak-ng IPA -> model symbol. Longest match first. Symbols already in the
# inventory map to themselves (see _DE_TABLE construction).
_DE_MAP = {
    "ɔø": "ɔɪ", "ɔʏ": "ɔɪ", "ɔy": "ɔɪ",
    "ɑː": "aː", "ɑ": "a", "ɛː": "ɛ", "æ": "ɛ", "ɒ": "ɔ",
    "e": "eː", "o": "oː", "u": "uː", "y": "yː", "ø": "øː",
    "ɜ": "ɐ", "ɜː": "ɐ", "ɚ": "ɐ",
    "ɾ": "ʁ", "r": "ʁ", "ʀ": "ʁ", "ɹ": "ʁ", "χ": "x",
    "g": "ɡ", "l̩": "əl", "m̩": "əm", "ɐ̯": "ɐ", "ɪ̯": "ɪ", "ʊ̯": "ʊ",
}
_DE_DROP = set("ʔ'ː-_ ̩̯")
_SYMBOLS = {s for s in SYMBOL_TO_ID if not s.startswith("<")} - PUNCT - {'"', "'", "(", ")"}


def _de_table():
    table = {s: [s] for s in _SYMBOLS}
    for k, v in _DE_MAP.items():
        table[k] = [v] if v in SYMBOL_TO_ID else list(v)
    return table


_DE_TABLE = _de_table()
_DE_KEYS = sorted(_DE_TABLE, key=len, reverse=True)
_espeak = None
_espeak_lock = threading.Lock()


def _espeak_de():
    global _espeak
    if _espeak is None:
        import espeakng_loader
        from phonemizer.backend import EspeakBackend
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        _espeak = EspeakBackend("de", preserve_punctuation=True, with_stress=True,
                                language_switch="remove-flags", punctuation_marks=';:,.!?¡¿—…"«»“”()')
    return _espeak


def phonemize_de(text: str):
    """German text -> (ids, stress) lists including <bos>/<eos>."""
    with _espeak_lock:
        ipa = _espeak_de().phonemize([text], strip=True)[0]
    ids, stress, pending, have_phone = [BOS], [0], 0, False
    for word in ipa.split():
        if have_phone and ids[-1] not in {SYMBOL_TO_ID[p] for p in PUNCT}:
            ids.append(SP); stress.append(0)
        i = 0
        while i < len(word):
            ch = word[i]
            if ch in "ˈˌ":
                pending = 1 if ch == "ˈ" else 2; i += 1; continue
            if ch in PUNCT or ch in "¡¿":
                if ch in PUNCT and have_phone:
                    if ids[-1] == SP:
                        ids.pop(); stress.pop()
                    if ids[-1] not in {SYMBOL_TO_ID[p] for p in PUNCT}:
                        ids.append(SYMBOL_TO_ID[ch]); stress.append(0)
                i += 1; continue
            for key in _DE_KEYS:
                if word.startswith(key, i):
                    for sym in _DE_TABLE[key]:
                        is_vowel = sym in VOWEL_SET or sym in ("ɐ", "əl")
                        ids.append(SYMBOL_TO_ID[sym])
                        stress.append(pending if is_vowel else 0)
                        if is_vowel:
                            pending = 0
                    have_phone = True
                    i += len(key)
                    break
            else:
                i += 1                       # unknown/diacritic character: drop
    if not have_phone:
        raise NothingToSay("the text has nothing to pronounce")
    if ids[-1] == SP:
        ids.pop(); stress.pop()
    ids.append(EOS); stress.append(0)
    return ids, stress


def phonemize_en(text: str):
    import dizione
    try:
        p = dizione.phonemize(text)
    except ValueError as exc:
        raise NothingToSay(str(exc)) from exc
    return list(p.ids), list(p.stress)


# ------------------------------------------------------------------ text splitting
_SPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"""(?:(?<=[.!?…])|(?<=[.!?…]["')\]»“”]))\s+(?=["'(\[„«]?[A-ZÄÖÜ0-9])""")
_NOT_AN_END = re.compile(r"(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m|"
                         r"z\.b|bzw|usw|ca|nr|hr|fr|str|vgl|evtl|ggf|d\.h|u\.a)|\b[A-ZÄÖÜ]|\d)\.$", re.IGNORECASE)
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")


def sentences(text: str):
    parts, current = [], ""
    for piece in _SENTENCE_END.split(_SPACE.sub(" ", text).strip()):
        current = f"{current} {piece}" if current else piece
        if not _NOT_AN_END.search(current):
            parts.append(current); current = ""
    if current:
        parts.append(current)
    out = []
    for part in parts:
        if any(c.isalnum() for c in part):
            out.append(part if part.rstrip("\"'”’)]»")[-1:] in ".!?…,;:" else part + ".")
    return out


def _shorter(text):
    pieces = _CLAUSE_END.split(text)
    if len(pieces) > 1:
        return pieces
    words = text.split(" ")
    if len(words) < 2:
        return []
    half = (len(words) + 1) // 2
    return [" ".join(words[:half]), " ".join(words[half:])]


# ------------------------------------------------------------------ language detection
_DE_WORDS = set("der die das und ist nicht ich du er sie es wir ihr ein eine einen dem den des mit auf für "
                "von zu im ist sind war bin bist hat haben wird werden auch noch aber oder wie was wo wer "
                "warum heute morgen guten tag danke bitte ja nein sehr gut schön mein dein kein man nur "
                "schon mal dann wenn weil dass doch hallo".split())
_EN_WORDS = set("the and is are not you he she it we they a an of to in for on with this that was were be "
                "have has will would can could what where who why how hello thanks please yes no very good "
                "my your i am do does did".split())


def detect_language(text: str) -> str:
    if re.search(r"[äöüßÄÖÜ]", text):
        return "de"
    words = re.findall(r"[a-zA-Z']+", text.lower())
    de = sum(w in _DE_WORDS for w in words)
    en = sum(w in _EN_WORDS for w in words)
    return "de" if de > en else "en"


# ------------------------------------------------------------------ the model
class VocaTTS:
    sample_rate = 24000

    def __init__(self, model_dir: Path = MODEL_DIR):
        meta = json.loads((model_dir / "config.json").read_text())
        if meta.get("format") != "designed-tts/model/v1" or meta.get("family") != "voca":
            raise ValueError(f"{model_dir}: not a designed-tts voca model")
        weights = model_dir / meta["weights"]["file"]
        if hashlib.sha256(weights.read_bytes()).hexdigest() != meta["weights"]["sha256"]:
            raise ValueError(f"{weights}: checksum mismatch")
        self.config = Config.from_dict(meta["config"])
        self.model = TinyVoca(self.config)
        self.model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        self.model.eval()
        self.seed = int(meta.get("default_seed", 0))
        self.voices = list(self.config.speakers)
        self.lock = threading.Lock()

    def voice_language(self, voice: str) -> str:
        return "de" if "-de-" in voice else "en"

    @torch.inference_mode()
    def _run(self, ids, stress, vid, speed, seed):
        wave, lengths, _ = self.model.synthesize(
            torch.tensor([ids]), torch.tensor([stress]), torch.tensor([len(ids)]),
            torch.tensor([vid]), speed=speed, seed=seed)
        wave = wave[0, :int(lengths[0])]
        if not torch.isfinite(wave).all():
            raise RuntimeError("model produced non-finite audio")
        return wave.numpy()

    def synthesize(self, text: str, voice: str, language: str | None = None, speed: float = 1.0) -> np.ndarray:
        """Text of any length -> float32 mono waveform at 24 kHz, peak <= 0.99."""
        if voice not in self.voices:
            raise ValueError(f"unknown voice {voice!r}")
        language = language or self.voice_language(voice)
        phonemize = phonemize_de if language == "de" else phonemize_en
        vid = self.voices.index(voice)
        limit = self.config.max_tokens - 12
        waves = []
        pause = np.zeros(int(0.12 * self.sample_rate), dtype=np.float32)

        def read(piece):
            try:
                ids, stress = phonemize(piece)
            except NothingToSay:
                return
            if len(ids) <= limit:
                try:
                    waves.append(self._run(ids, stress, vid, speed, self.seed + len(waves)))
                    return
                except UtteranceTooLong:
                    pass
            parts = _shorter(piece)
            if not parts:
                raise ValueError(f"word too long to speak: {piece[:40]!r}")
            for part in parts:
                read(part)

        with self.lock:
            for sentence in sentences(text):
                read(sentence)
        if not waves:
            raise NothingToSay("nothing in the text could be pronounced")
        out = np.concatenate([x for w in waves for x in (w, pause)][:-1]).astype(np.float32)
        peak = float(np.abs(out).max()) if out.size else 0.0
        return out * (0.99 / peak) if peak > 0.99 else out
