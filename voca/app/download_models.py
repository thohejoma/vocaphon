"""Fetch all models into ./models (or $VOCA_MODELS_DIR). Idempotent; offline when everything is present."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("VOCA_MODELS_DIR", ROOT / "models"))
os.environ.setdefault("HF_HOME", str(MODELS / "hf"))
os.environ.setdefault("FERMION_CACHE_DIR", str(MODELS / "fermion"))

from huggingface_hub import snapshot_download

if not (MODELS / "tiny-voca-de-en/voca64/model.safetensors").exists():
    snapshot_download("Fraunhofer-IEM/tiny-voca-de-en", allow_patterns=["voca64/*", "*.md", "LICENSE*"],
                      local_dir=MODELS / "tiny-voca-de-en")
if not (MODELS / "dizione/phonemizer/g2p.npz").exists():
    snapshot_download("n4ze3m/tiny-voca", allow_patterns=["phonemizer/*"], local_dir=MODELS / "dizione")

# Phonon: fermion's own fetcher (pinned SHA-256, uses its cache when already unpacked).
from fermion._speech import fetch
from fermion.transcribe import _resolve

repo, key, pin, local_dir = _resolve(os.environ.get("STT_MODEL", "FermionResearch/Phonon-2.5"))
if local_dir is None:
    fetch.ensure(repo, key, pin)
print("all models ready in", MODELS)
