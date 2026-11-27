"""Models the add-on can serve (pinned revisions + SHA-256), and their download into $LLM_MODEL_DIR."""
import hashlib
import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("LLM_MODEL_DIR", Path(__file__).resolve().parent.parent / "models" / "llm"))

# name: what Home Assistant's Ollama integration shows/selects (qwen3:4b-instruct is also its default).
# sampling: the model card's recommended settings, used unless a request overrides them.
MODELS = {
    "qwen3-4b-instruct": {
        "name": "qwen3:4b-instruct", "family": "qwen3", "parameter_size": "4.0B", "quant": "Q4_0",
        "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "revision": "a06e946bb6b655725eafa393f4a9745d460374c9",
        "file": "Qwen3-4B-Instruct-2507-Q4_0.gguf",
        "sha256": "e0ba675d86ab277c61701c6793659b2ae801d95e3be791464c321e6fbf613be2",
        "hybrid_thinking": False,
        "sampling": ["--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0"],
    },
    "qwen3-1.7b": {
        "name": "qwen3:1.7b", "family": "qwen3", "parameter_size": "1.7B", "quant": "Q4_0",
        "repo": "unsloth/Qwen3-1.7B-GGUF", "revision": "d7f544eead698dbd1f15126ef60b45a1e1933222",
        "file": "Qwen3-1.7B-Q4_0.gguf",
        "sha256": "c876f159707a4e4f70e045106c69db15bfc935a4981706fd4f65c6e7ea1e81c5",
        "hybrid_thinking": True,                    # thinking off unless the request sets think=true
        "sampling": ["--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0"],
    },
}


def model_path(key: str) -> Path:
    return MODEL_DIR / MODELS[key]["file"]


def ensure(key: str) -> Path:
    """Download (once) and verify the model; files of other models are removed to free disk space."""
    from huggingface_hub import hf_hub_download

    m, path = MODELS[key], model_path(key)
    stamp = path.with_suffix(".verified")
    if path.exists() and stamp.exists() and stamp.read_text().strip() == m["sha256"]:
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {m['repo']}/{m['file']} ...", flush=True)
    hf_hub_download(m["repo"], m["file"], revision=m["revision"], local_dir=MODEL_DIR)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    if h.hexdigest() != m["sha256"]:
        path.unlink()
        raise SystemExit(f"{path.name}: checksum mismatch, deleted; restart to download again")
    stamp.write_text(m["sha256"])
    for other, om in MODELS.items():
        if other != key:
            for p in (model_path(other), model_path(other).with_suffix(".verified")):
                p.unlink(missing_ok=True)
    return path


if __name__ == "__main__":
    print("model ready:", ensure(os.environ.get("LLM_MODEL", "qwen3-4b-instruct")))
