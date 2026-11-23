"""Add-on entrypoint: read options, fetch models into /data, announce Wyoming, start the server."""
import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

opts = json.loads(Path("/data/options.json").read_text()) if Path("/data/options.json").exists() else {}
models = Path(os.environ.get("VOCA_MODELS_DIR", "/data/models"))

env = dict(os.environ)
env.update({
    "VOCA_MODELS_DIR": str(models),
    "HOST": "0.0.0.0",
    "PORT": "8000",
    "WYOMING_URI": "tcp://0.0.0.0:10200",
    "WYOMING_VOICE_DE": opts.get("voice_de", "fresh-de-f00"),
    "WYOMING_VOICE_EN": opts.get("voice_en", "fresh-en-f00"),
    "STT_MODEL": opts.get("stt_model", "FermionResearch/Phonon-2.5"),
    "LOG_LEVEL": opts.get("log_level", "info"),
})


def log(msg):
    print(f"[voca] {msg}", flush=True)


# Models: downloaded once into /data (kept across restarts and updates), checksum-verified.
log(f"checking models in {models} (first start downloads ~0.4 GB)")
subprocess.run([sys.executable, "/app/download_models.py"], env=env, check=True)
env["HF_HUB_OFFLINE"] = "1"

# Tell Home Assistant where the Wyoming service is (Settings → Devices & services shows it as discovered).
token = os.environ.get("SUPERVISOR_TOKEN")
if token:
    body = json.dumps({"service": "wyoming", "config": {"uri": f"tcp://{socket.gethostname()}:10200"}}).encode()
    req = urllib.request.Request("http://supervisor/discovery", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            log(f"Wyoming discovery sent ({r.status})")
    except Exception as exc:  # not fatal: the integration can be added by hand
        log(f"Wyoming discovery failed: {exc}")

os.chdir("/app")
os.execvpe(sys.executable, [sys.executable, "/app/server.py"], env)
