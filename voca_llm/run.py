"""Add-on entrypoint: read options, fetch the model into /data, start the Ollama-compatible server."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

opts = json.loads(Path("/data/options.json").read_text()) if Path("/data/options.json").exists() else {}
env = dict(os.environ)
env.update({
    "LLM_MODEL": opts.get("model", "qwen3-4b-instruct"),
    "LLM_CTX": str(opts.get("context_size", 8192)),
    "LLM_THREADS": str(opts.get("threads", 4)),
    "HOST": "0.0.0.0",
    "PORT": "11434",
    "LOG_LEVEL": opts.get("log_level", "info"),
})


def log(msg):
    print(f"[voca-llm] {msg}", flush=True)


log(f"checking model {env['LLM_MODEL']} in {env['LLM_MODEL_DIR']} (first start downloads 1-2.4 GB)")
subprocess.run([sys.executable, "/opt/llm/models.py"], env=env, check=True)
env["HF_HUB_OFFLINE"] = "1"
log(f"Home Assistant: add the Ollama integration with URL http://{socket.gethostname()}:11434")
os.chdir("/opt/llm")
os.execvpe(sys.executable, [sys.executable, "/opt/llm/llm_server.py"], env)
