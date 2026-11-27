"""Ollama-compatible front end for llama.cpp's llama-server, for Home Assistant's built-in Ollama integration.

  GET  /api/tags, /api/version, /api/ps     what the Ollama client asks for (one model: the configured one)
  POST /api/show, /api/pull                 answered locally (the model is already on disk)
  POST /api/chat                            -> llama-server /v1/chat/completions (streaming, tool calls)
  *    /v1/...                              passed through to llama-server (OpenAI-compatible API)

Prompt caching: on a Raspberry Pi the ~3k-token Home Assistant system prompt + tool list takes minutes to
process, so it must come from llama-server's prompt cache. Home Assistant puts "Current time is HH:MM:SS..."
into the system prompt, which changes every second and breaks the cache. This server moves that line into the
newest user message (at minute resolution), so the system prompt and tools stay byte-identical across requests.

Environment: LLM_MODEL (key in models.py), LLM_MODEL_DIR, LLM_CTX, LLM_THREADS, LLM_PORT, LLAMA_SERVER, LOG_LEVEL.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MODEL_DIR, MODELS, model_path  # noqa: E402

_LOG = logging.getLogger("voca.llm")
MODEL_KEY = os.environ.get("LLM_MODEL", "qwen3-4b-instruct")
MODEL = MODELS[MODEL_KEY]
CTX = int(os.environ.get("LLM_CTX", "8192"))
THREADS = int(os.environ.get("LLM_THREADS", "4"))
BACKEND_PORT = int(os.environ.get("LLAMA_PORT", "8081"))
BACKEND = f"http://127.0.0.1:{BACKEND_PORT}"
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "/app/llama-server")
# The stable prompt prefix (system prompt + tools) of the last chat, replayed after a restart to fill the cache.
WARMUP_FILE = Path(os.environ.get("LLM_STATE_DIR", MODEL_DIR.parent)) / f"warmup-{MODEL_KEY}.json"

TIME_LINE = re.compile(r"Current time is (\d{1,2}):(\d{2})(?::\d{2})?\. Today's date is (\d{4}-\d{2}-\d{2})\.[ \t]*\n?")
PASS_OPTIONS = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
                "num_predict": "max_tokens", "seed": "seed", "stop": "stop", "repeat_penalty": "repeat_penalty",
                "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty"}

_proc: subprocess.Popen | None = None
_client: httpx.AsyncClient | None = None
_ready = asyncio.Event()


# ------------------------------------------------------------------ llama-server process
def _start_backend():
    global _proc
    cmd = [LLAMA_SERVER, "-m", str(model_path(MODEL_KEY)), "--host", "127.0.0.1", "--port", str(BACKEND_PORT),
           "-c", str(CTX), "-np", "1", "-t", str(THREADS), "--jinja", "--no-webui",
           # 8-bit KV cache (needs flash attention): half the memory of f16, negligible quality loss
           "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
           "--cache-ram", os.environ.get("LLM_CACHE_RAM", "512"),
           *MODEL["sampling"], *os.environ.get("LLAMA_ARGS", "").split()]
    _LOG.info("starting %s", " ".join(cmd))
    _proc = subprocess.Popen(cmd)

    def watch():                                   # llama-server gone -> exit so the add-on gets restarted
        code = _proc.wait()
        _LOG.error("llama-server exited with code %s", code)
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


async def _wait_ready():
    t0 = time.monotonic()
    while True:
        try:
            r = await _client.get(f"{BACKEND}/health", timeout=2)
            if r.status_code == 200:
                _LOG.info("model %s loaded in %.0f s", MODEL["name"], time.monotonic() - t0)
                _ready.set()
                await _warmup()
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1)


def _save_warmup(req: dict):
    """Remember system prompt + tools (a few KB) when they change."""
    msgs = req["messages"]
    if not msgs or msgs[0]["role"] != "system":
        return
    prefix = json.dumps({"messages": msgs[:1], "tools": req.get("tools")}, ensure_ascii=False)
    try:
        if WARMUP_FILE.exists() and WARMUP_FILE.read_text(encoding="utf-8") == prefix:
            return
        tmp = WARMUP_FILE.with_suffix(".tmp")
        tmp.write_text(prefix, encoding="utf-8")
        tmp.replace(WARMUP_FILE)
    except OSError as exc:
        _LOG.warning("could not save warm-up prompt: %s", exc)


async def _warmup():
    """Process the remembered prefix once, so the first real request after a restart is fast."""
    try:
        prefix = json.loads(WARMUP_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    t0 = time.monotonic()
    req = {"messages": prefix["messages"] + [{"role": "user", "content": "Hallo"}], "max_tokens": 1}
    if prefix.get("tools"):
        req["tools"] = prefix["tools"]
    if MODEL["hybrid_thinking"]:
        req["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        r = await _client.post(f"{BACKEND}/v1/chat/completions", json=req)
        n = (r.json().get("timings") or {}).get("prompt_n", 0) if r.status_code == 200 else 0
        _LOG.info("warm-up: cached %d prompt tokens in %.0f s", n, time.monotonic() - t0)
    except (httpx.HTTPError, ValueError) as exc:
        _LOG.warning("warm-up failed: %s", exc)


@asynccontextmanager
async def lifespan(_app):
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5))
    if os.environ.get("LLM_EXTERNAL_BACKEND") != "1":
        _start_backend()
    waiter = asyncio.create_task(_wait_ready())
    yield
    waiter.cancel()
    if _proc and _proc.poll() is None:
        _proc.terminate()
    await _client.aclose()


app = FastAPI(title="voca llm", lifespan=lifespan)


# ------------------------------------------------------------------ Ollama metadata endpoints
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_info() -> dict:
    p = model_path(MODEL_KEY)
    return {"name": MODEL["name"], "model": MODEL["name"], "modified_at": _now_iso(),
            "size": p.stat().st_size if p.exists() else 0, "digest": MODEL["sha256"],
            "details": {"parent_model": "", "format": "gguf", "family": MODEL["family"], "families": [MODEL["family"]],
                        "parameter_size": MODEL["parameter_size"], "quantization_level": MODEL["quant"]}}


@app.get("/")
def root():
    return Response("Ollama is running")           # what Ollama answers; some clients probe it


@app.get("/api/version")
def version():
    return {"version": "0.12.0"}


@app.get("/api/tags")
def tags():
    return {"models": [_model_info()]}


@app.get("/api/ps")
def ps():
    info = _model_info()
    return {"models": [{**info, "expires_at": "2318-01-01T00:00:00Z", "size_vram": 0}] if _ready.is_set() else []}


@app.post("/api/show")
async def show(_req: Request):
    info = _model_info()
    return {"modelfile": "", "parameters": "", "template": "", "details": info["details"],
            "model_info": {"general.architecture": MODEL["family"], f"{MODEL['family']}.context_length": CTX},
            "capabilities": ["completion", "tools"], "modified_at": info["modified_at"]}


@app.post("/api/pull")
async def pull(req: Request):
    body = await req.json()
    name = body.get("model") or body.get("name")
    if name != MODEL["name"]:
        _LOG.warning("pull of %r requested; this server only serves %r (change the add-on option instead)",
                     name, MODEL["name"])
        return JSONResponse({"error": f"this server only serves {MODEL['name']}; pick that model "
                                      "(or change the model in the Voca LLM add-on options)"}, 404)
    if body.get("stream", True) is False:
        return {"status": "success"}
    return StreamingResponse(iter([json.dumps({"status": "success"}) + "\n"]), media_type="application/x-ndjson")


# ------------------------------------------------------------------ /api/chat
def _to_openai(messages: list[dict]) -> list[dict]:
    """Ollama messages -> OpenAI messages; moves the volatile time line out of the system prompt."""
    out, pending, n, stamp = [], [], 0, None
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            def cut(match):
                nonlocal stamp
                stamp = f"Current time is {int(match[1]):02d}:{match[2]}. Today's date is {match[3]}."
                return ""
            content = TIME_LINE.sub(cut, content)
            out.append({"role": "system", "content": content})
        elif role == "assistant":
            msg = {"role": "assistant", "content": content}
            if m.get("tool_calls"):
                msg["tool_calls"] = []
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    call_id = f"call_{n}"; n += 1
                    pending.append(call_id)
                    msg["tool_calls"].append({"id": call_id, "type": "function", "function": {
                        "name": fn.get("name", ""), "arguments": args if isinstance(args, str) else json.dumps(args)}})
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "content": content,
                        "tool_call_id": pending.pop(0) if pending else f"call_{n}"})
        else:
            out.append({"role": "user", "content": content})
    if stamp:
        last_user = next((i for i in range(len(out) - 1, -1, -1) if out[i]["role"] == "user"), None)
        if last_user is not None:
            out[last_user] = {**out[last_user], "content": f"[{stamp}]\n{out[last_user]['content']}"}
        else:
            out.append({"role": "user", "content": f"[{stamp}]"})
    return out


def _request_body(body: dict, stream: bool) -> dict:
    req = {"model": MODEL["name"], "messages": _to_openai(body.get("messages", [])), "stream": stream}
    if body.get("tools"):
        req["tools"] = body["tools"]
    for k, v in (body.get("options") or {}).items():
        if k in PASS_OPTIONS and v is not None:
            req[PASS_OPTIONS[k]] = v
    if MODEL["hybrid_thinking"]:                 # Qwen3 hybrid: think only when Home Assistant asks for it
        req["chat_template_kwargs"] = {"enable_thinking": bool(body.get("think"))}
    fmt = body.get("format")
    if fmt == "json":
        req["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        req["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": fmt}}
    return req


def _ollama_calls(calls: dict) -> list[dict]:
    out = []
    for _, c in sorted(calls.items()):
        try:
            args = json.loads(c["arguments"]) if c["arguments"].strip() else {}
        except json.JSONDecodeError:
            _LOG.warning("model produced invalid tool arguments: %r", c["arguments"])
            args = {}
        out.append({"function": {"name": c["name"], "arguments": args}})
    return out


def _log_timings(t: dict | None, t0: float):
    if t:
        _LOG.info("prompt %d tok (%d cached) %.1f s, answer %d tok %.1f s (%.1f tok/s), total %.1f s",
                  t.get("prompt_n", 0) + t.get("cache_n", 0), t.get("cache_n", 0), t.get("prompt_ms", 0) / 1000,
                  t.get("predicted_n", 0), t.get("predicted_ms", 0) / 1000, t.get("predicted_per_second", 0),
                  time.monotonic() - t0)


def _final(t: dict | None, t0: float, reason: str) -> dict:
    t = t or {}
    return {"model": MODEL["name"], "created_at": _now_iso(), "message": {"role": "assistant", "content": ""},
            "done": True, "done_reason": "length" if reason == "length" else "stop",
            "total_duration": int((time.monotonic() - t0) * 1e9),
            "prompt_eval_count": t.get("prompt_n", 0), "prompt_eval_duration": int(t.get("prompt_ms", 0) * 1e6),
            "eval_count": t.get("predicted_n", 0), "eval_duration": int(t.get("predicted_ms", 0) * 1e6)}


async def _backend_ready():
    try:
        await asyncio.wait_for(_ready.wait(), timeout=600)
    except TimeoutError:
        raise RuntimeError("the model is still loading")


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    stream = body.get("stream", True)
    t0 = time.monotonic()
    try:
        await _backend_ready()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, 503)
    req = _request_body(body, stream)
    _save_warmup(req)
    _LOG.debug("chat request: %s", json.dumps(req, ensure_ascii=False)[:4000])

    if not stream:
        r = await _client.post(f"{BACKEND}/v1/chat/completions", json=req)
        if r.status_code != 200:
            return JSONResponse({"error": r.text}, r.status_code)
        data = r.json()
        msg = data["choices"][0]["message"]
        out = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            out["thinking"] = msg["reasoning_content"]
        if msg.get("tool_calls"):
            out["tool_calls"] = _ollama_calls({i: {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                                               for i, c in enumerate(msg["tool_calls"])})
        _log_timings(data.get("timings"), t0)
        return {**_final(data.get("timings"), t0, data["choices"][0].get("finish_reason")), "message": out}

    async def gen():
        calls: dict[int, dict] = {}
        timings, reason = None, "stop"
        try:
            async with _client.stream("POST", f"{BACKEND}/v1/chat/completions", json=req) as r:
                if r.status_code != 200:
                    err = (await r.aread()).decode(errors="replace")
                    _LOG.error("llama-server %s: %s", r.status_code, err[:500])
                    yield json.dumps({"error": err}) + "\n"
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    timings = data.get("timings") or timings
                    for choice in data.get("choices", []):
                        delta = choice.get("delta") or {}
                        reason = choice.get("finish_reason") or reason
                        msg = {}
                        if delta.get("content"):
                            msg["content"] = delta["content"]
                        if delta.get("reasoning_content"):
                            msg["thinking"] = delta["reasoning_content"]
                        for tc in delta.get("tool_calls") or []:
                            c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                            fn = tc.get("function") or {}
                            c["name"] += fn.get("name") or ""
                            c["arguments"] += fn.get("arguments") or ""
                        if msg:
                            yield json.dumps({"model": MODEL["name"], "created_at": _now_iso(),
                                              "message": {"role": "assistant", "content": "", **msg},
                                              "done": False}) + "\n"
        except httpx.HTTPError as exc:
            _LOG.error("llama-server request failed: %s", exc)
            yield json.dumps({"error": f"llama-server request failed: {exc}"}) + "\n"
            return
        if calls:
            tool_calls = _ollama_calls(calls)
            _LOG.debug("tool calls: %s", tool_calls)
            yield json.dumps({"model": MODEL["name"], "created_at": _now_iso(),
                              "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
                              "done": False}) + "\n"
        _log_timings(timings, t0)
        yield json.dumps(_final(timings, t0, reason)) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ------------------------------------------------------------------ OpenAI passthrough + health
@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def openai_passthrough(path: str, request: Request):
    try:
        await _backend_ready()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, 503)
    upstream = _client.build_request(request.method, f"{BACKEND}/v1/{path}", content=await request.body(),
                                     headers={"content-type": request.headers.get("content-type", "application/json")})
    r = await _client.send(upstream, stream=True)

    async def body():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose()
    return StreamingResponse(body(), status_code=r.status_code, media_type=r.headers.get("content-type"))


@app.get("/health")
def health():
    return {"status": "ok" if _ready.is_set() else "loading", "model": MODEL["name"], "ctx": CTX}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "11434")),
                log_level="warning")
