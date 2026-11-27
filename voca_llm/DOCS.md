# Voca LLM

An offline large language model as a conversation agent for Assist, running on the CPU. Made for a
**Raspberry Pi 5 with 8 GB RAM**; also runs on any amd64/aarch64 machine.

- **Engine:** [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (pinned build b12808)
- **Model:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), Q4_0
  (2.4 GB, Apache-2.0). Good tool calling, good German, answers without a "thinking" phase.
  Alternative: `qwen3-1.7b` (1 GB, about twice as fast, less reliable with devices).
- **Interface:** the Ollama API, so it works with Home Assistant's built-in **Ollama** integration,
  including *Control Home Assistant* (it can switch devices and read sensor states).

## Setup

1. Start the add-on. The first start downloads the model (2.4 GB) into the add-on's data folder and
   checks it. Watch the **Log** tab until it shows `model qwen3:4b-instruct loaded`. It also prints the
   URL to use, e.g. `http://e720e961-voca-llm:11434`.
2. *Settings → Devices & services → Add integration → **Ollama***. URL: the one from the log
   (`http://<hostname>:11434`), no API key.
3. Add a **conversation agent** to the Ollama entry:
   - Model: **qwen3:4b-instruct** (shown as downloaded)
   - **Control Home Assistant**: Assist
   - Context window size: **8192** (or what you set as `context_size`)
   - Keep alive: -1, Think: off, Max history: 10 is plenty for voice
4. *Settings → Voice assistants*: edit your assistant, set **Conversation agent** to the Ollama agent,
   and turn on **Prefer handling commands locally**. Then simple commands ("Licht Küche an") are handled
   instantly by Home Assistant itself and only everything else goes to the LLM.
5. Expose only the entities you want to control by voice (*Voice assistants → Expose*). Every exposed
   entity makes the prompt longer.

## What to expect on a Raspberry Pi 5

The model generates roughly 4 words per second. The expensive part is reading the prompt: Home Assistant
sends a system prompt with all exposed devices and tools (1,000–4,000 tokens) with every request.

- This add-on keeps that prompt in llama.cpp's cache. It moves Home Assistant's per-second
  "Current time is …" line out of the system prompt (into the newest message), which would otherwise
  invalidate the cache on every request.
- **The first request after the add-on starts, or after you change exposed entities / the agent's
  instructions, processes the whole prompt: one to three minutes.** After that only the new message is
  processed: a device command takes about 5–15 seconds, a two-sentence answer 15–25 seconds.
- After a restart the add-on replays the last system prompt in the background (`warm-up` in the log),
  so the first question is fast again once that finished.

Measured on a 4-core Intel i5-7500T (a Pi 5 is in the same range): cold prompt of 1,200 tokens 65–78 s;
with cache: "Mach das Licht in der Küche an" 6 s per step (tool call, then answer), answers at 3.5–4
tokens per second. Each request's numbers are in the log (`prompt … tok (… cached)`).

## Memory

About 3.2 GB with `context_size` 8192 (model 2.3 GB, 8-bit KV cache, prompt cache). Together with
Home Assistant and Voca Speech this fits in 8 GB. `qwen3-1.7b` needs about 1.5 GB.

## Options

| Option | Meaning |
|---|---|
| `model` | `qwen3-4b-instruct` (recommended) or `qwen3-1.7b`. Switching downloads the other model and deletes the old one |
| `context_size` | Tokens the model sees at once. 8192 is Home Assistant's default; 16384 only for very many exposed entities (+0.6 GB RAM) |
| `threads` | CPU threads. 4 = all cores of a Pi 5 |
| `log_level` | `info`: token counts and timings per request; `debug`: also the requests and tool calls |

## Other clients

Port 11434 (disabled by default, enable it under *Network*) serves the Ollama API and an OpenAI-compatible
API at `/v1` (`/v1/chat/completions`, `/v1/models`), e.g. for Open WebUI.
