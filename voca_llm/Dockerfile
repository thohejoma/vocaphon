# llama.cpp's official llama-server image (Ubuntu, CPU backends for every x86/ARM variant) + a small Python front end.
ARG BUILD_FROM=ghcr.io/ggml-org/llama.cpp:server-b12808
FROM ${BUILD_FROM}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LLM_MODEL_DIR=/data/models \
    HF_HOME=/data/hf \
    LLAMA_SERVER=/app/llama-server \
    LD_LIBRARY_PATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

COPY app/requirements.txt /opt/llm/requirements.txt
RUN /opt/venv/bin/pip install -r /opt/llm/requirements.txt

COPY app/ /opt/llm/
COPY run.py /run.py

ENTRYPOINT []
CMD ["/opt/venv/bin/python", "/run.py"]
