ARG BUILD_FROM=docker.io/library/python:3.14-slim
FROM ${BUILD_FROM}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VOCA_MODELS_DIR=/data/models

WORKDIR /app
COPY app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app/ /app/
COPY run.py /run.py

CMD ["python", "/run.py"]
