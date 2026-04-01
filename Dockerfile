FROM ghcr.io/ai-xlabs-innovation/ltx-video-worker:latest

SHELL ["/bin/bash", "-lc"]
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN set -euo pipefail; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update && apt-get install -y --no-install-recommends rclone ffmpeg ca-certificates git && rm -rf /var/lib/apt/lists/*; \
    PY_BIN="$(command -v python3 || command -v python)"; \
    "${PY_BIN}" -m pip install --break-system-packages --no-cache-dir -r /app/requirements.txt; \
    git clone --depth 1 https://github.com/Lightricks/LTX-2 /opt/LTX-2; \
    "${PY_BIN}" -m pip install --break-system-packages --no-cache-dir uv; \
    cd /opt/LTX-2; \
    "${PY_BIN}" -m uv sync --frozen --no-dev; \
    /opt/LTX-2/.venv/bin/pip install --no-cache-dir runpod gdown requests huggingface_hub

COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/LTX-2/.venv/bin:${PATH}"
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["exec /opt/LTX-2/.venv/bin/python -u /app/handler.py"]
