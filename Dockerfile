FROM ghcr.io/ai-xlabs-innovation/ltx-video-worker:latest

SHELL ["/bin/bash", "-lc"]
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN set -euo pipefail; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update && apt-get install -y --no-install-recommends rclone ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*; \
    PY_BIN="$(command -v python3 || command -v python)"; \
    "${PY_BIN}" -m pip install --break-system-packages --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["PY_BIN=\"$(command -v python3 || command -v python)\"; exec \"${PY_BIN}\" -u /app/handler.py"]
