# syntax=docker/dockerfile:1.7

# ---------- Stage 1: build the SPA ----------
FROM node:20-bookworm-slim AS spa
WORKDIR /spa
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build


# ---------- Stage 2: FastAPI runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1
WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv

# chunker_core (path dep) + backend
COPY pyproject.toml ./
COPY chunker_core/ ./chunker_core/
COPY backend/pyproject.toml ./backend/pyproject.toml
COPY backend/app/ ./backend/app/
COPY prompt.txt ./prompt.txt

RUN uv pip install --no-cache -e . -e ./backend

# Built SPA — main.py mounts /app/frontend/build/client at /
COPY --from=spa /spa/build/ ./frontend/build/

ENV CHUNKER_DATA_DIR=/data \
    CHUNKER_ENABLE_SPLITTER=false \
    CHUNKER_VLLM_BASE_URL=http://host.docker.internal:8001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
