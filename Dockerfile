# Repo layout expected:  ./backend (FastAPI)   ./frontend (Vite + React)   ./Dockerfile
# ---------- 1) build the React frontend ----------
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
ENV VITE_API_URL=""
RUN npm run build

# ---------- 2) Python API + built frontend ----------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.cache/torch \
    YOLO_CONFIG_DIR=/app/.cache/ultralytics
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
# CPU-only torch keeps the image much smaller
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY backend/ .
COPY --from=web /web/dist ./static

# download the models at build time so the first start is fast
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" \
    && python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]