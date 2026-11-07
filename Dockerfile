FROM node:22-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Install the CPU wheel so the deployed service does not download CUDA libraries.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install fastapi 'uvicorn[standard]' numpy

COPY backend/main.py /app/backend/main.py
COPY backend/models/webcam_asl_rgb.pt /app/backend/models/webcam_asl_rgb.pt
COPY backend/evaluation/asl_hg_webcam/evaluation_summary.json /app/backend/evaluation/asl_hg_webcam/evaluation_summary.json
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

WORKDIR /app/backend
EXPOSE 10000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
