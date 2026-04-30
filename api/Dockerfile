# Skin Detective inference API — slim production image, CPU-only.
#
# Build:  docker build -f api/Dockerfile -t skin-detective-api .
# Run:    docker run --rm -p 8000:8000 \
#           -v "$PWD/models":/app/models:ro \
#           -e SKIN_DETECTIVE_MODEL_PATH=/app/models/detector.pt \
#           skin-detective-api
#
# Resource note: Faster R-CNN ResNet-50 inference needs ~1.5–2 GB RAM on CPU.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch wheels keep the image small.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1 && \
    pip install \
        fastapi==0.115.4 \
        uvicorn[standard]==0.32.0 \
        python-multipart==0.0.12 \
        Pillow==11.0.0 \
        numpy==1.26.4

# Project source — only the bits the API needs.
COPY acne_configs.py ./
COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
