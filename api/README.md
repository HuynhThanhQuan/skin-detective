# Skin Detective Inference API

FastAPI wrapper around the Faster R-CNN acne detector.

- `GET  /health` — liveness probe; reports whether trained weights were loaded
- `POST /predict` — multipart upload (`file=@image.jpg`); returns detections + severity grade
- `GET  /docs` — interactive OpenAPI

## Local

```bash
# install once
pip install -r requirements-dev.txt

# run with reload
uvicorn api.main:app --reload --port 8000

# or via docker compose (CPU image, ~1.5 GB resident)
docker compose -f api/docker-compose.yml up --build
```

## Smoke test

```bash
curl -s http://localhost:8000/health | jq
curl -s -X POST -F file=@notebook/sample.jpg http://localhost:8000/predict | jq
```

## Weights

Drop the trained Faster R-CNN state dict at `models/detector.pt` (or set
`SKIN_DETECTIVE_MODEL_PATH=/path/to/weights.pt`). Without weights the API still
boots and serves randomly-initialised predictions — useful for wiring tests, not
diagnosis.

## Resource requirements

CPU inference of Faster R-CNN ResNet-50 needs **~1.5–2 GB RAM** for a typical
~1024×1024 smartphone image. The current $4 DigitalOcean droplet (1 GB) cannot
host this; bump to a 2 GB / $12 plan or run on a machine with a GPU.
