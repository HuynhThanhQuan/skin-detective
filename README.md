# Skin Detective

> AI engine behind **Skin Detective** — automatic acne object detection and severity grading
> from smartphone images. Published in *Diagnostics* 2022 (Q1, JCR; Scopus IF 3.9).

[![ci](https://github.com/HuynhThanhQuan/skin-detective/actions/workflows/ci.yml/badge.svg)](https://github.com/HuynhThanhQuan/skin-detective/actions/workflows/ci.yml)
[![paper](https://img.shields.io/badge/Diagnostics-12%282022%29%201879-blue)](https://doi.org/10.3390/diagnostics12081879)

A two-stage pipeline: a **Faster R-CNN** detector localises five lesion classes on the face,
then a **LightGBM** classifier consumes the detections and grades overall severity (mild /
moderate / severe).

## Highlights

- **Paper:** [MDPI](https://www.mdpi.com/2075-4418/12/8/1879) · [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9406819/) · [DOI 10.3390/diagnostics12081879](https://doi.org/10.3390/diagnostics12081879)
- **Press:** [VnExpress (AI Tech Matching grant)](https://vnexpress.net/5-du-an-duoc-dau-tu-ai-tech-matching-4514085.html) · [VnExpress (product)](https://vnexpress.net/skin-detective-ung-dung-tich-hop-tri-tue-nhan-tao-phat-hien-cac-benh-ve-da-va-ket-noi-bac-si-da-lieu-4498851.html) · [Tuổi Trẻ](https://tuoitre.vn/khoi-nghiep-voi-khat-vong-khong-de-nguoi-dan-hoi-chut-la-di-vien-20230321231316476.htm) · [CafeF](https://cafef.vn/uoc-mo-nguoi-o-nong-thon-van-duoc-kham-bac-si-gioi-dang-sau-ung-dung-ai-ho-tro-kham-benh-danh-rieng-cho-nguoi-viet-20221120161048606.chn) · [video](https://drive.google.com/file/d/1_tZqrh5ARUuLCWThNOvOH1k30iosA1LL/view)

## Repository layout

```
skin-detective/
├── api/                      # FastAPI inference service (CPU/GPU)
│   ├── main.py · inference.py · schemas.py
│   ├── Dockerfile · docker-compose.yml
│   └── tests/                # contract tests for the API surface
├── notebook/                 # research / EDA / training notebooks
├── test/                     # unit smoke tests
├── doc/                      # paper + conference materials
├── train.py · trainer.py     # detector training entry-point
├── engine.py                 # train_one_epoch / evaluate loops
├── acne_dataset.py           # COCO-formatted dataset adapter
├── acne_configs.py           # class IDs, colours, short labels
├── coco_eval.py · coco_utils.py
├── requirements.txt          # runtime deps (pinned)
├── requirements-dev.txt      # adds pytest, ruff, fastapi, uvicorn
├── Dockerfile                # CPU image (Python 3.11 + Torch 2.4.1)
├── Dockerfile.cuda           # GPU image (CUDA 12.1)
└── Makefile                  # `make help`
```

## 1. Quick start

### Option A — Docker (recommended)

```bash
# CPU image (research / API):
docker build -t skin-detective .
docker run --rm -it -p 8888:8888 \
  -v "$PWD/data":/app/data \
  -v "$PWD/models":/app/models \
  skin-detective
# → JupyterLab at http://localhost:8888

# GPU image (for training):
docker build -f Dockerfile.cuda -t skin-detective:cuda .
docker run --rm -it --gpus all -p 8888:8888 \
  -v "$PWD/data":/app/data \
  -v "$PWD/models":/app/models \
  skin-detective:cuda
```

### Option B — Local virtualenv

```bash
make install            # creates .venv, installs CPU torch + dev deps
make test               # smoke tests on CPU
make api                # FastAPI on http://localhost:8000
```

> See [`Makefile`](Makefile) for the full target list (`make help`).

## 2. Inference API

```bash
make api-up                   # docker compose -f api/docker-compose.yml up --build
curl -s http://localhost:8000/health | jq
curl -s -X POST -F file=@notebook/sample.jpg \
  http://localhost:8000/predict | jq
```

Response shape (`POST /predict`):

```json
{
  "image_width": 1024, "image_height": 1024,
  "detections": [
    {"class_id": 1, "class_short": "thuong_viem", "score": 0.91,
     "box": {"x": 312, "y": 488, "w": 64, "h": 70}}
  ],
  "severity_grade": "moderate",
  "severity_probabilities": {"mild": 0.18, "moderate": 0.66, "severe": 0.16},
  "inference_ms": 1140.3,
  "model_version": "fasterrcnn-r50-v1"
}
```

Trained weights: drop the Faster R-CNN state dict at `models/detector.pt` (or set
`SKIN_DETECTIVE_MODEL_PATH`). Without weights the API still boots and serves
randomly-initialised predictions — useful for wiring tests, not diagnosis.
See [`api/README.md`](api/README.md).

## 3. Training

Dataset must be COCO-formatted:

```
data/
├── train/{image, coco_instances.json}
├── val/{image, coco_instances.json}
└── test/{image, coco_instances.json}
```

```bash
make train                # python train.py --data ./data --epochs 100
# checkpoints land in ./models/
# tensorboard logs in ./logs/<date>/<time>/tb/{train,test}
tensorboard --logdir logs/
```

For the **grade classifier** (LightGBM on detection statistics), run
`notebook/acne_circle_final.ipynb`.

## 4. Class taxonomy

Defined in [`acne_configs.py`](acne_configs.py) (Vietnamese labels with English shortcodes):

| ID | Label                    | Short        |
|----|--------------------------|--------------|
| 0  | dát tăng sắc tố (vết thâm) | `vet_tham`     |
| 1  | sang thương viêm           | `thuong_viem`  |
| 2  | sẹo mụn (lõm/lồi)          | `seo_mun`      |
| 3  | còi (đóng/mở)              | `coi`          |
| 4  | sang thương nang & nốt     | `thuong_nang`  |

## 5. Citation

```bibtex
@article{huynh2022acne,
  title   = {Automatic Acne Object Detection and Acne Severity Grading Using Smartphone Images and Artificial Intelligence},
  author  = {Huynh, Quan T. and others},
  journal = {Diagnostics},
  volume  = {12}, number = {8}, pages = {1879}, year = {2022},
  doi     = {10.3390/diagnostics12081879}
}
```

## 6. License

[MIT](LICENSE).

---

Maintained by [Huỳnh Thanh Quan](https://linkedin.com/in/charles-huynh) — part of the
[Curious Machine](https://curiousmachine.152-42-201-32.sslip.io) project portfolio.
