.DEFAULT_GOAL := help
SHELL := bash

PY ?= python3
VENV ?= .venv
ACT  := source $(VENV)/bin/activate

.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN{FS=":.*##"; printf "\n\033[1mtargets\033[0m\n"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------- environment ----------
.PHONY: venv
venv:  ## Create a local virtualenv at .venv.
	$(PY) -m venv $(VENV)
	$(ACT) && pip install --upgrade pip

.PHONY: install
install: venv  ## Install dev requirements (CPU torch wheels).
	$(ACT) && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
	$(ACT) && pip install -r requirements-dev.txt

# ---------- quality ----------
.PHONY: lint
lint:  ## Ruff lint.
	$(ACT) && ruff check .

.PHONY: fmt
fmt:  ## Ruff autofix + format.
	$(ACT) && ruff check --fix .
	$(ACT) && ruff format .

.PHONY: test
test:  ## Run pytest (CPU).
	$(ACT) && pytest test/ api/tests/ -q

# ---------- ML lifecycle ----------
.PHONY: train
train:  ## Train the detector (expects ./data/{train,val,test} in COCO format).
	$(ACT) && python train.py --data ./data --epochs 100

.PHONY: train-mt
train-mt:  ## Train the single-stage multi-task model (detection + severity).
	$(ACT) && python train_multitask.py \
		--data ./data/final/f_circle/ds \
		--grade-csv ./data/final/grade1/label.csv ./data/final/grade2/label.csv \
		--model fasterrcnn_r50_fpn --severity-head attention --epochs 50

.PHONY: eval
eval:  ## Re-run evaluate-only loop on the current ./models checkpoint.
	$(ACT) && python -c "import trainer; print('Use trainer.run() with epochs=0 or call evaluate() directly.')"

# ---------- API ----------
.PHONY: api
api:  ## Run the FastAPI inference service locally on :8000.
	$(ACT) && uvicorn api.main:app --reload --port 8000

.PHONY: api-image
api-image:  ## Build the inference Docker image.
	docker build -f api/Dockerfile -t skin-detective-api .

.PHONY: api-up
api-up:  ## Start API via docker compose.
	docker compose -f api/docker-compose.yml up --build -d

.PHONY: api-down
api-down:
	docker compose -f api/docker-compose.yml down

# ---------- demo / notebooks ----------
.PHONY: lab
lab:  ## Launch JupyterLab.
	$(ACT) && jupyter lab
