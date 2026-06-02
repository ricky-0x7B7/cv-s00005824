# Target di comodità. Esegui `make help` per la lista.
.PHONY: help env kernel notebook test prepare smoke-train train eval infer baseline docker-build docker-jupyter clean

ENV_NAME := s00005824
DATA := data/processed/sh17/data.yaml
MODEL := yolov8n.pt
WEIGHTS := models/best.pt
SAMPLE := samples/example.jpg

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

env: ## Crea l'ambiente conda da environment.yml
	conda env create -f environment.yml

kernel: ## Registra il kernel Jupyter per questo ambiente
	python -m ipykernel install --user --name $(ENV_NAME) --display-name "Python ($(ENV_NAME))"

notebook: ## Avvia Jupyter Lab
	jupyter lab

test: ## Esegui la suite di test
	pytest -q

prepare: ## Costruisci lo split YOLO e data.yaml da data/raw/sh17
	python scripts/prepare_dataset.py --raw data/raw/sh17 --out data/processed/sh17

smoke-train: ## Smoke test veloce sul 2% dei dati (verifica che la pipeline giri end to end)
	python scripts/train_yolo.py --data $(DATA) --model $(MODEL) --epochs 1 --imgsz 320 --batch 8 --device auto --fraction 0.02 --name smoke_test

train: ## Addestramento completo (dispositivo auto: mps/cuda/cpu)
	python scripts/train_yolo.py --data $(DATA) --model $(MODEL) --epochs 80 --imgsz 640 --batch 16 --device auto

baseline: ## Addestra la baseline classica HOG + SVM
	python scripts/train_baseline.py --raw data/raw/sh17 --config configs/baseline_svm.yaml --output models/hog_svm_baseline.pkl

eval: ## Valuta i pesi YOLO addestrati ed esporta metriche/figure
	python scripts/evaluate_yolo.py --weights $(WEIGHTS) --data $(DATA) --device auto

infer: ## Esegui inferenza + compliance su un'immagine di esempio
	python scripts/run_inference.py --weights $(WEIGHTS) --source $(SAMPLE) --device auto

docker-build: ## Costruisci l'immagine Docker solo CPU
	docker build -t $(ENV_NAME):cpu -f Dockerfile.cpu .

docker-jupyter: ## Avvia Jupyter Lab dentro l'immagine Docker CPU
	docker run --rm -it -p 8888:8888 -v "$$(pwd)":/app $(ENV_NAME):cpu

clean: ## Rimuovi le cache e gli artefatti generati
	rm -rf .pytest_cache **/__pycache__ runs/debug
