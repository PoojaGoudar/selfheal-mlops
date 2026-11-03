PY ?= python
export PYTHONPATH := $(CURDIR)

.PHONY: help install bootstrap local api sim monitor dash mlflow test e2e lint reset docker-up docker-down k8s-build k8s-up

help:           ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/'

install:        ## create venv and install dependencies
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt

bootstrap:      ## create data + first champion model (once)
	$(PY) -m selfheal.bootstrap

local:          ## run API + simulator + monitor + dashboard locally (Ctrl-C stops all)
	bash scripts/run_local.sh

api:            ## FastAPI service on :8000
	$(PY) -m uvicorn selfheal.serving.app:app --port 8000

sim:            ## live-stream simulator
	$(PY) -m selfheal.simulator

monitor:        ## drift monitor + healing state machine
	$(PY) -m selfheal.monitoring.monitor

dash:           ## Streamlit dashboard on :8501
	$(PY) -m streamlit run selfheal/dashboard.py

mlflow:         ## MLflow UI on :5000
	mlflow server --backend-store-uri sqlite:///runtime/mlflow.db --port 5000

test:           ## unit tests (about 30 s)
	$(PY) -m pytest

e2e:            ## full headless healing loop (about 5 min)
	$(PY) -m selfheal.e2e

lint:
	ruff check selfheal tests

reset:          ## wipe all runtime state
	rm -rf runtime

docker-up:      ## whole platform in Docker
	docker compose up --build

docker-down:
	docker compose down

k8s-build:      ## build the image inside Minikube
	minikube image build -t selfheal-mlops:latest .

k8s-up:         ## deploy to Minikube
	kubectl apply -k k8s
