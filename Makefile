# StockM v1.0 — common commands
# Usage: make <target>   (run from project root)

.PHONY: help install dev data features train evaluate backtest predict test lint format typecheck docs serve clean docker-build

PYTHON ?= python
PIP    ?= pip

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode
	$(PIP) install -e .

dev:  ## Install package + dev dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

data:  ## Fetch raw OHLCV data
	$(PYTHON) scripts/data/fetch_data.py

features:  ## Build the active feature set
	$(PYTHON) -m stockM.pipelines.feature_pipeline

train:  ## Run the training pipeline
	$(PYTHON) -m stockM.pipelines.training_pipeline

evaluate:  ## Evaluate the best model
	$(PYTHON) -m stockM.pipelines.evaluation_pipeline

backtest:  ## Run a historical backtest
	$(PYTHON) -m stockM.pipelines.backtest_pipeline

predict:  ## Run inference on the latest data
	$(PYTHON) -m stockM.pipelines.inference_pipeline

test:  ## Run the full test suite
	$(PYTHON) -m pytest

lint:  ## Lint the codebase (ruff)
	$(PYTHON) -m ruff check src tests scripts

format:  ## Format code (black + isort)
	$(PYTHON) -m black src tests scripts
	$(PYTHON) -m isort src tests scripts

typecheck:  ## Type-check with mypy
	$(PYTHON) -m mypy src/stockM

docs:  ## Build the documentation site
	$(PYTHON) -m mkdocs build

serve:  ## Serve the prediction API locally
	uvicorn stockM.deployment.serving:app --reload --host 0.0.0.0 --port 8000

docker-build:  ## Build the API Docker image
	docker build -f deployment/docker/Dockerfile.api -t stockM-api:1.0 .

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
