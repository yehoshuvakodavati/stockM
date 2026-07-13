# StockM

> **StockM v1.0** — A self-built machine learning system that predicts the next day's
> stock movement (UP/DOWN) using only historical OHLCV market data.

StockM is **not** a wrapper around OpenAI, Gemini, Claude, or any external LLM API.
It is a from-scratch ML pipeline: data ingestion → preprocessing → feature
engineering → training → evaluation → backtesting → prediction.

---

## Version Roadmap

| Version | Scope                                            | Status      |
|---------|--------------------------------------------------|-------------|
| **1.0** | Historical OHLCV data → UP/DOWN prediction       | **current** |
| 2.0     | Technical indicators                             | planned     |
| 3.0     | Fundamental analysis                             | planned     |
| 4.0     | Financial news & NLP                             | planned     |
| 5.0     | Global events & macroeconomics                   | planned     |
| 6.0     | Company relationship knowledge graph             | planned     |
| 7.0     | Complete multi-modal AI trading system           | planned     |

The architecture is modular: each future version adds a new *feature source* behind
a stable interface (`FeatureGroup`) without touching the pipelines, training, or
evaluation code.

---

## Project Structure

```
stockM/
├── configs/        Configuration (YAML) — single source of truth
├── data/           Data lake: raw, interim, processed, external, validation, backups
├── src/stockM/     Source code (installable package, src layout)
├── models/         Model artifacts: checkpoints, best, exported, archived, registry
├── experiments/    Experiment tracking: runs, metrics, params, comparisons
├── reports/        Human-readable reports per lifecycle stage
├── notebooks/      Exploration notebooks (not production)
├── api/            Prediction API (FastAPI), versioned
├── frontend/       Frontend placeholder (v7)
├── deployment/     Docker, Kubernetes (placeholder), monitoring
├── .github/        CI/CD workflows
├── docs/           Architecture, setup, developer, API, roadmap
├── tests/          Unit, integration, pipeline tests
├── scripts/        Operational entry points
├── research/       Papers, ideas, experiments, benchmarks
├── cache/          Transient caches (gitignored)
├── artifacts/      Generic build/pipeline artifacts (gitignored)
├── metadata/       Dataset & run metadata manifests
├── logs/           Runtime application logs (gitignored)
└── versioning/     Data & model version manifests
```

See `docs/architecture/` for diagrams and `docs/roadmap/` for the version plan.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the package in editable mode + dev dependencies
pip install -e ".[dev]"

# 3. Copy environment template and edit secrets
cp .env.example .env

# 4. Fetch historical OHLCV data
python scripts/data/fetch_data.py

# 5. Run the training pipeline
python -m stockM.pipelines.training_pipeline
#   or:  make train
```

---

## Common Commands

```bash
make install      # install package + dev deps
make data         # fetch & process raw data
make features     # build the active feature set
make train        # run the training pipeline
make evaluate     # evaluate the best model
make backtest     # run a historical backtest
make test         # run the full test suite
make lint         # lint + format check
make docs         # build the documentation site
make clean        # remove caches and generated artifacts
```

---

## Design Principles

- **Clean architecture** — each layer depends only on the one beneath it.
- **Config-driven** — all tunable behavior lives in `configs/`; nothing is hard-coded.
- **Reproducible** — data and pipelines are versioned (DVC + git); experiments are tracked.
- **Modular & extensible** — new data sources and feature groups plug in without rewrites.
- **Production-grade** — structured logging, monitoring, CI/CD, and deployment assets from day one.

---

## License

See [LICENSE](LICENSE).
