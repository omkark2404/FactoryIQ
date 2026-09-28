# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - FactoryIQ Architecture Overhaul
### Added
* Comprehensive CI/CD via GitHub Actions (`ci.yml`) and Dependabot.
* FastAPI `lifespan` context manager for optimized ML model loading.
* Unified configurations via `app/config.py` using `.env`.
* Security enhancements: Strict CORS, API Key Auth, and file upload validation.
* Real evaluation scripts (`scripts/evaluate.py`) producing metrics.
* `MODEL_CARD.md`, `CONTRIBUTING.md`, and complete architecture documentation.

### Changed
* Refactored entire codebase into `app/`, `models/`, `rag/`, and `scripts/`.
* Telemetry training pipeline now utilizes Scikit-Learn `Pipeline`, `StandardScaler`, and `TimeSeriesSplit`.
* Computer Vision pipeline now calibrates dynamic anomaly thresholds using 95th percentile validation loss.
* RAG pipeline strictly enforces "not found in SOPs" guardrails.
