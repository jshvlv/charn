# Churn Prediction Service

## Overview
This repository contains a Python service that exposes an HTTP API for working with a churn dataset, training a churn classification model, and generating churn predictions for new customers.

## Current status
- FastAPI application is available.
- Dataset utilities are available (preview, basic info, train/test split info).
- Training endpoint supports configurable model type and hyperparameters.
- Model artifacts are saved to disk and loaded on application startup.
- `POST /predict` produces churn class and class probabilities (single object or batch).

## Requirements
- Python 3.10+

## Install
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
uvicorn main:app --reload
```

By default the service listens on `http://127.0.0.1:8000`.

## API docs
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Quick check (health)
```bash
curl -sS http://127.0.0.1:8000/
```

Expected response:
```json
{"message":"ml churn service is running"}
```

## Endpoints
### Dataset
- `GET /dataset/preview?n=10`
  - Returns first `n` rows from the dataset as JSON
  - `n` is limited to `1..100`

- `GET /dataset/info`
  - Returns dataset size, feature names, churn distribution

- `GET /dataset/split-info?test_size=0.2&random_state=42`
  - Prepares data (X/y separation, missing values handling)
  - Splits into train/test using stratified `train_test_split`
  - Returns split sizes and churn distribution for train and test

### Model
- `POST /model/train?test_size=0.2&random_state=42`
  - Trains a pipeline on the dataset using a JSON training config
  - Preprocessing:
    - numeric features: `StandardScaler`
    - categorical features: `OneHotEncoder(handle_unknown="ignore")`
  - Model: `LogisticRegression` or `RandomForestClassifier`
  - Returns `accuracy` and `f1` on the test split

Example (logreg):
```bash
curl -sS -X POST "http://127.0.0.1:8000/model/train?test_size=0.2&random_state=42" \
  -H "content-type: application/json" \
  -d '{"model_type":"logreg","hyperparameters":{"max_iter":1000,"class_weight":"balanced"}}'
```

Example (random_forest):
```bash
curl -sS -X POST "http://127.0.0.1:8000/model/train?test_size=0.2&random_state=42" \
  -H "content-type: application/json" \
  -d '{"model_type":"random_forest","hyperparameters":{"n_estimators":200,"max_depth":8}}'
```

- `GET /model/status`
  - Returns whether a model is loaded, when it was trained, metrics, model type and hyperparameters
- `GET /model/metrics?model_type=&limit=`
  - Returns latest metrics and history of trainings (filter by model_type, limit recent records)
- `GET /health`
  - Returns basic service health: dataset/model availability

### Predict
- `POST /predict`
  - Accepts a single feature vector or a list of feature vectors
  - Returns `churn`, `proba_churn_0`, `proba_churn_1`

## Project layout
- `main.py` — FastAPI entrypoint
- `app/api/schemas/` — Pydantic schemas
- `app/data/` — dataset utilities and dataset file
- `app/features/` — data preparation and splitting
- `app/model/` — training utilities
- `artifacts/` — local model artifacts (ignored by git)
- `tests/` — tests
- `docker/` — Docker-related files
- `scripts/` — helper scripts

## Docker
Build:
```bash
docker build -t churn-service .
```

Run:
```bash
docker run -p 8000:8000 churn-service
```

Check:
```bash
curl -sS http://127.0.0.1:8000/health
# Docs:
# http://127.0.0.1:8000/docs
```


