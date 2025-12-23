# Churn Prediction Service

## Overview
This repository contains a Python service that exposes an HTTP API for churn prediction. The implementation will be extended iteratively.

## Current status
- Minimal FastAPI application is available.
- Endpoint `GET /` returns a health-style message: `{"message": "ml churn service is running"}`.

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

## Quick check
```bash
curl -sS http://127.0.0.1:8000/
```

Expected response:
```json
{"message":"ml churn service is running"}
```

## Project layout
- `main.py` — FastAPI entrypoint
- `app/` — application modules (API, data, features, model, services)
- `artifacts/` — local model artifacts (ignored by git)
- `tests/` — tests
- `docker/` — Docker-related files
- `scripts/` — helper scripts


