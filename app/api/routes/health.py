from __future__ import annotations

import logging
from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from app.data.churn_dataset import load_churn_dataframe
from app.model.registry import get_status as get_model_status

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
def root() -> dict:
    return RedirectResponse(url="/docs")


@router.get("/health")
def health() -> dict:
    dataset_ok = True
    try:
        df = load_churn_dataframe()
        dataset_ok = not df.empty
    except Exception as e:  # noqa: BLE001
        logger.warning("Health check dataset load failed: %s", e)
        dataset_ok = False

    status = get_model_status()
    model_ok = bool(status.get("trained"))
    return {
        "status": "ok" if dataset_ok else "degraded",
        "model_loaded": model_ok,
        "dataset_loaded": dataset_ok,
    }

