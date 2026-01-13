from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from app.api.schemas.churn import ErrorResponse, FeatureVectorChurn, PredictionResponseChurn
from app.features.schema import FEATURE_COLUMNS
from app.model.registry import get_model as get_trained_model

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    responses={
        400: {
            "description": "Validation or prediction error",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "examples": {
                        "model_not_trained": {
                            "summary": "Model not trained",
                            "value": {
                                "code": "model_not_trained",
                                "message": "Model is not trained. Train it via POST /model/train.",
                                "details": None,
                            },
                        },
                        "missing_features": {
                            "summary": "Missing required features",
                            "value": {
                                "code": "invalid_features",
                                "message": "Missing required features",
                                "details": {"missing_columns": ["monthly_fee", "usage_hours"]},
                            },
                        },
                    }
                }
            },
        },
    },
)
def predict(
    payload: Annotated[
        FeatureVectorChurn | list[FeatureVectorChurn],
        Body(
            ...,
            openapi_examples={
                "single_customer": {
                    "summary": "Single customer",
                    "value": {
                        "monthly_fee": 19.99,
                        "usage_hours": 12.5,
                        "support_requests": 1,
                        "account_age_months": 6,
                        "failed_payments": 0,
                        "region": "europe",
                        "device_type": "mobile",
                        "payment_method": "card",
                        "autopay_enabled": 1,
                    },
                },
                "batch": {
                    "summary": "Batch",
                    "value": [
                        {
                            "monthly_fee": 9.99,
                            "usage_hours": 27.92,
                            "support_requests": 1,
                            "account_age_months": 14,
                            "failed_payments": 1,
                            "region": "america",
                            "device_type": "desktop",
                            "payment_method": "card",
                            "autopay_enabled": 1,
                        },
                        {
                            "monthly_fee": 49.99,
                            "usage_hours": 2.0,
                            "support_requests": 5,
                            "account_age_months": 2,
                            "failed_payments": 3,
                            "region": "asia",
                            "device_type": "mobile",
                            "payment_method": "paypal",
                            "autopay_enabled": 0,
                        },
                    ],
                },
            },
        ),
    ],
) -> PredictionResponseChurn | list[PredictionResponseChurn]:
    model = get_trained_model()
    if model is None:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "model_not_trained",
                "message": "Model is not trained. Train it via POST /model/train.",
                "details": None,
            },
        )

    logger.info(
        "Predict request: batch=%s items=%d",
        isinstance(payload, list),
        len(payload) if isinstance(payload, list) else 1,
    )

    is_batch = isinstance(payload, list)
    items = payload if is_batch else [payload]

    records = [item.model_dump() for item in items]
    x = pd.DataFrame.from_records(records, columns=FEATURE_COLUMNS)
    missing_cols = x.columns[x.isna().any()].tolist()
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_features",
                "message": "Missing required features",
                "details": {"missing_columns": missing_cols},
            },
        )

    try:
        proba = model.predict_proba(x)
        pred = model.predict(x)
    except ValueError as e:
        logger.exception("Prediction error")
        raise HTTPException(
            status_code=400,
            detail={"code": "prediction_error", "message": "Failed to run prediction", "details": str(e)},
        ) from e

    responses: list[PredictionResponseChurn] = []
    for i in range(len(items)):
        responses.append(
            PredictionResponseChurn(
                churn=int(pred[i]),
                proba_churn_0=float(proba[i][0]),
                proba_churn_1=float(proba[i][1]),
            )
        )

    return responses if is_batch else responses[0]

