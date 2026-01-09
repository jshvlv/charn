from typing import Annotated

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import logging
from pathlib import Path

from app.api.schemas.churn import ErrorResponse, FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn
from app.data.churn_dataset import dataset_info, load_churn_dataframe, preview_rows
from app.features.schema import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, feature_schema
from app.features.split import TARGET_COLUMN, make_train_test_split, split_info
from app.model.registry import get_model as get_trained_model
from app.model.registry import get_status as get_model_status
from app.model.registry import init_from_disk as init_model_from_disk
from app.model.registry import set_model as set_current_model
from app.model.store import save_churn_model
from app.model.training import evaluate_churn_model, train_churn_model

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "service.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.on_event("startup")
def _startup_load_model() -> None:
    init_model_from_disk()


@app.get("/")
def root() -> dict:
    return {"message": "ml churn service is running"}


def _error_response(code: str, message: str, details=None, status_code: int = 400) -> JSONResponse:
    payload = {"code": code, "message": message, "details": details}
    return JSONResponse(status_code=status_code, content=payload)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
        payload = exc.detail
    else:
        payload = {
            "code": f"HTTP_{exc.status_code}",
            "message": str(exc.detail) if exc.detail else "HTTP error",
            "details": None,
        }
    logger.warning("HTTPException: %s", payload)
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return _error_response(
        code="internal_error",
        message="Internal server error",
        details=str(exc),
        status_code=500,
    )


@app.post(
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

    is_batch = isinstance(payload, list)
    items = payload if is_batch else [payload]

    records = [item.model_dump() for item in items]
    # Enforce stable feature set and order to match training.
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


@app.get("/dataset/preview")
def dataset_preview(n: int = Query(10, ge=1, le=100)) -> list[dict]:
    rows = preview_rows(n)
    return [r.model_dump() for r in rows]


@app.get("/dataset/info")
def dataset_details() -> dict:
    return dataset_info()


@app.get("/dataset/split-info")
def dataset_split_info(
    test_size: float = Query(0.2, gt=0.0, lt=1.0),
    random_state: int = Query(42, ge=0),
) -> dict:
    return split_info(test_size=test_size, random_state=random_state)


@app.post(
    "/model/train",
    responses={
        400: {
            "description": "Validation or training error",
            "content": {
                "application/json": {
                    "schema": ErrorResponse.model_json_schema(),
                    "examples": {
                        "missing_dataset": {
                            "summary": "Dataset not found",
                            "value": {
                                "code": "HTTP_404",
                                "message": "Dataset file not found",
                                "details": None,
                            },
                        },
                        "empty_dataset": {
                            "summary": "Dataset is empty",
                            "value": {
                                "code": "HTTP_400",
                                "message": "Dataset is empty",
                                "details": None,
                            },
                        },
                        "invalid_target": {
                            "summary": "Target missing or single class",
                            "value": {
                                "code": "HTTP_400",
                                "message": "Target column 'churn' must contain at least two classes",
                                "details": None,
                            },
                        },
                        "training_error": {
                            "summary": "Training error",
                            "value": {
                                "code": "training_error",
                                "message": "Failed to train model",
                                "details": "Some sklearn error message",
                            },
                        },
                    },
                }
            },
        },
        422: {"description": "Request validation error"},
    },
)
def model_train(
    config: Annotated[
        TrainingConfigChurn,
        Body(
            ...,
            openapi_examples={
                "logreg": {
                    "summary": "Logistic regression (balanced)",
                    "value": {
                        "model_type": "logreg",
                        "hyperparameters": {"max_iter": 1000, "class_weight": "balanced"},
                    },
                },
                "random_forest": {
                    "summary": "Random forest",
                    "value": {
                        "model_type": "random_forest",
                        "hyperparameters": {"n_estimators": 200, "max_depth": 8},
                    },
                },
            },
        ),
    ],
    test_size: float = Query(0.2, gt=0.0, lt=1.0),
    random_state: int = Query(42, ge=0),
) -> dict:
    try:
        df = load_churn_dataframe()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail={"code": "HTTP_404", "message": "Dataset file not found", "details": None}
        )

    if df.empty:
        raise HTTPException(
            status_code=400, detail={"code": "HTTP_400", "message": "Dataset is empty", "details": None}
        )

    if TARGET_COLUMN not in df.columns:
        raise HTTPException(
            status_code=400,
            detail={"code": "HTTP_400", "message": "Target column 'churn' not found in dataset", "details": None},
        )

    if df[TARGET_COLUMN].nunique(dropna=False) < 2:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "HTTP_400",
                "message": "Target column 'churn' must contain at least two classes",
                "details": None,
            },
        )

    split = make_train_test_split(test_size=test_size, random_state=random_state)
    try:
        pipeline = train_churn_model(
            split.x_train,
            split.y_train,
            numeric_columns=split.numeric_columns,
            categorical_columns=split.categorical_columns,
            model_type=config.model_type,
            hyperparameters=config.hyperparameters,
        )
        metrics = evaluate_churn_model(pipeline, split.x_test, split.y_test)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "training_error", "message": "Failed to train model", "details": str(e)},
        ) from e

    meta = save_churn_model(
        pipeline,
        metrics={"accuracy": metrics.accuracy, "f1": metrics.f1},
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
        feature_schema=feature_schema(),
    )
    set_current_model(pipeline, meta)

    return {
        "accuracy": metrics.accuracy,
        "f1": metrics.f1,
        "train_size": int(len(split.x_train)),
        "test_size": int(len(split.x_test)),
        "trained_at": meta.trained_at,
        "version": meta.version,
        "model_type": meta.model_type,
        "hyperparameters": meta.hyperparameters,
    }


@app.get("/model/status")
def model_status() -> dict:
    return get_model_status()


@app.get("/model/schema")
def model_schema() -> dict:
    return {
        "features": feature_schema(),
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
    }