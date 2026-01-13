from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Query

from app.api.schemas.churn import ErrorResponse, TrainingConfigChurn
from app.features.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, feature_schema
from app.features.split import TARGET_COLUMN, make_train_test_split
from app.model.history import append_record, get_history
from app.model.registry import get_status as get_model_status
from app.model.registry import set_model as set_current_model
from app.model.store import save_churn_model
from app.model.training import evaluate_churn_model, train_churn_model
from app.data.churn_dataset import load_churn_dataframe

router = APIRouter(prefix="/model")
logger = logging.getLogger(__name__)


@router.post(
    "/train",
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

    logger.info("Training started: model_type=%s", config.model_type)
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
        logger.exception("Training error")
        raise HTTPException(
            status_code=400,
            detail={"code": "training_error", "message": "Failed to train model", "details": str(e)},
        ) from e

    logger.info(
        "Training finished: model_type=%s metrics=%s",
        config.model_type,
        {"accuracy": metrics.accuracy, "f1": metrics.f1, "roc_auc": metrics.roc_auc},
    )

    meta = save_churn_model(
        pipeline,
        metrics={
            "accuracy": metrics.accuracy,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
        },
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
        feature_schema=feature_schema(),
    )
    set_current_model(pipeline, meta)

    append_record(
        {
            "timestamp": meta.trained_at,
            "model_type": meta.model_type,
            "hyperparameters": meta.hyperparameters,
            "metrics": {
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc,
            },
            "version": meta.version,
            "test_size": test_size,
            "random_state": random_state,
        }
    )

    return {
        "accuracy": metrics.accuracy,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "train_size": int(len(split.x_train)),
        "test_size": int(len(split.x_test)),
        "trained_at": meta.trained_at,
        "version": meta.version,
        "model_type": meta.model_type,
        "hyperparameters": meta.hyperparameters,
    }


@router.get("/status")
def model_status() -> dict:
    return get_model_status()


@router.get("/schema")
def model_schema() -> dict:
    return {
        "features": feature_schema(),
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
    }


@router.get("/metrics")
def model_metrics(model_type: str | None = None, limit: int = Query(None, ge=1, le=100)) -> dict:
    history = get_history(model_type=model_type, limit=limit)
    latest = history[0] if history else None
    return {
        "latest": latest,
        "history": history,
    }

