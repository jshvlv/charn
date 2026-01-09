from typing import Annotated

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query

from app.api.schemas.churn import FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn
from app.data.churn_dataset import dataset_info, load_churn_dataframe, preview_rows
from app.features.split import TARGET_COLUMN, make_train_test_split, split_info
from app.model.registry import get_model as get_trained_model
from app.model.registry import get_status as get_model_status
from app.model.registry import init_from_disk as init_model_from_disk
from app.model.registry import set_model as set_current_model
from app.model.store import save_churn_model
from app.model.training import evaluate_churn_model, train_churn_model

app = FastAPI()


@app.on_event("startup")
def _startup_load_model() -> None:
    init_model_from_disk()


@app.get("/")
def root() -> dict:
    return {"message": "ml churn service is running"}


@app.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    responses={
        400: {"description": "Model is not trained"},
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
        raise HTTPException(status_code=400, detail="Model is not trained. Train it via POST /model/train.")

    is_batch = isinstance(payload, list)
    items = payload if is_batch else [payload]

    x = pd.DataFrame([item.model_dump() for item in items])
    proba = model.predict_proba(x)
    pred = model.predict(x)

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


@app.post("/model/train")
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
        raise HTTPException(status_code=404, detail="Dataset file not found")

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    if TARGET_COLUMN not in df.columns:
        raise HTTPException(status_code=400, detail="Target column 'churn' not found in dataset")

    if df[TARGET_COLUMN].nunique(dropna=False) < 2:
        raise HTTPException(status_code=400, detail="Target column 'churn' must contain at least two classes")

    split = make_train_test_split(test_size=test_size, random_state=random_state)
    pipeline = train_churn_model(
        split.x_train,
        split.y_train,
        numeric_columns=split.numeric_columns,
        categorical_columns=split.categorical_columns,
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
    )
    metrics = evaluate_churn_model(pipeline, split.x_test, split.y_test)

    meta = save_churn_model(
        pipeline,
        metrics={"accuracy": metrics.accuracy, "f1": metrics.f1},
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
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