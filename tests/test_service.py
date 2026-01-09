from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient


def _assert_prediction_item(item: dict[str, Any]) -> None:
    assert set(item.keys()) == {"churn", "proba_churn_0", "proba_churn_1"}
    assert item["churn"] in (0, 1)
    for k in ("proba_churn_0", "proba_churn_1"):
        assert isinstance(item[k], float)
        assert 0.0 <= item[k] <= 1.0
    assert math.isclose(item["proba_churn_0"] + item["proba_churn_1"], 1.0, rel_tol=1e-6, abs_tol=1e-6)


def _train_logreg(client: TestClient) -> dict:
    body = {"model_type": "logreg", "hyperparameters": {"max_iter": 300, "class_weight": "balanced"}}
    resp = client.post("/model/train", params={"test_size": 0.2, "random_state": 42}, json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_health(app_client: TestClient) -> None:
    resp = app_client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "ml churn service is running"}


def test_dataset_info_and_preview(app_client: TestClient) -> None:
    info = app_client.get("/dataset/info")
    assert info.status_code == 200
    body = info.json()
    assert body["rows"] > 0 and body["columns"] > 0
    assert "churn" not in body["feature_names"]

    preview = app_client.get("/dataset/preview", params={"n": 2})
    assert preview.status_code == 200
    rows = preview.json()
    assert isinstance(rows, list) and len(rows) == 2

    bad = app_client.get("/dataset/preview", params={"n": 0})
    assert bad.status_code == 422


def test_predict_requires_trained_model(app_client: TestClient) -> None:
    resp = app_client.post(
        "/predict",
        json={
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
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "model_not_trained"


def test_train_and_status_and_predict_single(app_client: TestClient) -> None:
    train_body = _train_logreg(app_client)
    assert "accuracy" in train_body and "f1" in train_body
    assert train_body["model_type"] == "logreg"

    status = app_client.get("/model/status")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["trained"] is True
    assert status_body["model_type"] == "logreg"
    assert "hyperparameters" in status_body

    pred = app_client.post(
        "/predict",
        json={
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
    )
    assert pred.status_code == 200
    _assert_prediction_item(pred.json())


def test_train_and_predict_batch(app_client: TestClient) -> None:
    _train_logreg(app_client)

    pred = app_client.post(
        "/predict",
        json=[
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
    )
    assert pred.status_code == 200
    body = pred.json()
    assert isinstance(body, list) and len(body) == 2
    for item in body:
        _assert_prediction_item(item)


def test_predict_invalid_features(app_client: TestClient) -> None:
    _train_logreg(app_client)
    resp = app_client.post(
        "/predict",
        json={"usage_hours": 10, "region": "europe"},  # missing required fields
    )
    # Pydantic запрос отфильтрует и вернёт 422, так как обязательные поля отсутствуют.
    assert resp.status_code == 422


def test_model_schema_endpoint(app_client: TestClient) -> None:
    resp = app_client.get("/model/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert "features" in body
    names = [f["name"] for f in body["features"]]
    expected = [
        "monthly_fee",
        "usage_hours",
        "support_requests",
        "account_age_months",
        "failed_payments",
        "autopay_enabled",
        "region",
        "device_type",
        "payment_method",
    ]
    assert names == expected
    assert body["numeric_features"] and body["categorical_features"]


def test_train_error_empty_dataset(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.data.churn_dataset as churn_dataset
    import main as main_module

    # Patch all entry points to the loader used by the API.
    monkeypatch.setattr(churn_dataset, "load_churn_dataframe", lambda path=None: pd.DataFrame())
    monkeypatch.setattr(main_module, "load_churn_dataframe", lambda path=None: pd.DataFrame())

    resp = app_client.post(
        "/model/train",
        params={"test_size": 0.2, "random_state": 42},
        json={"model_type": "logreg", "hyperparameters": {"max_iter": 100}},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["message"] == "Dataset is empty"

