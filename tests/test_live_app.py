from __future__ import annotations

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30.0


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
        yield c


def test_health(client: httpx.Client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "ml churn service is running"}


def test_model_schema(client: httpx.Client) -> None:
    resp = client.get("/model/schema")
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
    assert set(body["numeric_features"]) == {
        "monthly_fee",
        "usage_hours",
        "support_requests",
        "account_age_months",
        "failed_payments",
        "autopay_enabled",
    }
    assert set(body["categorical_features"]) == {"region", "device_type", "payment_method"}


def test_dataset_endpoints(client: httpx.Client) -> None:
    info = client.get("/dataset/info")
    assert info.status_code == 200
    body = info.json()
    assert body["rows"] > 0
    assert body["columns"] > 0

    prev = client.get("/dataset/preview", params={"n": 3})
    assert prev.status_code == 200
    rows = prev.json()
    assert isinstance(rows, list) and len(rows) == 3


def test_train_and_status_and_predict_live(client: httpx.Client) -> None:
    train = client.post(
        "/model/train",
        params={"test_size": 0.2, "random_state": 42},
        json={"model_type": "logreg", "hyperparameters": {"max_iter": 300, "class_weight": "balanced"}},
    )
    assert train.status_code == 200, train.text
    train_body = train.json()
    assert "accuracy" in train_body and "f1" in train_body

    status = client.get("/model/status")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["trained"] is True
    assert status_body["model_type"] == "logreg"

    pred = client.post(
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
    assert pred.status_code == 200, pred.text
    body = pred.json()
    assert set(body.keys()) == {"churn", "proba_churn_0", "proba_churn_1"}
    assert 0.0 <= body["proba_churn_0"] <= 1.0 and 0.0 <= body["proba_churn_1"] <= 1.0
    assert abs(body["proba_churn_0"] + body["proba_churn_1"] - 1.0) < 1e-6

