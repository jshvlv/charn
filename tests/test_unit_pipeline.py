from __future__ import annotations

import pandas as pd
import pytest

from app.features.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from app.features.split import make_train_test_split, prepare_xy
from app.model.training import evaluate_churn_model, train_churn_model


@pytest.fixture()
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "monthly_fee": 10.0,
                "usage_hours": 20.0,
                "support_requests": 1,
                "account_age_months": 6,
                "failed_payments": 0,
                "autopay_enabled": 1,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "churn": 0,
            },
            {
                "monthly_fee": 50.0,
                "usage_hours": 2.0,
                "support_requests": 5,
                "account_age_months": 2,
                "failed_payments": 3,
                "autopay_enabled": 0,
                "region": "america",
                "device_type": "desktop",
                "payment_method": "paypal",
                "churn": 1,
            },
            {
                "monthly_fee": 30.0,
                "usage_hours": 8.0,
                "support_requests": 2,
                "account_age_months": 10,
                "failed_payments": 1,
                "autopay_enabled": 1,
                "region": "asia",
                "device_type": "mobile",
                "payment_method": "card",
                "churn": 0,
            },
        ]
    )


def test_prepare_xy_with_numeric_categorical(tiny_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.data.churn_dataset as churn_dataset
    import app.features.split as split

    # почистить кэш оригинала
    if hasattr(churn_dataset.load_churn_dataframe, "cache_clear"):
        churn_dataset.load_churn_dataframe.cache_clear()

    # подменить лоадер в обоих модулях
    monkeypatch.setattr(churn_dataset, "load_churn_dataframe", lambda path=None: tiny_df.copy(), raising=True)
    monkeypatch.setattr(split, "load_churn_dataframe", lambda path=None: tiny_df.copy(), raising=True)

    x, y, num_cols, cat_cols = prepare_xy()

    assert num_cols == NUMERIC_COLUMNS
    assert cat_cols == CATEGORICAL_COLUMNS
    assert x.shape[0] == len(tiny_df)
    assert y.shape[0] == len(tiny_df)


def test_train_and_eval_pipeline(tiny_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.data.churn_dataset as churn_dataset

    monkeypatch.setattr(churn_dataset, "load_churn_dataframe", lambda path=None: tiny_df.copy(), raising=True)

    split = make_train_test_split(test_size=0.34, random_state=42)
    model = train_churn_model(
        split.x_train,
        split.y_train,
        numeric_columns=split.numeric_columns,
        categorical_columns=split.categorical_columns,
        model_type="logreg",
        hyperparameters={"max_iter": 200},
    )

    metrics = evaluate_churn_model(model, split.x_test, split.y_test)
    # Проверяем, что метрики посчитались и лежат в разумных пределах.
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    if metrics.roc_auc is not None:
        assert 0.0 <= metrics.roc_auc <= 1.0
