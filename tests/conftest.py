from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure project root on sys.path for module imports in tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    # Маленький воспроизводимый датасет с обоими классами.
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
                "monthly_fee": 20.0,
                "usage_hours": 15.0,
                "support_requests": 0,
                "account_age_months": 12,
                "failed_payments": 0,
                "autopay_enabled": 1,
                "region": "asia",
                "device_type": "mobile",
                "payment_method": "card",
                "churn": 0,
            },
            {
                "monthly_fee": 30.0,
                "usage_hours": 5.0,
                "support_requests": 3,
                "account_age_months": 4,
                "failed_payments": 2,
                "autopay_enabled": 0,
                "region": "europe",
                "device_type": "tablet",
                "payment_method": "paypal",
                "churn": 1,
            },
        ]
    )


def _base_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """
    Базовый TestClient с перенаправлением артефактов и очисткой кэшей.
    """
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Patch artifacts paths
    import app.model.store as store

    artifacts_dir = tmp_path / "artifacts" / "models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store, "ARTIFACTS_DIR", artifacts_dir, raising=True)
    monkeypatch.setattr(store, "ACTIVE_META_FILE", artifacts_dir / "active.json", raising=True)
    monkeypatch.setattr(store, "LEGACY_MODEL_FILE", artifacts_dir / "churn_model.joblib", raising=True)

    # Clear registry
    import app.model.registry as registry

    monkeypatch.setattr(registry, "_pipeline", None, raising=True)
    monkeypatch.setattr(registry, "_meta", None, raising=True)

    # Clear dataset cache
    import app.data.churn_dataset as churn_dataset

    if hasattr(churn_dataset.load_churn_dataframe, "cache_clear"):
        churn_dataset.load_churn_dataframe.cache_clear()

    from main import app

    return TestClient(app)


@pytest.fixture()
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Клиент без подмены датасета (использует реальный churn_dataset.csv)."""
    return _base_client(tmp_path, monkeypatch)


@pytest.fixture()
def app_client_with_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_df: pd.DataFrame
) -> TestClient:
    """Клиент с подменой загрузки датасета на маленький синтетический DataFrame."""
    import app.data.churn_dataset as churn_dataset
    import app.api.routes.model as model_routes
    import app.api.routes.health as health_routes

    client = _base_client(tmp_path, monkeypatch)

    # Очистим кеш до подмены
    if hasattr(churn_dataset.load_churn_dataframe, "cache_clear"):
        churn_dataset.load_churn_dataframe.cache_clear()

    # Подменяем загрузку датасета в нужных местах.
    monkeypatch.setattr(churn_dataset, "load_churn_dataframe", lambda path=None: synthetic_df.copy(), raising=True)
    monkeypatch.setattr(model_routes, "load_churn_dataframe", lambda path=None: synthetic_df.copy(), raising=True)
    monkeypatch.setattr(health_routes, "load_churn_dataframe", lambda path=None: synthetic_df.copy(), raising=True)

    return client

