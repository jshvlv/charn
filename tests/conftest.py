from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """
    Isolated TestClient:
    - adds project root to sys.path
    - redirects model artifacts to a temp directory
    - clears cached dataset loader and in-memory registry
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

    churn_dataset.load_churn_dataframe.cache_clear()

    from main import app

    return TestClient(app)

