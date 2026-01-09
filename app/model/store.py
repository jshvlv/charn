from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
from sklearn.pipeline import Pipeline


def _project_root() -> Path:
    # app/model/store.py -> app/model -> app -> project root
    return Path(__file__).resolve().parents[2]


ARTIFACTS_DIR = _project_root() / "artifacts" / "models"
ACTIVE_META_FILE = ARTIFACTS_DIR / "active.json"
LEGACY_MODEL_FILE = ARTIFACTS_DIR / "churn_model.joblib"


@dataclass(frozen=True)
class ModelMeta:
    version: str
    trained_at: str
    metrics: dict[str, float]
    model_path: str
    model_type: str
    hyperparameters: dict[str, Any]
    feature_schema: list[dict[str, str]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_version() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{ts}_{suffix}"


def save_churn_model(
    pipeline: Pipeline,
    *,
    metrics: dict[str, float],
    model_type: str,
    hyperparameters: dict[str, Any] | None = None,
    feature_schema: list[dict[str, str]] | None = None,
    artifacts_dir: Path | None = None,
) -> ModelMeta:
    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    version = _new_version()
    version_dir = artifacts_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    model_file = version_dir / "model.joblib"
    meta_file = version_dir / "meta.json"

    joblib.dump(pipeline, model_file)

    meta = ModelMeta(
        version=version,
        trained_at=_utc_now_iso(),
        metrics={k: float(v) for k, v in metrics.items()},
        model_path=str(model_file),
        model_type=str(model_type),
        hyperparameters=dict(hyperparameters or {}),
        feature_schema=list(feature_schema or []),
    )

    meta_file.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Ensure active metadata directory exists (useful for test-time path overrides).
    ACTIVE_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_META_FILE.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Compatibility path for simple setups (requested example path).
    joblib.dump(pipeline, artifacts_dir / "churn_model.joblib")

    return meta


def load_churn_model(
    *,
    artifacts_dir: Path | None = None,
) -> tuple[Pipeline | None, ModelMeta | None]:
    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    if ACTIVE_META_FILE.exists():
        meta_raw: dict[str, Any] = json.loads(ACTIVE_META_FILE.read_text(encoding="utf-8"))
        meta = ModelMeta(
            version=str(meta_raw["version"]),
            trained_at=str(meta_raw["trained_at"]),
            metrics={k: float(v) for k, v in dict(meta_raw.get("metrics", {})).items()},
            model_path=str(meta_raw["model_path"]),
            model_type=str(meta_raw.get("model_type", "unknown")),
            hyperparameters=dict(meta_raw.get("hyperparameters", {}) or {}),
            feature_schema=list(meta_raw.get("feature_schema", []) or []),
        )
        model_path = Path(meta.model_path)
        if model_path.exists():
            return joblib.load(model_path), meta
        # Fallback to legacy file if path moved/was deleted.
        legacy = artifacts_dir / "churn_model.joblib"
        if legacy.exists():
            return joblib.load(legacy), meta
        return None, meta

    # If no metadata exists, try legacy location only.
    legacy = artifacts_dir / "churn_model.joblib"
    if legacy.exists():
        pipeline = joblib.load(legacy)
        meta = ModelMeta(
            version="unknown",
            trained_at="unknown",
            metrics={},
            model_path=str(legacy),
            model_type="unknown",
            hyperparameters={},
            feature_schema=[],
        )
        return pipeline, meta

    return None, None

