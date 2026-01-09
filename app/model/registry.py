from __future__ import annotations

from dataclasses import asdict

from sklearn.pipeline import Pipeline

from app.model.store import ModelMeta, load_churn_model

_pipeline: Pipeline | None = None
_meta: ModelMeta | None = None


def init_from_disk() -> None:
    global _pipeline, _meta
    _pipeline, _meta = load_churn_model()


def set_model(pipeline: Pipeline, meta: ModelMeta) -> None:
    global _pipeline, _meta
    _pipeline = pipeline
    _meta = meta


def get_status() -> dict:
    if _pipeline is None or _meta is None:
        return {"trained": False, "trained_at": None, "metrics": None}

    return {
        "trained": True,
        "trained_at": _meta.trained_at,
        "metrics": _meta.metrics,
        "version": _meta.version,
        "model_path": _meta.model_path,
        "model_type": _meta.model_type,
        "hyperparameters": _meta.hyperparameters,
        "feature_schema": _meta.feature_schema,
    }


def get_model() -> Pipeline | None:
    return _pipeline

