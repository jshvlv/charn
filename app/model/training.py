from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class TrainingMetrics:
    accuracy: float
    f1: float
    roc_auc: float | None = None


def create_estimator(model_type: str, hyperparameters: dict) -> object:
    model_type_normalized = (model_type or "").strip().lower()
    hyperparameters = dict(hyperparameters or {})

    if model_type_normalized in {"logreg", "logistic_regression", "logisticregression"}:
        params = {"max_iter": 1000}
        params.update(hyperparameters)
        return LogisticRegression(**params)

    if model_type_normalized in {"random_forest", "rf", "randomforest", "randomforestclassifier"}:
        # Keep n_jobs=1 by default to avoid potential hangs in constrained environments.
        params = {"n_estimators": 200, "random_state": 42, "n_jobs": 1}
        params.update(hyperparameters)
        return RandomForestClassifier(**params)

    raise ValueError("Unsupported model_type. Use 'logreg' or 'random_forest'.")


def train_churn_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    model_type: str = "logreg",
    hyperparameters: dict | None = None,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ],
        remainder="drop",
    )

    model = create_estimator(model_type, hyperparameters or {})

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(x_train, y_train)
    return pipeline


def evaluate_churn_model(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> TrainingMetrics:
    y_pred = pipeline.predict(x_test)
    proba = None
    roc_auc = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(x_test)
        if proba is not None and proba.shape[1] >= 2:
            try:
                roc_auc = float(roc_auc_score(y_test, proba[:, 1]))
            except Exception:
                roc_auc = None

    return TrainingMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=roc_auc,
    )


