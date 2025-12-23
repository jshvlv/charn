from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from app.data.churn_dataset import load_churn_dataframe

NUMERIC_COLUMNS = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_COLUMNS = [
    "region",
    "device_type",
    "payment_method",
]

TARGET_COLUMN = "churn"


@dataclass(frozen=True)
class SplitResult:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    numeric_columns: list[str]
    categorical_columns: list[str]


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in out.columns and out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    for col in CATEGORICAL_COLUMNS:
        if col in out.columns and out[col].isna().any():
            out[col] = out[col].fillna("unknown")

    return out


def prepare_xy(
    *,
    path: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = load_churn_dataframe(path)
    df = _fill_missing_values(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return x, y, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


def make_train_test_split(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    path: str | None = None,
) -> SplitResult:
    x, y, numeric_cols, categorical_cols = prepare_xy(path=path)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return SplitResult(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
    )


def class_distribution(y: pd.Series) -> dict[int, int]:
    vc = y.value_counts(dropna=False).sort_index()
    return {int(k): int(v) for k, v in vc.to_dict().items()}


def split_info(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    path: str | None = None,
) -> dict[str, Any]:
    split = make_train_test_split(
        test_size=test_size,
        random_state=random_state,
        path=path,
    )

    return {
        "train_size": int(len(split.x_train)),
        "test_size": int(len(split.x_test)),
        "train_churn_distribution": class_distribution(split.y_train),
        "test_churn_distribution": class_distribution(split.y_test),
        "numeric_columns": split.numeric_columns,
        "categorical_columns": split.categorical_columns,
        "target_column": TARGET_COLUMN,
        "test_size_fraction": float(test_size),
        "random_state": int(random_state),
    }


