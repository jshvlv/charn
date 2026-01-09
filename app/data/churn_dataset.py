from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from app.api.schemas.churn import DatasetRowChurn


def _default_dataset_path() -> Path:
    return Path(__file__).resolve().parent / "churn_dataset.csv"


@lru_cache(maxsize=1)
def load_churn_dataframe(path: str | Path | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path is not None else _default_dataset_path()
    df = pd.read_csv(dataset_path)
    return df


def preview_rows(n: int, *, path: str | Path | None = None) -> list[DatasetRowChurn]:
    df = load_churn_dataframe(path)
    records: list[dict[str, Any]] = df.head(n).to_dict(orient="records")
    return [DatasetRowChurn(**row) for row in records]


def dataset_info(*, path: str | Path | None = None) -> dict[str, Any]:
    df = load_churn_dataframe(path)
    n_rows, n_cols = df.shape

    columns = list(df.columns)
    feature_names = [c for c in columns if c != "churn"]

    churn_counts: dict[int, int] = {}
    if "churn" in df.columns:
        vc = df["churn"].value_counts(dropna=False).sort_index()
        churn_counts = {int(k): int(v) for k, v in vc.to_dict().items()}

    return {
        "rows": int(n_rows),
        "columns": int(n_cols),
        "feature_names": feature_names,
        "churn_distribution": churn_counts,
    }