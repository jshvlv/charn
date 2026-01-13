from __future__ import annotations

from fastapi import APIRouter, Query

from app.data.churn_dataset import dataset_info, preview_rows
from app.features.split import split_info

router = APIRouter(prefix="/dataset")


@router.get("/preview")
def dataset_preview(n: int = Query(10, ge=1, le=100)) -> list[dict]:
    rows = preview_rows(n)
    return [r.model_dump() for r in rows]


@router.get("/info")
def dataset_details() -> dict:
    return dataset_info()


@router.get("/split-info")
def dataset_split_info(
    test_size: float = Query(0.2, gt=0.0, lt=1.0),
    random_state: int = Query(42, ge=0),
) -> dict:
    return split_info(test_size=test_size, random_state=random_state)

