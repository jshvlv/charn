from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str  # "float" | "int" | "str"
    kind: str  # "numeric" | "categorical"


FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec(name="monthly_fee", dtype="float", kind="numeric"),
    FeatureSpec(name="usage_hours", dtype="float", kind="numeric"),
    FeatureSpec(name="support_requests", dtype="int", kind="numeric"),
    FeatureSpec(name="account_age_months", dtype="int", kind="numeric"),
    FeatureSpec(name="failed_payments", dtype="int", kind="numeric"),
    FeatureSpec(name="autopay_enabled", dtype="int", kind="numeric"),
    FeatureSpec(name="region", dtype="str", kind="categorical"),
    FeatureSpec(name="device_type", dtype="str", kind="categorical"),
    FeatureSpec(name="payment_method", dtype="str", kind="categorical"),
]


FEATURE_COLUMNS: list[str] = [s.name for s in FEATURE_SPECS]
NUMERIC_COLUMNS: list[str] = [s.name for s in FEATURE_SPECS if s.kind == "numeric"]
CATEGORICAL_COLUMNS: list[str] = [s.name for s in FEATURE_SPECS if s.kind == "categorical"]


def feature_schema() -> list[dict[str, str]]:
    return [{"name": s.name, "type": s.dtype, "kind": s.kind} for s in FEATURE_SPECS]

