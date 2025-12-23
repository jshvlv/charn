from pydantic import BaseModel, Field


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int = Field(..., ge=0, le=1)


class DatasetRowChurn(FeatureVectorChurn):
    churn: int = Field(..., ge=0, le=1)


