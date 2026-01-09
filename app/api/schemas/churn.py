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


class PredictionResponseChurn(BaseModel):
    churn: int = Field(..., ge=0, le=1, description="Predicted churn class (1 = churn, 0 = no churn).")
    proba_churn_0: float = Field(..., ge=0.0, le=1.0, description="Predicted probability for class 0.")
    proba_churn_1: float = Field(..., ge=0.0, le=1.0, description="Predicted probability for class 1.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "churn": 0,
                    "proba_churn_0": 0.82,
                    "proba_churn_1": 0.18,
                }
            ]
        }
    }


class TrainingConfigChurn(BaseModel):
    model_type: str = Field(..., description="Model type identifier (e.g. 'logreg' or 'random_forest').")
    hyperparameters: dict = Field(default_factory=dict, description="Estimator hyperparameters.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"model_type": "logreg", "hyperparameters": {"max_iter": 1000, "class_weight": "balanced"}},
                {"model_type": "random_forest", "hyperparameters": {"n_estimators": 200, "max_depth": 8}},
            ]
        }
    }


class ErrorResponse(BaseModel):
    code: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error message.")
    details: dict | list | str | None = Field(default=None, description="Optional error details.")

