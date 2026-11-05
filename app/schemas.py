"""Request and response models.

Pydantic validates every request before it reaches the model: wrong types,
missing fields or out-of-range values are rejected with a 422 error.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HeartFeatures(BaseModel):
    """Clinical features for one patient. Names match the Kaggle dataset columns."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
            "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
            "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
        }
    })

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia status (0-3)")


# Field order of the schema; used as a fallback if metadata.json is missing
FEATURE_ORDER = list(HeartFeatures.model_fields)


class PredictionResponse(BaseModel):
    heart_disease: bool = Field(..., description="True if heart disease is predicted")
    probability: float = Field(..., ge=0, le=1, description="Predicted probability of heart disease")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())   # allow field names starting with "model_"
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: str
    features: list[str]
    target: str
    metrics: dict
    sklearn_version: Optional[str] = None
    trained_at: Optional[str] = None
