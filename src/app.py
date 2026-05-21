"""
FastAPI сервис для turn detection.

Запуск локально:
    uvicorn src.app:app --reload --port 8000

Запрос:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "I want a hotel in"}'
"""
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import extract_features, features_to_dataframe


MODEL_PATH = Path(__file__).parent.parent / "models" / "turn_detector.pkl"


# Загрузка артефактов при старте приложения (один раз)
_artifacts = joblib.load(MODEL_PATH)
_model = _artifacts["model"]
_ohe = _artifacts["ohe"]
_threshold = _artifacts["threshold"]
_feature_columns = _artifacts["feature_columns"]


# Схемы request/response через Pydantic
class PredictRequest(BaseModel):
    text: str = Field(..., description="Префикс реплики пользователя",
                      examples=["I want a hotel in"])


class PredictResponse(BaseModel):
    is_end_of_turn: bool = Field(..., description="Закончил ли пользователь говорить")
    confidence: float = Field(..., description="Уверенность модели (0..1)")
    threshold: float = Field(..., description="Порог классификации")
    features: dict = Field(..., description="Извлечённые признаки")


# FastAPI app
app = FastAPI(
    title="Turn Detector API",
    description="Определяет, закончил ли пользователь свою реплику",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Healthcheck для оркестраторов (Docker, Kubernetes)."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Предсказание для одного текста."""
    features = extract_features(request.text)
    X = features_to_dataframe(features, _ohe, _feature_columns)
    
    proba = float(_model.predict_proba(X)[0, 1])
    is_end = proba > _threshold
    
    return PredictResponse(
        is_end_of_turn=bool(is_end),
        confidence=proba,
        threshold=float(_threshold),
        features=features,
    )
