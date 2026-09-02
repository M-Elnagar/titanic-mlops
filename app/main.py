from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from contextlib import asynccontextmanager

# قاموس لتخزين الموديل في الذاكرة
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # تحميل الموديل عند بدء تشغيل السيرفر
    try:
        ml_models["titanic_pipeline"] = joblib.load("models/titanic_model.joblib")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield
    # تنظيف الموارد عند إيقاف السيرفر
    ml_models.clear()

app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

# 1. Pydantic Schema للتحقق من المدخلات
class PassengerData(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class: 1, 2, or 3")
    Sex: str = Field(..., pattern="^(male|female)$", description="Gender: male or female")
    Age: float = Field(..., ge=0, le=120, description="Age in years")
    SibSp: int = Field(..., ge=0, description="Number of siblings / spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents / children aboard")
    Fare: float = Field(..., ge=0, description="Passenger fare")

# 2. Pydantic Schema لشكل الـ Response
class PredictionResponse(BaseModel):
    survived: int
    survival_probability: float

# Endpoint 1: فحص حالة السيرفر (Health Check)
@app.get("/health")
def health_check():
    model_loaded = "titanic_pipeline" in ml_models
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

# Endpoint 2: التوقع (Inference)
@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerData):
    model = ml_models.get("titanic_pipeline")
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # تحويل البيانات إلى DataFrame بنفس الفورمات اللي الموديل اتدرب عليه
    input_data = pd.DataFrame([{
        "Pclass": passenger.Pclass,
        "Sex": 0 if passenger.Sex == "male" else 1,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return PredictionResponse(
        survived=int(prediction),
        survival_probability=round(float(probability), 4)
    )