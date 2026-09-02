import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """تشغيل أحداث الـ lifespan لتحميل الموديل قبل بدء الاختبارات"""
    with TestClient(app) as test_client:
        yield test_client

def test_health_check(client):
    """اختبار نقطة فحص الحالة"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_predict_valid_data(client):
    """اختبار توقع سليم لبيانات راكب"""
    payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 28.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "survived" in data
    assert "survival_probability" in data
    assert data["survived"] in [0, 1]
    assert 0.0 <= data["survival_probability"] <= 1.0

def test_predict_invalid_data(client):
    """اختبار رفض البيانات الخاطئة (عمر بالسالب)"""
    payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": -5.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422