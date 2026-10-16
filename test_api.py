from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Smart Email Classifier" in response.text


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={"text": "Congratulations, claim your free prize now"},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] in {"legitimate", "spam_phishing"}
    assert 0 <= response.json()["confidence"] <= 1


def test_rejects_short_input():
    response = client.post("/predict", json={"text": "Hi"})
    assert response.status_code == 422


def test_rejects_empty_input():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
