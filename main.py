"""FastAPI service and minimal browser interface."""
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.train import MODEL_PATH, train_model

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train a demo model if needed, then load it at startup."""
    global model
    if not MODEL_PATH.exists():
        train_model()
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Smart Email Classifier",
    description="Educational ML demo for email spam/phishing classification.",
    version="1.0.0",
    lifespan=lifespan,
)


class EmailRequest(BaseModel):
    text: str = Field(min_length=5, max_length=5000)


@app.get("/", response_class=HTMLResponse)
def home():
    """Serve a small HTML page for trying the classifier."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Email Classifier</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 680px; margin: 50px auto;
           padding: 20px; background: #f4f6f9; color: #1f2937; }
    main { background: white; padding: 24px; border-radius: 12px; }
    textarea { width: 100%; min-height: 150px; padding: 10px;
               box-sizing: border-box; }
    button { margin-top: 12px; padding: 11px 18px; cursor: pointer; }
    #result { margin-top: 18px; white-space: pre-wrap; }
  </style>
</head>
<body><main>
  <h1>Smart Email Classifier</h1>
  <p>Enter email text to estimate its category.</p>
  <textarea id="email" maxlength="5000" placeholder="Paste email text here..."></textarea>
  <button onclick="classifyEmail()">Analyze email</button>
  <div id="result" role="status" aria-live="polite"></div>
</main>
<script>
async function classifyEmail() {
  const text = document.getElementById('email').value;
  const result = document.getElementById('result');
  result.textContent = 'Analyzing...';
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await response.json();
    if (!response.ok) {
      result.textContent = typeof data.detail === 'string'
        ? data.detail : 'Please enter between 5 and 5000 characters.';
      return;
    }
    result.textContent = 'Prediction: ' + data.prediction +
      '\\nDemo model confidence: ' + (data.confidence * 100).toFixed(1) + '%' +
      '\\n' + data.notice;
  } catch (error) {
    result.textContent = 'Unable to contact the API.';
  }
}
</script>
</body></html>"""


@app.get("/health")
def health():
    """Return a simple service health status."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: EmailRequest):
    """Predict an email category and return the model's confidence estimate."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    try:
        prediction = str(model.predict([request.text])[0])
        probabilities = model.predict_proba([request.text])[0]
        confidence = float(max(probabilities))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Unable to analyze this text")

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "notice": (
            "Educational demo only. This prediction does not prove that an "
            "email is safe or malicious."
        ),
    }
