from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import uvicorn
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FakeNews AI",
    description="RoBERTa-based fake news detection API",
    version="1.0.0"
)

# CORS middleware - allows frontend from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
MODEL_PATH = './models/roberta_fake_news'
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Please run training first.")
        return False

    logger.info("Loading model for API...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    logger.info("Model loaded successfully!")
    return True

class NewsRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text field cannot be empty')
        return v.strip()

@app.on_event("startup")
async def startup_event():
    if not load_model():
        logger.warning("Server started without model. /predict endpoint will return errors.")

@app.post("/predict")
async def predict(request: NewsRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please ensure the model files exist in ./models/roberta_fake_news/"
        )
    
    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()

        label = "Fake" if prediction == 1 else "Real"
        
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                "real": probabilities[0][0].item(),
                "fake": probabilities[0][1].item()
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API and model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

# Serve results directory as static files (for charts/images)
if os.path.exists("results"):
    app.mount("/results", StaticFiles(directory="results"), name="results")

# Serve static files (HTML/CSS/JS)
@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
