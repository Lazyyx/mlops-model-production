from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with the specific origin of your web UI if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")


class Review(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float

@app.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(review: Review):
    """Analyze the sentiment of a given movie review."""
    result = sentiment_pipeline(review.text)[0]
    return SentimentResponse(label=result['label'], score=result['score'])

@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}