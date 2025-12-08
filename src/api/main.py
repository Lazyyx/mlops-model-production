from src.models.MTCNN import detect_faces
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Face Detection API is running."}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/detect")
async def detect(file: bytes = File(...)):
    """Endpoint to detect faces in an uploaded image."""
    boxes = detect_faces(file)
    boxes_list = boxes.tolist() if boxes is not None else []

    return {"boxes": boxes_list}