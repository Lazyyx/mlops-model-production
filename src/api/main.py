from src.models.MTCNN import detect_faces
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from src.api.security import verify_api_key
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


@app.post("/detect", dependencies=[Depends(verify_api_key)])
async def detect(file: bytes = File(...)):
    boxes = detect_faces(file)
    boxes_list = boxes.tolist() if boxes is not None else []

    detection_result = detect_faces(file)
    if detection_result is not None and isinstance(detection_result, tuple):
        boxes = detection_result[0]
    elif detection_result is not None:
        boxes = detection_result
    else:
        boxes = None
        
    boxes_list = boxes.tolist() if boxes is not None else []
    
    return {"boxes": boxes_list}
