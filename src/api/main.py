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
    
    detection_result = detect_faces(file)
    
    if detection_result is not None and isinstance(detection_result, tuple):
        extracted_boxes = detection_result[0]
    elif detection_result is not None:
        extracted_boxes = detection_result
    else:
        extracted_boxes = None
        
    if extracted_boxes is not None and isinstance(extracted_boxes, np.ndarray):
        boxes_list = extracted_boxes.tolist()
    else:
        boxes_list = []
        
    return {"boxes": boxes_list}
