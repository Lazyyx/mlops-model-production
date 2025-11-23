from src.models.MTCNN import detect_faces
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

@app.post("/detect")
async def detect(file: bytes = File(...)):
    """Endpoint to detect faces in an uploaded image."""
    boxes = detect_faces(file)
    boxes_list = boxes.tolist() if boxes is not None else []

    return {"boxes": boxes_list}
