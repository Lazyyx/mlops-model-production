# src/api/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from src.models.MTCNN import detect_faces
from src.api.security import verify_api_key, limit_session_calls
from PIL import UnidentifiedImageError

app = FastAPI(
    title="Face Detection API",
    description="Advanced MTCNN face detection service with configurable thresholds.",
    version="2.0.0",
)


@app.get("/")
async def root():
    """Root endpoint to verify the API is running."""
    return {"message": "Face Detection API is running."}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

COMMON_DETECTION_PARAMS = dict(
    min_face_size=Query(20, description="Minimum face size in pixels"),
    threshold_pnet=Query(0.6, description="PNet threshold"),
    threshold_rnet=Query(0.7, description="RNet threshold"),
    threshold_onet=Query(0.7, description="ONet threshold"),
)


@app.post("/detect", dependencies=[Depends(limit_session_calls)])
async def detect_basic(
    file: UploadFile = File(..., description="Image file"),
    min_face_size: int = COMMON_DETECTION_PARAMS["min_face_size"],
    threshold_pnet: float = COMMON_DETECTION_PARAMS["threshold_pnet"],
    threshold_rnet: float = COMMON_DETECTION_PARAMS["threshold_rnet"],
    threshold_onet: float = COMMON_DETECTION_PARAMS["threshold_onet"],
):
    """
    Basic face detection: returns only bounding boxes [x1, y1, x2, y2]
    """
    img_bytes = await file.read()

    try:
        boxes, _, _ = detect_faces(
            img_bytes,
            min_face_size=min_face_size,
            threshold_pnet=threshold_pnet,
            threshold_rnet=threshold_rnet,
            threshold_onet=threshold_onet,
        )
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image format")

    return {"boxes": boxes}


@app.post("/detect/keypoints", dependencies=[Depends(limit_session_calls)])
async def detect_with_keypoints(
    file: UploadFile = File(..., description="Image file"),
    min_face_size: int = COMMON_DETECTION_PARAMS["min_face_size"],
    threshold_pnet: float = COMMON_DETECTION_PARAMS["threshold_pnet"],
    threshold_rnet: float = COMMON_DETECTION_PARAMS["threshold_rnet"],
    threshold_onet: float = COMMON_DETECTION_PARAMS["threshold_onet"],
):
    """
    Returns bounding boxes + facial keypoints
    """
    img_bytes = await file.read()

    boxes, keypoints, _ = detect_faces(
        img_bytes,
        min_face_size=min_face_size,
        threshold_pnet=threshold_pnet,
        threshold_rnet=threshold_rnet,
        threshold_onet=threshold_onet,
    )

    return {"boxes": boxes, "keypoints": keypoints}


@app.post("/detect/full", dependencies=[Depends(limit_session_calls)])
async def detect_full(
    file: UploadFile = File(..., description="Image file"),
    min_face_size: int = COMMON_DETECTION_PARAMS["min_face_size"],
    threshold_pnet: float = COMMON_DETECTION_PARAMS["threshold_pnet"],
    threshold_rnet: float = COMMON_DETECTION_PARAMS["threshold_rnet"],
    threshold_onet: float = COMMON_DETECTION_PARAMS["threshold_onet"],
    score_min: float = Query(0.8, description="Minimum confidence required to keep detection"),
):
    """
    Full detection: boxes + keypoints + confidence score
    Automatically filters out detections below `score_min`.
    """
    img_bytes = await file.read()

    boxes, keypoints, scores = detect_faces(
        img_bytes,
        min_face_size=min_face_size,
        threshold_pnet=threshold_pnet,
        threshold_rnet=threshold_rnet,
        threshold_onet=threshold_onet,
    )

    filtered = [
        {"box": b, "score": s, "keypoints": k}
        for b, k, s in zip(boxes, keypoints, scores)
        if s is None or s >= score_min
    ]

    return {"detections": filtered}
