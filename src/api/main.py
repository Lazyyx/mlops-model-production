# src/api/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from src.models.MTCNN import detect_faces
from src.api.security import verify_api_key, limit_session_calls
from PIL import UnidentifiedImageError, Image, ImageDraw
from io import BytesIO
from typing import List, Dict, Optional
import base64


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

@app.post("/detect/crop", dependencies=[Depends(verify_api_key)])
async def detect_crops(
    file: UploadFile = File(..., description="Image file"),
    min_face_size: int = COMMON_DETECTION_PARAMS["min_face_size"],
    threshold_pnet: float = COMMON_DETECTION_PARAMS["threshold_pnet"],
    threshold_rnet: float = COMMON_DETECTION_PARAMS["threshold_rnet"],
    threshold_onet: float = COMMON_DETECTION_PARAMS["threshold_onet"],
    score_min: float = Query(0.0, description="Optional min score filter for crops"),
):
    """
    Detect faces and return cropped faces as base64-encoded JPEGs.

    Response format:
    {
      "faces": [
        {
          "box": [x1, y1, x2, y2],
          "score": 0.98,
          "image_base64": "...."
        },
        ...
      ]
    }
    """
    img_bytes = await file.read()

    try:
        boxes, keypoints, scores = detect_faces(
            img_bytes,
            min_face_size=min_face_size,
            threshold_pnet=threshold_pnet,
            threshold_rnet=threshold_rnet,
            threshold_onet=threshold_onet,
        )
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image format")

    if not boxes:
        return {"faces": []}

    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    faces_payload: List[Dict[str, Optional[object]]] = []

    for box, score in zip(boxes, scores):
        if score is not None and score < score_min:
            continue

        try:
            x1, y1, x2, y2 = map(int, box)
        except Exception:
            continue

        crop = pil_img.crop((x1, y1, x2, y2))
        buf = BytesIO()
        crop.save(buf, format="JPEG")
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

        faces_payload.append(
            {
                "box": [x1, y1, x2, y2],
                "score": float(score) if score is not None else None,
                "image_base64": b64_str,
            }
        )

    return {"faces": faces_payload}


# ---------------------- NEW: /detect/annotated ---------------------- #

@app.post("/detect/annotated", dependencies=[Depends(verify_api_key)])
async def detect_annotated(
    file: UploadFile = File(..., description="Image file"),
    min_face_size: int = COMMON_DETECTION_PARAMS["min_face_size"],
    threshold_pnet: float = COMMON_DETECTION_PARAMS["threshold_pnet"],
    threshold_rnet: float = COMMON_DETECTION_PARAMS["threshold_rnet"],
    threshold_onet: float = COMMON_DETECTION_PARAMS["threshold_onet"],
    score_min: float = Query(0.0, description="Optional min score filter for drawing"),
    draw_keypoints: bool = Query(True, description="Whether to draw keypoints"),
    draw_scores: bool = Query(True, description="Whether to draw scores near boxes"),
):
    """
    Detect faces and return an annotated image (JPEG) with boxes, keypoints and scores drawn.

    This endpoint returns a binary JPEG image.
    In Streamlit you can simply do:

        resp = requests.post(..., files={"file": (...)}, params={...})
        st.image(resp.content)
    """
    img_bytes = await file.read()

    try:
        boxes, keypoints, scores = detect_faces(
            img_bytes,
            min_face_size=min_face_size,
            threshold_pnet=threshold_pnet,
            threshold_rnet=threshold_rnet,
            threshold_onet=threshold_onet,
        )
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image format")

    if not boxes:
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    else:
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(pil_img)

        for idx, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = map(int, box)
            except Exception:
                continue

            # Box
            draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)

            # Score
            score = scores[idx] if idx < len(scores) else None
            if draw_scores and score is not None:
                text = f"{float(score):.2f}"
                draw.text((x1, max(y1 - 10, 0)), text, fill="yellow")

            # Keypoints
            if draw_keypoints and keypoints and idx < len(keypoints) and keypoints[idx]:
                for _, pt in keypoints[idx].items():
                    try:
                        px, py = pt
                        r = 3
                        draw.ellipse((px - r, py - r, px + r, py + r), fill="red")
                    except Exception:
                        continue

    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")