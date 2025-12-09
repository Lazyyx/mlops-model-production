# src/models/MTCNN.py
from mtcnn import MTCNN
from PIL import Image
from io import BytesIO
import numpy as np
from typing import List, Dict, Tuple, Any

detector = MTCNN()


def _to_py(value: Any) -> Any:
    """
    Convertit récursivement les types numpy (np.int64, np.float64, etc.)
    en types Python natifs (int, float, list, dict).
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_py(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_py(v) for k, v in value.items()}
    return value


def detect_faces(
    image_bytes: bytes,
    min_face_size: int = 20,
    threshold_pnet: float = 0.6,
    threshold_rnet: float = 0.7,
    threshold_onet: float = 0.7,
) -> Tuple[List[List[int]], List[Dict[str, List[int]]], List[float]]:
    """
    Detect faces with customizable thresholds.

    Returns:
        boxes:     list of [x1, y1, x2, y2] (ints)
        keypoints: list of dicts {name: [x, y], ...} (ints)
        scores:    list of floats
    """
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = np.asarray(pil_img)

    detections = detector.detect_faces(
        img,
        min_face_size=min_face_size,
        threshold_pnet=threshold_pnet,
        threshold_rnet=threshold_rnet,
        threshold_onet=threshold_onet,
    )

    boxes: List[List[int]] = []
    keypoints: List[Dict[str, List[int]]] = []
    scores: List[float] = []

    for det in detections:
        box = det.get("box")
        if not box or len(box) != 4:
            continue

        x, y, w, h = box
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        boxes.append([x1, y1, x2, y2])

        # Score
        score = det.get("confidence", None)
        scores.append(float(score) if score is not None else None)

        # Keypoints
        kps = det.get("keypoints", {}) or {}
        kps_py: Dict[str, List[int]] = {}
        for name, coords in kps.items():
            cx, cy = coords
            kps_py[name] = [int(cx), int(cy)]
        keypoints.append(kps_py)

    boxes = _to_py(boxes)
    keypoints = _to_py(keypoints)
    scores = _to_py(scores)

    return boxes, keypoints, scores
