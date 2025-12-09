import io
import os
from typing import List, Dict, Optional

import requests
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

API_URL_DEFAULT = os.getenv("API_URL", "http://backend:8000")


# ---------- Utils d'affichage ----------

def display_detections(
    image_bytes: bytes,
    boxes: List[List[float]],
    keypoints: Optional[List[Optional[Dict]]] = None,
    scores: Optional[List[Optional[float]]] = None,
    show_keypoints: bool = True,
    show_scores: bool = True,
):
    """
    Affiche l'image avec les bounding boxes, keypoints et scores optionnels.
    """
    if not boxes:
        st.info("No faces detected.")
        return

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)

    for idx, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = map(int, box)
        except Exception:
            continue

        # Rectangle pour la face
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Score (si dispo)
        if scores and show_scores and idx < len(scores) and scores[idx] is not None:
            ax.text(
                x1,
                max(y1 - 5, 0),
                f"{scores[idx]:.2f}",
                fontsize=10,
                color="yellow",
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )

        # Keypoints (si dispo)
        if keypoints and show_keypoints and idx < len(keypoints) and keypoints[idx]:
            for _, pt in keypoints[idx].items():
                try:
                    px, py = pt
                    circ = Circle((px, py), radius=2, color="red")
                    ax.add_patch(circ)
                except Exception:
                    continue

    ax.axis("off")
    st.pyplot(fig)


def read_image_bytes(image_input):
    """
    Lit les bytes depuis un UploadedFile (file_uploader / camera_input).
    Retourne (image_bytes, filename, mime).
    """
    if hasattr(image_input, "getvalue"):
        image_bytes = image_input.getvalue()
        filename = image_input.name or "uploaded_image.jpg"
        mime = image_input.type or "image/jpeg"
    else:
        image_bytes = image_input.read()
        filename = "camera_image.jpg"
        mime = "image/jpeg"
    return image_bytes, filename, mime


def send_image_to_api(
    image_bytes: bytes,
    api_url: str,
    token: Optional[str] = None,
    filename: str = "image.jpg",
    mime: str = "image/jpeg",
    params: Optional[Dict] = None,
):
    """
    Envoie l'image à l'API via multipart/form-data (champ 'file').
    """
    headers = {}
    if token:
        headers["X-API-Key"] = token

    files = {"file": (filename, image_bytes, mime)}

    response = requests.post(
        api_url,
        files=files,
        headers=headers,
        params=params or {},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# ---------- App principale Streamlit ----------

def main():
    st.title("Face Detection Demo (MTCNN)")

    # ---- Barre latérale : config API + paramètres de détection ----
    with st.sidebar:
        st.header("API Settings")

        api_key_input = st.text_input(
            "X-API-Key (Token)",
            type="password",
            help="Optionnel si ton API est protégée par une clé.",
        )

        st.markdown("---")
        st.header("Detection mode")

        mode_label = st.selectbox(
            "Mode",
            [
                "Faces only (boxes)",
                "Faces + keypoints",
                "Full (boxes + keypoints + scores)",
            ],
        )

        # Mapping mode → endpoint
        endpoint_map = {
            "Faces only (boxes)": "/detect",
            "Faces + keypoints": "/detect/keypoints",
            "Full (boxes + keypoints + scores)": "/detect/full",
        }
        endpoint_path = endpoint_map[mode_label]

        st.markdown("---")
        st.header("Detection parameters")

        min_face_size = st.slider(
            "Min face size (pixels)",
            min_value=10,
            max_value=200,
            value=20,
            step=5,
            help="Filtre les visages trop petits.",
        )

        threshold_pnet = st.slider(
            "PNet threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )
        threshold_rnet = st.slider(
            "RNet threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
        )
        threshold_onet = st.slider(
            "ONet threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
        )

        score_min = None
        if mode_label == "Full (boxes + keypoints + scores)":
            score_min = st.slider(
                "Min confidence score",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Filtre les détections de faible confiance.",
            )

        st.markdown("---")
        show_keypoints = st.checkbox("Show keypoints (if available)", value=True)
        show_scores = st.checkbox("Show scores (if available)", value=True)


    st.write("Choose an image to detect faces from:")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    camera_img = st.camera_input("Or take a picture with your webcam")

    image_input = uploaded_file or camera_img

    if image_input is None:
        st.info("Upload an image or take one using the camera to start.")
        return

    image_bytes, filename, mime = read_image_bytes(image_input)

    if st.button("Detect Faces"):
        
        api_url = f"{API_URL_DEFAULT.rstrip('/')}{endpoint_path}"

        params = {
            "min_face_size": min_face_size,
            "threshold_pnet": threshold_pnet,
            "threshold_rnet": threshold_rnet,
            "threshold_onet": threshold_onet,
        }
        if score_min is not None:
            params["score_min"] = score_min

        with st.spinner("Sending image to detection API..."):
            try:
                result = send_image_to_api(
                    image_bytes=image_bytes,
                    api_url=api_url,
                    token=api_key_input or None,
                    filename=filename,
                    mime=mime,
                    params=params,
                )
            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error from API: {http_err} ({http_err.response.text})")
                return
            except Exception as e:
                st.error(f"Detection failed: {e}")
                return

        boxes: List[List[float]] = []
        keypoints: Optional[List[Optional[Dict]]] = None
        scores: Optional[List[Optional[float]]] = None

        if mode_label == "Faces only (boxes)":
            boxes = result.get("boxes", [])

        elif mode_label == "Faces + keypoints":
            boxes = result.get("boxes", [])
            keypoints = result.get("keypoints", [])

        elif mode_label == "Full (boxes + keypoints + scores)":
            detections = result.get("detections", [])
            boxes = [det.get("box") for det in detections if det.get("box") is not None]
            keypoints = [det.get("keypoints") for det in detections]
            scores = [det.get("score") for det in detections]

        # Affichage graphique
        display_detections(
            image_bytes=image_bytes,
            boxes=boxes,
            keypoints=keypoints,
            scores=scores,
            show_keypoints=show_keypoints,
            show_scores=show_scores,
        )

        # Affichage brute JSON dans un expander pour debug
        with st.expander("Raw API response"):
            st.json(result)


if __name__ == "__main__":
    main()
