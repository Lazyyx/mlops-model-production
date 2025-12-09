import io
import base64
import os
import datetime
import requests
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

API_URL = os.getenv("API_URL", "http://localhost:8000")

def display_images_side_by_side(image, boxes):
    """Display original image with detected face boxes.

    Accepts `image` as bytes or a PIL Image. `boxes` is expected to be a
    list/iterable of [x1, y1, x2, y2] coordinates (or None).
    """
    st.subheader("Detected Faces")

    # Ensure we have a PIL Image to display
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)

    if boxes:
        for box in boxes:
            try:
                x1, y1, x2, y2 = map(int, box)
            except Exception:
                # skip malformed box
                continue
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    ax.axis('off')
    st.pyplot(fig)

def send_image_to_api(image_bytes, api_url, token, filename="image.jpg", mime="image/jpeg"):
    """Send image bytes to the face detection API and return the response.

    Uses multipart/form-data with the field name `file` so FastAPI's
    `file: bytes = File(...)` handler receives the bytes correctly.
    """
    
    headers = {
        "X-API-Key": token
        }
    
    files = {"file": (filename, image_bytes, mime)}
    response = requests.post(api_url, files=files, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()

def read_image_bytes(image_input):
    """Read image bytes from uploaded file or camera input."""
    if hasattr(image_input, "getvalue"):
        # For uploaded files
        image_bytes = image_input.getvalue()
        filename = image_input.name
        mime = image_input.type
    else:
        # For camera input
        image_bytes = image_input.read()
        filename = "camera_image.jpg"
        mime = "image/jpeg"
    return image_bytes, filename, mime

def main():
    st.title("Face Detection Demo")

    with st.sidebar:
        st.header("Configuration API")

        api_key_input = st.text_input("X-API-Key (Token)", type="password", key="api_token_key")

    st.write("Choose an image to detect faces from:")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    camera_img = st.camera_input("Or take a picture with your webcam")

    image_input = None
    if uploaded_file is not None:
        image_input = uploaded_file
    elif camera_img is not None:
        image_input = camera_img

    if image_input is None:
        st.info("Upload an image or take one using the camera to start.")
        return

    image_bytes, filename, mime = read_image_bytes(image_input)

    if st.button("Detect Faces"):
        if not api_key_input:
            st.error("Veuillez entrer votre X-API-Key (Token) dans la barre latérale.")
            return
        
        with st.spinner("Sending image to detection API..."):
            try:
                detect_endpoint = f"{API_URL}/detect"
                result = send_image_to_api(image_bytes, detect_endpoint, api_key_input)
            except requests.exceptions.HTTPError as http_err:
                if http_err.response.status_code == 401:
                    st.error("Accès refusé (401). Vérifiez votre X-API-Key.")
                else:
                    st.error(f"Erreur HTTP lors de la détection : {http_err}")
            except Exception as e:
                st.error(f"Detection failed: {e}")
                return

        boxes = result.get("boxes")

        if boxes is not None:
            display_images_side_by_side(image_bytes, boxes)
        else:
            st.success("No processed image returned by API.")

if __name__ == "__main__":
    main()