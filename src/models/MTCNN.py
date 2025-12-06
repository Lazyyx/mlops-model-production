from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

# Initialize face detector
mtcnn = MTCNN(keep_all=True)

# # Load your image
# image_path = "data/test1.jpg"
# image = Image.open(image_path)

# # Detect faces
# boxes, _ = mtcnn.detect(image)

# # boxes is a NumPy array of shape (num_faces, 4)
# # Each box is [x1, y1, x2, y2]
# print("Detected bounding boxes:", boxes)

# # Optional: visualize results
# fig, ax = plt.subplots()
# ax.imshow(image)
# if boxes is not None:
#     for box in boxes:
#         x1, y1, x2, y2 = box
#         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                  linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
# plt.show()

def detect_faces(image_bytes):
    """Detect faces in an image and return bounding boxes."""
    image = Image.open(BytesIO(image_bytes))
    boxes, _ = mtcnn.detect(image)
    return boxes