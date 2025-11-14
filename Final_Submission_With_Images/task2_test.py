# task2_test.py
"""
Load the saved embedding model and provide face verification utilities.
Provides:
- verify_faces(image_bytes1, image_bytes2) -> dict with similarity, decision, boxes
"""
import io
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms
import os
from typing import Tuple, List

MODEL_PATH = os.path.join("models", "inception_resnet_v1.pt")

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MTCNN for detection
mtcnn = MTCNN(keep_all=True, device=DEVICE)

def load_model():
    model = InceptionResnetV1(pretrained=None).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        # fallback to pretrained download if not saved
        model = InceptionResnetV1(pretrained='vggface2').to(DEVICE)
    model.eval()
    return model

# load once
_model = load_model()

def _read_image_bytes(image_bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')

def detect_faces_pil(pil_img: Image.Image) -> Tuple[List[Tuple[int,int,int,int]], List[Image.Image]]:
    # Returns list of boxes and list of cropped face PIL images
    boxes, probs = mtcnn.detect(pil_img)
    if boxes is None:
        return [], []
    faces = []
    cropped = []
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        cropped_img = pil_img.crop((x1, y1, x2, y2)).resize((160,160))
        cropped.append(cropped_img)
        faces.append((x1, y1, x2, y2))
    return faces, cropped

def embedding_from_pil(face_pil: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((160,160)),
        transforms.ToTensor()
    ])
    img_t = transform(face_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = _model(img_t)
    emb = emb.cpu().numpy()[0]
    # L2 normalize
    emb = emb / np.linalg.norm(emb)
    return emb

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def verify_faces(image_bytes1: bytes, image_bytes2: bytes):
    pil1 = _read_image_bytes(image_bytes1)
    pil2 = _read_image_bytes(image_bytes2)
    boxes1, crops1 = detect_faces_pil(pil1)
    boxes2, crops2 = detect_faces_pil(pil2)

    # If multiple faces, pick the largest face (area) in each image for verification
    def pick_largest_box(boxes, crops):
        if not boxes:
            return None, None
        areas = [( (b[2]-b[0])*(b[3]-b[1]), i) for i,b in enumerate(boxes)]
        _, idx = max(areas)
        return boxes[idx], crops[idx]

    b1, crop1 = pick_largest_box(boxes1, crops1)
    b2, crop2 = pick_largest_box(boxes2, crops2)

    if crop1 is None or crop2 is None:
        return {
            "error": "face_not_found",
            "message": "Could not detect face in one or both images",
            "boxes_image1": boxes1,
            "boxes_image2": boxes2
        }

    emb1 = embedding_from_pil(crop1)
    emb2 = embedding_from_pil(crop2)
    score = cosine_similarity(emb1, emb2)
    # Decision threshold: cosine similarity > 0.5~0.6 means likely same; this threshold may be tuned
    decision = "same person" if score > 0.55 else "different person"

    return {
        "similarity": score,
        "decision": decision,
        "boxes_image1": boxes1,
        "boxes_image2": boxes2
    }

# If run as a script, allow quick CLI test
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python task2_test.py image1.jpg image2.jpg")
        sys.exit(1)
    with open(sys.argv[1], "rb") as f:
        b1 = f.read()
    with open(sys.argv[2], "rb") as f:
        b2 = f.read()
    out = verify_faces(b1, b2)
    print(out)
