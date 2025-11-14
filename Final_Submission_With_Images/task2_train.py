# task2_train.py
"""
Prepare and save the face embedding model.
This script downloads the pretrained weights for InceptionResnetV1 and saves the model state
so that task2_test.py / FastAPI can load it quickly.
"""
import torch
from facenet_pytorch import InceptionResnetV1
import os

OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUT_DIR, "inception_resnet_v1.pt")

def main():
    print("Loading pretrained InceptionResnetV1 (this will download weights on first run)...")
    model = InceptionResnetV1(pretrained='vggface2').eval()  # or 'casia-webface'
    # Save the model state_dict for faster loading by test script
    torch.save(model.state_dict(), MODEL_PATH)
    print("Saved model state to:", MODEL_PATH)

if __name__ == "__main__":
    main()
