import io
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Load Haar Cascade functionality
# We need the xml file. OpenCV usually includes it, or we can download it.
# Ideally, we should use the one from cv2.data.haarcascades
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocesses an image for the FER model.
    1. Detect Face (if any) and Crop.
    2. Convert to Grayscale.
    3. Resize to 48x48.
    4. Normalize.
    """
    # 1. Convert bytes to numpy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        print(f"Face Detection: Found {len(faces)} faces. Cropping largest.")
        # Get the largest face
        # x, y, w, h
        largest_face = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        x, y, w, h = largest_face
        
        # Crop
        face_img = gray[y:y+h, x:x+w]
    else:
        print("Face Detection: No faces found. Using full image.")
        # No face detected, use entire image
        face_img = gray
    
    # Convert back to PIL for transforms (or continue with cv2, but sticking to PIL for consistency)
    pil_image = Image.fromarray(face_img)

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    
    # Transform returns (1, 48, 48)
    image_tensor = transform(pil_image)
    image_tensor = image_tensor.unsqueeze(0) # (1, 1, 48, 48)
    
    return image_tensor

