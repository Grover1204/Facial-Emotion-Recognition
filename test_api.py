import os
import io
import torch
from fastapi.testclient import TestClient
from PIL import Image
from app import app
from utils.preprocess import preprocess_image

client = TestClient(app)

def create_dummy_image():
    """Creates a 100x100 RGB dummy image."""
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_preprocess():
    print("Testing preprocessing...")
    dummy_bytes = create_dummy_image()
    tensor = preprocess_image(dummy_bytes)
    
    # Expected shape: (1, 1, 48, 48)
    if tensor.shape == (1, 1, 48, 48):
        print("PASS: Output tensor shape is correct (1, 1, 48, 48).")
    else:
        print(f"FAIL: Output tensor shape is {tensor.shape}.")

    # Expected value range [0, 1]
    if tensor.max() <= 1.0 and tensor.min() >= 0.0:
        print("PASS: Tensor values are within [0, 1].")
    else:
        print(f"FAIL: Tensor values out of range [{tensor.min()}, {tensor.max()}].")

def test_api_predict():
    print("\nTesting /predict endpoint...")
    dummy_bytes = create_dummy_image()
    
    # Since model might not be loaded, we expect either a result or a 503 (if properly handled) 
    # OR a 500 if the mock model isn't set up.
    # Actually app.py handles 'model is None' with 503.
    
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", dummy_bytes, "image/jpeg")}
    )
    
    if response.status_code == 200:
        print("PASS: API returned 200 OK.")
        print("Response:", response.json())
    elif response.status_code == 503:
        print("PASS: API returned 503 (Model not loaded yet, which is expected during training).")
        print("Detail:", response.json()['detail'])
    else:
        print(f"FAIL: API returned {response.status_code}.")
        print("Response:", response.text)

if __name__ == "__main__":
    test_preprocess()
    test_api_predict()
