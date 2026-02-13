import requests
import io
from PIL import Image

# 1. Define the Endpoint
URL = "http://127.0.0.1:8000/predict"
print(f"✅ Step 1: Endpoint Targeted: {URL}")

# 2. Prepare Image for Upload
print("✅ Step 2: Preparing Image Upload...")
# Create a dummy image (gray square)
img = Image.new('L', (48, 48), color=128)
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_bytes = img_byte_arr.getvalue()

files = {"file": ("demo_face.jpg", img_bytes, "image/jpeg")}
print(f"   -> Image prepared ({len(img_bytes)} bytes).")

# 3. Call the API (which calls the ML Model)
print("⏳ Step 3: Sending Request to API (Calling ML Model)...")
try:
    response = requests.post(URL, files=files)
    
    # 4. Receive JSON Response
    print("✅ Step 4: Received Response!")
    print("\n--- JSON RESPONSE ---")
    print(response.json())
    print("---------------------")
    
    if response.status_code == 200:
        print("\n🎉 SUCCESS: FastAPI is working correctly.")
    else:
        print(f"\n❌ FAILED: Status Code {response.status_code}")

except Exception as e:
    print(f"\n❌ ERROR: Could not connect to API. Is it running? {e}")
