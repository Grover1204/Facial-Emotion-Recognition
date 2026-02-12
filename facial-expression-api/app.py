import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from utils.preprocess import preprocess_image
from utils.labels import EMOTION_LABELS
from model.architecture import FERModel

# --- Model Definition ---
# Imported from model.architecture


# --- App State & Lifespan ---
model = None
DEVICE = torch.device("cpu") # Inference on CPU is fine usually

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    try:
        model = FERModel()
        state_dict = torch.load("model/model.pth", map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will start but predictions will fail until model is fixed.")
    
    yield
    
    # Cleanup
    model = None

app = FastAPI(title="Facial Expression Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Facial Expression Recognition API"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded or training in progress.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        # Read and preprocess
        contents = await file.read()
        input_tensor = preprocess_image(contents)
        input_tensor = input_tensor.to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_idx = predicted_class.item()
            confidence_score = confidence.item()
            
        emotion = EMOTION_LABELS.get(predicted_idx, "Unknown")
        
        return {
            "emotion": emotion,
            "confidence": confidence_score,
            "raw_probabilities": {EMOTION_LABELS[i]: float(probabilities[0][i]) for i in range(7)}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
