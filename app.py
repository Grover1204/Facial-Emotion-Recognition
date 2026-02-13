import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from utils.preprocess import preprocess_image
from utils.labels import EMOTION_LABELS
from model.architecture import FERModel

# --- Model Definition ---
# Imported from model.architecture


# --- App State & Lifespan ---
model = None
load_error = None
DEVICE = torch.device("cpu") # Inference on CPU is fine usually

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, load_error
    try:
        temp_model = FERModel()
        
        # Check potential paths
        if os.path.exists("model/model.pth"):
            model_path = "model/model.pth"
        elif os.path.exists("model.pth"):
            model_path = "model.pth"
        else:
            raise FileNotFoundError("model.pth not found in 'model/' or root.")
            
        state_dict = torch.load(model_path, map_location=DEVICE)
        temp_model.load_state_dict(state_dict)
        temp_model.to(DEVICE)
        temp_model.eval()
        model = temp_model
        print("Model loaded successfully.")
    except Exception as e:
        load_error = str(e)
        print(f"Error loading model: {e}")
        print("API will start but predictions will fail until model is fixed.")
        model = None
    
    yield
    
    # Cleanup
    model = None

app = FastAPI(title="Facial Expression Recognition API", lifespan=lifespan)

@app.get("/debug")
def debug_status():
    import os
    cwd = os.getcwd()
    files_root = os.listdir(cwd)
    files_model = os.listdir("model") if os.path.exists("model") else "model dir missing"
    model_path = "model/model.pth"
    model_size = os.path.getsize(model_path) if os.path.exists(model_path) else "File missing"
    
    return {
        "status": "Model Loaded" if model else "Model Failed",
        "error": load_error,
        "cwd": cwd,
        "files_root": files_root,
        "files_model": files_model,
        "model_file_exists": os.path.exists(model_path),
        "model_size_bytes": model_size
    }

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
