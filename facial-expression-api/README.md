# Facial Expression Recognition API

A FastAPI-based machine learning API for detecting facial expressions from images.
Built with PyTorch and trained on the FER2013 dataset.

## Features
- **CNN Model**: Custom Convolutional Neural Network trained on 7 emotions.
- **FastAPI Backend**: High-performance async API.
- **REST Endpoint**: Single `/predict` endpoint for image analysis.
- **Docker-ready**: easy deployment structure.

## Setup

1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    pip install -r facial-expression-api/requirements.txt
    # Note: 'opencv-python-headless' is now required for face detection
    ```
3.  **Train the Model** (Required for first run):
    This will download the dataset (~200MB) from Kaggle via `kagglehub` automatically.
    ```bash
    python3 facial-expression-api/model/train.py
    ```
    *Note: Training may take 15-30 minutes on CPU.*

4.  **Run the API**:
    ```bash
    cd facial-expression-api
    uvicorn app:app --reload
    ```

## Usage

### Endpoint: `POST /predict`

**Request:** `multipart/form-data` with an image file key `file`.

**Response:**
```json
{
  "emotion": "Happy",
  "confidence": 0.98,
  "raw_probabilities": {
    "Angry": 0.01,
    "Disgust": 0.00,
    ...
  }
}
```

### Example (cURL)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/image.jpg"
```

## Project Structure
- `app.py`: Main FastAPI application.
- `model/`: Contains training script (`train.py`) and shared architecture (`architecture.py`).
- `utils/`: Helper scripts for preprocessing and labels.
