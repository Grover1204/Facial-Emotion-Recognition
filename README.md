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
    pip install -r requirements.txt
    # Note: 'opencv-python-headless' is now required for face detection
    ```
3.  **Train the Model** (Required for first run):
    This will download the dataset (~200MB) from Kaggle via `kagglehub` automatically.
    ```bash
    python3 model/train.py
    ```
    *Note: Training may take 15-30 minutes on CPU.*

4.  **Run the API**:
    ```bash
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

## Deployment

### Hugging Face Spaces (Docker)
This project is ready for deployment on Hugging Face Spaces.

1.  **Create a New Space**:
    -   Go to [Hugging Face Spaces](https://huggingface.co/spaces).
    -   Select **Docker** as the SDK.
    -   Choose a name (e.g., `facial-emotion-api`).

2.  **Upload Code**:
    -   Push this repository to the Space (using git).
    -   **Important**: Ensure `model/model.pth` is included. Since it is large, use Git LFS:
        ```bash
        git lfs install
        git lfs track "model/model.pth"
        git add .gitattributes
        git commit -m "Add model with LFS"
        git push space main
        ```
    -   Alternatively, upload the file manually via the "Files" tab on Hugging Face.

3.  **Access the API**:
    -   Your API will be live at `https://<your-username>-<space-name>.hf.space/docs`.
    -   The `/predict` endpoint will be available for public use.

