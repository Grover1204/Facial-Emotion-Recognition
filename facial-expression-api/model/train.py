import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import kagglehub

# --- Configuration ---
DATASET_HANDLE = "ahmedmoorsy/facial-expression"
MODEL_PATH = "model/model.pth"
WIDTH, HEIGHT = 48, 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Data Downloading & Loading ---
def load_data():
    """Downloads dataset via kagglehub and loads it."""
    print("Downloading dataset using kagglehub...")
    try:
        # Download latest version
        path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"Dataset downloaded to: {path}")
        
        # Find the CSV file
        csv_file = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith("fer2013.csv") or file.endswith("facial-expression.csv") or file == "fer2013.csv":
                    csv_file = os.path.join(root, file)
                    break
        
        if not csv_file:
            # Fallback check if the user provided CSV exists locally
            if os.path.exists("fer2013.csv"):
                csv_file = "fer2013.csv"
            else:
                 # Check if the file is just named differently in the download
                files = os.listdir(path)
                if len(files) > 0 and files[0].endswith('.csv'):
                    csv_file = os.path.join(path, files[0])
                else:
                    print("ERROR: Could not find a CSV file in the downloaded dataset.")
                    return None, None, None, None

        print(f"Loading data from {csv_file}...")
        data = pd.read_csv(csv_file)
        
    except Exception as e:
         print(f"Error downloading or loading dataset: {e}")
         return None, None, None, None

    print("Processing pixels...")
    # The dataset might have 'pixels' or similar column. fer2013 has 'pixels'
    if 'pixels' not in data.columns:
        print(f"Error: CSV columns are {data.columns}. Expected 'pixels' column.")
        return None, None, None, None

    pixels = data['pixels'].tolist()
    X = []
    
    # Pre-allocate array for speed if possible, but list append is fine for this size
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        X.append(np.array(face, dtype=np.uint8))
    
    X = np.array(X)
    y = data['emotion'].values
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_val, y_train, y_val

# --- 2. Dataset Class ---
class FERDataset(Dataset):
    def __init__(self, pixels, emotions, transform=None):
        self.pixels = pixels
        self.emotions = emotions
        self.transform = transform

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, idx):
        image = self.pixels[idx].reshape(48, 48).astype(np.uint8)
        # Convert to PIL Image for transforms
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = int(self.emotions[idx])
        return image, label

# --- 3. Model Definition ---
try:
    from model.architecture import FERModel
except ImportError:
    from architecture import FERModel

# FERModel definition removed (imported)


# --- 4. Training Loop ---
def train_model():
    X_train, X_val, y_train, y_val = load_data()
    if X_train is None: 
        print("Failed to load data. Exiting.")
        return

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(), # Scaling to [0,1] happens here
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = FERDataset(X_train, y_train, transform=train_transform)
    val_dataset = FERDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FERModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {DEVICE}...")
    best_val_loss = float('inf')
    
    # Create model dir if not exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Downloading model checkpoint...") 
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model.")

    print("Training complete.")

if __name__ == "__main__":
    train_model()
