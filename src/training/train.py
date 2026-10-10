import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.dataset import PeopleCountingDataset
from src.data.preprocessing import get_train_transforms, get_val_transforms
from src.models.cnn import PeopleCountingModel

def train_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. Setup DataLoaders
    train_dataset = PeopleCountingDataset(
        csv_file="data/splits/train.csv", 
        image_dir="data/raw/frames",
        transform=get_train_transforms()
    )
    val_dataset = PeopleCountingDataset(
        csv_file="data/splits/val.csv", 
        image_dir="data/raw/frames",
        transform=get_val_transforms()
    )

    # Batch size 8 matches the 320x320 input size on an RTX 3050 4GB
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 2. Initialize Model, Loss, and Optimizer
    model = PeopleCountingModel().to(DEVICE)
    criterion = nn.SmoothL1Loss()
    
    # Added L2 regularization (weight_decay) to mitigate overfitting on ResNet50
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=1e-4
    )

    epochs = 20
    best_val_loss = float("inf")
    
    # Ensure checkpoint directory exists
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, counts in train_loader:
            images = images.to(DEVICE)
            counts = counts.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, counts)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)

        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, counts in val_loader:
                images = images.to(DEVICE)
                counts = counts.to(DEVICE)
                
                predictions = model(images)
                loss = criterion(predictions, counts)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 5. Checkpoint the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/checkpoints/best_model.pth")
            print("--> Best model saved.")

if __name__ == "__main__":
    train_model()