import torch
from PIL import Image
import argparse
from src.models.cnn import PeopleCountingModel
from src.data.preprocessing import get_val_transforms

def predict_single_image(image_path, model_path="models/checkpoints/best_model.pth"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load architecture and weights
    model = PeopleCountingModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Prepare image
    transform = get_val_transforms()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension
    
    # Predict
    with torch.no_grad():
        prediction = model(image_tensor)
        
    count = prediction.item()
    print(f"Image: {image_path}")
    print(f"Predicted People Count: {count:.2f} (Rounded: {int(round(count))})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()
    
    predict_single_image(args.image)