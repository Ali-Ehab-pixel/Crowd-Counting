import torch
import pandas as pd
from torch.utils.data import DataLoader
import os
from pathlib import Path

from src.data.dataset import PeopleCountingDataset
from src.data.preprocessing import get_val_transforms
from src.models.cnn import PeopleCountingModel
from src.evaluation.metrics import calculate_mae, calculate_rmse

def evaluate_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {DEVICE}")
    
    # 1. Setup DataLoader for the untouched test set
    test_dataset = PeopleCountingDataset(
        csv_file="data/splits/test.csv", 
        image_dir="data/raw/frames",
        transform=get_val_transforms()
    )
    # Batch size can be larger during inference since gradients aren't stored
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 2. Load the best saved checkpoint
    model = PeopleCountingModel().to(DEVICE)
    model_path = "models/checkpoints/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 3. Run Inference over the entire test set
    actuals = []
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for batch_idx, (images, counts) in enumerate(test_loader):
            images = images.to(DEVICE)
            preds = model(images)
            
            actuals.extend(counts.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            
            # Map predictions back to their original filenames
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + images.size(0)
            filenames.extend(test_dataset.data.iloc[start_idx:end_idx]["image"].tolist())
    
    # 4. Calculate Final Metrics
    mae = calculate_mae(actuals, predictions)
    rmse = calculate_rmse(actuals, predictions)
    
    print("\n--- Test Evaluation Results ---")
    print(f"Total Test Images: {len(actuals)}")
    print(f"MAE (Average Error): {mae:.2f} people")
    print(f"RMSE (Large Error Penalty): {rmse:.2f} people")
    
    # 5. Save detailed results for Error Analysis
    results_df = pd.DataFrame({
        "image": [Path(f).name for f in filenames],
        "actual": actuals,
        "predicted": [round(p, 2) for p in predictions],
        "error": [round(p - a, 2) for p, a in zip(predictions, actuals)]
    })
    
    # Sort by absolute error descending to surface the worst predictions first
    results_df["abs_error"] = results_df["error"].abs()
    results_df = results_df.sort_values(by="abs_error", ascending=False).drop(columns=["abs_error"])
    
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    results_path = "outputs/metrics/test_predictions.csv"
    results_df.to_csv(results_path, index=False)
    
    print(f"\nDetailed predictions saved to: {results_path}")

if __name__ == "__main__":
    evaluate_model()