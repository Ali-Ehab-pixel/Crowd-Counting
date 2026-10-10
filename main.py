import argparse
import sys
from pathlib import Path

# Ensure the src directory is in the Python path
sys.path.append(str(Path(__file__).parent))

# Import your pipeline modules
from src.data.split import create_splits
from src.training.train import train_model
from src.evaluation.evaluate import evaluate_model
from src.inference.predict import predict_single_image

def main():
    parser = argparse.ArgumentParser(description="Image-Based People Counting Pipeline")
    
    # Define the core commands — original regression + new detection commands
    parser.add_argument(
        "command", 
        choices=[
            # Original regression pipeline
            "split", "train", "evaluate", "predict",
            # Detection pipeline
            "detect", "detect-eval", "detect-train",
            "detect-errors", "data-prep",
        ], 
        help="Pipeline step to execute"
    )
    
    # Optional arguments for specific commands
    parser.add_argument(
        "--image", 
        type=str, 
        help="Path to the image file (required for 'predict'/'detect' commands)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Custom detection model path (for detect commands)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help="Dataset split for evaluation (default: test)"
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare detection with regression model"
    )

    args = parser.parse_args()

    # --- Original regression commands ---
    if args.command == "split":
        print("--- Starting Data Split ---")
        create_splits()
        print("--- Data Split Complete ---")

    elif args.command == "train":
        print("--- Starting Model Training ---")
        train_model()
        print("--- Training Complete ---")

    elif args.command == "evaluate":
        print("--- Starting Test Evaluation ---")
        evaluate_model()
        print("--- Evaluation Complete ---")

    elif args.command == "predict":
        if not args.image:
            print("Error: The 'predict' command requires the --image argument.")
            print("Example: python main.py predict --image data/raw/frames/0001.jpg")
            sys.exit(1)
            
        print(f"--- Running Inference on {args.image} ---")
        predict_single_image(args.image)
        print("--- Inference Complete ---")

    # --- Detection pipeline commands ---
    elif args.command == "detect":
        from src.detection.predict_detection import predict_detection
        if not args.image:
            print("Error: The 'detect' command requires the --image argument.")
            print("Example: python main.py detect --image data/raw/frames/seq_000001.jpg")
            sys.exit(1)
        print(f"--- Running Person Detection on {args.image} ---")
        predict_detection(args.image, custom_model_path=args.model, conf=args.conf)
        print("--- Detection Complete ---")

    elif args.command == "detect-eval":
        from src.detection.evaluate import evaluate_detection_counting, compare_with_regression
        print(f"--- Evaluating Detection on {args.split} set ---")
        results = evaluate_detection_counting(
            split=args.split, custom_model_path=args.model,
        )
        if args.compare:
            compare_with_regression(results)
        print("--- Detection Evaluation Complete ---")

    elif args.command == "detect-train":
        from src.detection.train_detector import train_detector
        print("--- Starting Custom Detector Training ---")
        train_detector()
        print("--- Detector Training Complete ---")

    elif args.command == "detect-errors":
        from src.detection.error_analysis import run_error_analysis
        print(f"--- Running Error Analysis on {args.split} set ---")
        run_error_analysis(split=args.split, custom_model_path=args.model)
        print("--- Error Analysis Complete ---")

    elif args.command == "data-prep":
        from src.detection.data_prep import prepare_data
        print("--- Running Data Preparation ---")
        prepare_data()
        print("--- Data Preparation Complete ---")

if __name__ == "__main__":
    main()