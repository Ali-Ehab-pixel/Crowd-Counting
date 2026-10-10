"""
Run the pretrained person-detection baseline.

This script:
1. Runs the pretrained COCO YOLOv8n detector on the test set
2. Computes counting metrics against ground-truth counts
3. Compares with the existing regression model
4. Saves visualizations and error analysis
5. Generates a per-image results table

This is a PRETRAINED BASELINE — the detector was not trained on your
crowd dataset. It uses COCO-pretrained weights that know 80 object
classes, including 'person' (class 0).

Usage:
    python run_baseline.py
    python run_baseline.py --split val
    python run_baseline.py --conf 0.3 --iou 0.5
"""

import sys
sys.path.insert(0, '.')

import argparse
from pathlib import Path

from src.detection.config import load_config
from src.detection.evaluate import (
    evaluate_detection_counting,
    compare_with_regression,
)
from src.detection.error_analysis import run_error_analysis
from src.detection.visualize import visualize_batch
from src.detection.detector import PersonDetector
from src.detection.data_prep import load_count_labels, prepare_data

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Run pretrained person-detection baseline"
    )
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold override")
    parser.add_argument("--iou", type=float, default=None,
                        help="NMS IoU threshold override")
    parser.add_argument("--max-vis", type=int, default=20,
                        help="Max visualization images")
    parser.add_argument("--skip-prep", action="store_true",
                        help="Skip data preparation step")
    parser.add_argument("--skip-error", action="store_true",
                        help="Skip error analysis")
    args = parser.parse_args()

    config = load_config()

    # Override thresholds if specified
    if args.conf is not None:
        config['inference']['confidence_threshold'] = args.conf
    if args.iou is not None:
        config['inference']['nms_iou_threshold'] = args.iou

    # Step 1: Data preparation
    if not args.skip_prep:
        print("\n" + "=" * 60)
        print("STEP 1: DATA PREPARATION")
        print("=" * 60)
        prepare_data(config)

    # Step 2: Evaluate counting
    print("\n" + "=" * 60)
    print("STEP 2: PRETRAINED BASELINE EVALUATION")
    print("=" * 60)
    print("\nNOTE: This uses a pretrained COCO YOLOv8n model.")
    print("It was NOT trained on your crowd dataset.")
    print(f"Confidence threshold: {config['inference']['confidence_threshold']}")
    print(f"NMS IoU threshold: {config['inference']['nms_iou_threshold']}")

    results = evaluate_detection_counting(
        config=config,
        split=args.split,
    )

    # Step 3: Compare with regression
    print("\n" + "=" * 60)
    print("STEP 3: COMPARISON WITH REGRESSION MODEL")
    print("=" * 60)
    compare_with_regression(results, config)

    # Step 4: Sample visualizations
    print("\n" + "=" * 60)
    print("STEP 4: SAMPLE VISUALIZATIONS")
    print("=" * 60)

    split_csv = Path(config['paths']['existing_splits_dir']) / f"{args.split}.csv"
    split_df = pd.read_csv(split_csv)
    image_paths = split_df['image'].tolist()

    # Load GT counts as dict for visualize_batch
    count_map = load_count_labels(config['paths']['count_labels'])

    detector = PersonDetector(config=config)
    vis_dir = Path(config['paths']['visualization_dir'])
    visualize_batch(
        image_paths[:args.max_vis],
        detector,
        str(vis_dir),
        gt_counts=count_map,
        max_images=args.max_vis,
    )

    # Step 5: Error analysis
    if not args.skip_error:
        print("\n" + "=" * 60)
        print("STEP 5: ERROR ANALYSIS")
        print("=" * 60)
        run_error_analysis(
            config=config,
            split=args.split,
            max_vis=args.max_vis,
        )

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults directory: {config['paths']['detection_outputs_dir']}")
    print(f"Visualizations:   {config['paths']['visualization_dir']}")
    print(f"Error analysis:   {config['paths']['error_analysis_dir']}")


if __name__ == "__main__":
    main()
