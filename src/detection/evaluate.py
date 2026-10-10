"""
Detection and counting evaluation.

Reports both detection metrics (when bbox GT exists) and counting metrics
(when count labels exist). Compares against the existing regression model.

Detection metrics (require bbox ground truth):
    - Precision, Recall, F1 at a stated IoU threshold
    - mAP@50, mAP@50:95 (computed by YOLOv8 val)
    - Matched-box IoU distribution

Counting metrics (require count ground truth):
    - MAE, MSE, RMSE
    - Mean signed error (bias indicator)
    - MAPE (for nonzero GT counts; zero-count cases reported separately)
    - Per-image error table
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.detection.config import load_config
from src.detection.detector import PersonDetector
from src.detection.data_prep import load_count_labels


# ---------------------------------------------------------------------------
# Counting metrics
# ---------------------------------------------------------------------------

def compute_counting_metrics(gt_counts, pred_counts):
    """Compute counting metrics from aligned arrays.

    Args:
        gt_counts: array-like of ground-truth counts.
        pred_counts: array-like of predicted counts.

    Returns:
        dict of metric_name → value
    """
    gt = np.array(gt_counts, dtype=np.float64)
    pred = np.array(pred_counts, dtype=np.float64)
    errors = pred - gt
    abs_errors = np.abs(errors)

    metrics = {
        'n_images': len(gt),
        'mae': float(np.mean(abs_errors)),
        'mse': float(np.mean(errors ** 2)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mean_signed_error': float(np.mean(errors)),
        'median_signed_error': float(np.median(errors)),
        'std_error': float(np.std(errors)),
        'max_overcount': float(np.max(errors)),
        'max_undercount': float(np.min(errors)),
    }

    # MAPE for nonzero GT
    nonzero_mask = gt > 0
    n_nonzero = int(np.sum(nonzero_mask))
    n_zero = int(np.sum(~nonzero_mask))

    if n_nonzero > 0:
        mape = float(np.mean(abs_errors[nonzero_mask] / gt[nonzero_mask]) * 100)
        metrics['mape_percent'] = mape
        metrics['mape_n_images'] = n_nonzero
    else:
        metrics['mape_percent'] = None
        metrics['mape_n_images'] = 0

    metrics['n_zero_gt_excluded_from_mape'] = n_zero

    # Counting accuracy: fraction within tolerance of 1 person
    within_1 = float(np.mean(abs_errors <= 1) * 100)
    within_2 = float(np.mean(abs_errors <= 2) * 100)
    within_5 = float(np.mean(abs_errors <= 5) * 100)
    metrics['accuracy_within_1'] = within_1
    metrics['accuracy_within_2'] = within_2
    metrics['accuracy_within_5'] = within_5

    return metrics


def build_per_image_table(image_names, gt_counts, pred_counts):
    """Build per-image error table.

    Returns:
        pd.DataFrame with columns: Image, GT_Count, Pred_Count,
            Signed_Error, Abs_Error, Relative_Error
    """
    gt = np.array(gt_counts, dtype=np.float64)
    pred = np.array(pred_counts, dtype=np.float64)
    errors = pred - gt
    abs_errors = np.abs(errors)

    rel_errors = []
    for g, ae in zip(gt, abs_errors):
        if g > 0:
            rel_errors.append(round(float(ae / g * 100), 2))
        else:
            rel_errors.append(None)  # undefined for zero GT

    df = pd.DataFrame({
        'Image': image_names,
        'GT_Count': gt.astype(int).tolist(),
        'Pred_Count': pred.astype(int).tolist(),
        'Signed_Error': errors.astype(int).tolist(),
        'Abs_Error': abs_errors.astype(int).tolist(),
        'Relative_Error_%': rel_errors,
    })

    # Sort by absolute error descending
    df = df.sort_values('Abs_Error', ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_detection_counting(config=None, custom_model_path=None,
                                split='test', save_results=True):
    """Evaluate detection-based counting on a dataset split.

    Runs the detector on all images in the specified split, computes
    counting metrics against ground-truth counts, and saves results.

    Detection metrics (precision, recall, mAP) are NOT computed here
    because they require bounding-box ground truth, which may not exist.
    Those are reported as unavailable unless bbox labels are present.

    Args:
        config: Config dict or None.
        custom_model_path: Path to custom-trained model, or None for pretrained.
        split: 'test', 'val', or 'train'.
        save_results: Whether to save CSV and JSON outputs.

    Returns:
        dict with metrics and per-image results.
    """
    if config is None:
        config = load_config()

    paths = config['paths']
    output_dir = Path(paths['detection_outputs_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load count labels
    count_map = load_count_labels(paths['count_labels'])

    # Load split CSV (existing format: image,count)
    split_csv = Path(paths['existing_splits_dir']) / f"{split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split file not found: {split_csv}")

    split_df = pd.read_csv(split_csv)

    # Initialize detector
    print(f"Loading detector...")
    model_label = "custom" if custom_model_path else "pretrained COCO"
    print(f"  Model: {model_label}")
    detector = PersonDetector(config=config,
                              custom_model_path=custom_model_path)

    # Run detection on all images
    image_names = []
    gt_counts = []
    pred_counts = []

    print(f"\nRunning detection on {len(split_df)} {split} images...")
    for i, row in split_df.iterrows():
        img_path = row['image']
        gt_count = int(row['count'])

        det = detector.detect(img_path)

        image_names.append(Path(img_path).name)
        gt_counts.append(gt_count)
        pred_counts.append(det['count'])

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(split_df)} images...")

    # Compute metrics
    print(f"\n{'='*60}")
    print(f"DETECTION COUNTING EVALUATION ({split} set)")
    print(f"{'='*60}")

    metrics = compute_counting_metrics(gt_counts, pred_counts)

    print(f"\nCounting Metrics:")
    print(f"  Images:           {metrics['n_images']}")
    print(f"  MAE:              {metrics['mae']:.2f}")
    print(f"  RMSE:             {metrics['rmse']:.2f}")
    print(f"  Mean Signed Err:  {metrics['mean_signed_error']:+.2f} "
          f"({'overcount bias' if metrics['mean_signed_error'] > 0 else 'undercount bias'})")
    print(f"  Std Error:        {metrics['std_error']:.2f}")
    print(f"  Max Overcount:    {metrics['max_overcount']:+.0f}")
    print(f"  Max Undercount:   {metrics['max_undercount']:+.0f}")
    if metrics['mape_percent'] is not None:
        print(f"  MAPE:             {metrics['mape_percent']:.1f}% "
              f"({metrics['mape_n_images']} images, "
              f"{metrics['n_zero_gt_excluded_from_mape']} zero-GT excluded)")
    print(f"  Within ±1:        {metrics['accuracy_within_1']:.1f}%")
    print(f"  Within ±2:        {metrics['accuracy_within_2']:.1f}%")
    print(f"  Within ±5:        {metrics['accuracy_within_5']:.1f}%")

    # Detection metrics disclaimer
    labels_dir = Path(paths['bbox_labels_dir'])
    has_bbox_labels = labels_dir.exists() and any(labels_dir.glob('*.txt'))

    print(f"\nDetection Metrics (Precision/Recall/mAP):")
    if has_bbox_labels:
        print("  Bounding-box labels found. Run YOLOv8 val for full detection metrics.")
    else:
        print("  UNAVAILABLE — No bounding-box ground truth exists.")
        print("  Cannot compute precision, recall, F1, or mAP from count labels alone.")
        print("  Complete the annotation workflow to enable detection evaluation.")

    # Per-image table
    table = build_per_image_table(image_names, gt_counts, pred_counts)

    if save_results:
        # Save metrics
        metrics_path = output_dir / f"{split}_detection_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")

        # Save per-image table
        table_path = output_dir / f"{split}_detection_predictions.csv"
        table.to_csv(table_path, index=False)
        print(f"Per-image results saved to {table_path}")

    return {
        'metrics': metrics,
        'table': table,
        'gt_counts': gt_counts,
        'pred_counts': pred_counts,
        'image_names': image_names,
    }


# ---------------------------------------------------------------------------
# Compare with regression model
# ---------------------------------------------------------------------------

def compare_with_regression(detection_results, config=None):
    """Compare detection counting with existing regression predictions.

    Loads the regression model's test predictions and computes the same
    metrics for side-by-side comparison.

    Args:
        detection_results: Output dict from evaluate_detection_counting().
        config: Config dict.

    Returns:
        pd.DataFrame comparison summary.
    """
    if config is None:
        config = load_config()

    reg_predictions_path = Path("outputs/metrics/test_predictions.csv")
    if not reg_predictions_path.exists():
        print("Regression predictions not found. Skipping comparison.")
        return None

    reg_df = pd.read_csv(reg_predictions_path)

    # Regression metrics from existing predictions
    reg_gt = reg_df['actual'].values
    reg_pred = reg_df['predicted'].values
    reg_metrics = compute_counting_metrics(reg_gt, reg_pred)

    det_metrics = detection_results['metrics']

    # Comparison table
    metric_names = ['mae', 'rmse', 'mean_signed_error', 'std_error',
                    'max_overcount', 'max_undercount',
                    'accuracy_within_1', 'accuracy_within_2',
                    'accuracy_within_5']

    rows = []
    for m in metric_names:
        rows.append({
            'Metric': m,
            'Regression': round(reg_metrics.get(m, 0), 2),
            'Detection': round(det_metrics.get(m, 0), 2),
        })

    comp_df = pd.DataFrame(rows)

    print(f"\n{'='*60}")
    print("REGRESSION vs DETECTION COMPARISON")
    print(f"{'='*60}")
    print(comp_df.to_string(index=False))

    # Save comparison
    output_dir = Path(config['paths']['detection_outputs_dir'])
    comp_path = output_dir / "regression_vs_detection.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\nComparison saved to {comp_path}")

    return comp_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate detection counting")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model", default=None, help="Custom model path")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with regression model")
    args = parser.parse_args()

    results = evaluate_detection_counting(
        split=args.split,
        custom_model_path=args.model,
    )

    if args.compare:
        compare_with_regression(results)
