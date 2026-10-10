"""
Error analysis for detection-based counting.

Identifies and saves diagnostic examples:
- Large undercounting / overcounting
- Dense scenes
- Small people
- Overlapping detections
- Low-confidence predictions
- Possible background false positives

Without bounding-box ground truth, these are labeled as diagnostic candidates
requiring human review, not confirmed errors.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detection.config import load_config
from src.detection.detector import PersonDetector
from src.detection.visualize import draw_detections, create_error_grid
from src.detection.data_prep import load_count_labels


def analyze_detections(detections_list, image_paths, gt_counts=None,
                       config=None):
    """Analyze detection results and flag diagnostic cases.

    Args:
        detections_list: List of detection dicts from detector.
        image_paths: List of image paths (aligned).
        gt_counts: Optional list of GT counts (aligned).
        config: Config dict.

    Returns:
        dict with categorized diagnostic cases.
    """
    if config is None:
        config = load_config()

    analysis = {
        'large_undercount': [],     # GT - pred >= 5
        'large_overcount': [],      # pred - GT >= 5
        'dense_scenes': [],         # many detections (>40)
        'small_people': [],         # many small boxes
        'overlapping_detections': [],  # high overlap between boxes
        'low_confidence': [],       # many low-conf detections
        'possible_false_positives': [],  # candidate FPs (diagnostic)
    }

    for i, det in enumerate(detections_list):
        img_name = Path(image_paths[i]).name
        boxes = det['bounding_boxes']
        scores = det['confidence_scores']
        count = det['count']
        gt = gt_counts[i] if gt_counts is not None else None

        entry = {
            'image': img_name,
            'image_path': str(image_paths[i]),
            'pred_count': count,
            'gt_count': gt,
        }

        # Count-based diagnostics (require GT)
        if gt is not None:
            error = count - gt
            entry['signed_error'] = int(error)

            if error <= -5:
                analysis['large_undercount'].append(entry.copy())
            if error >= 5:
                analysis['large_overcount'].append(entry.copy())

        # Dense scene
        if count >= 40:
            entry['category'] = 'dense'
            analysis['dense_scenes'].append(entry.copy())

        # Small people detection
        if len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            small_count = int(np.sum(areas < 1000))  # < ~32x32 pixels
            if small_count >= 3:
                entry['n_small_boxes'] = small_count
                analysis['small_people'].append(entry.copy())

        # Overlapping detections
        if len(boxes) >= 2:
            max_iou = _max_pairwise_iou(boxes)
            if max_iou > 0.5:
                entry['max_pairwise_iou'] = round(float(max_iou), 3)
                analysis['overlapping_detections'].append(entry.copy())

        # Low-confidence predictions
        if len(scores) > 0:
            low_conf_count = int(np.sum(scores < 0.35))
            if low_conf_count >= 3:
                entry['n_low_conf'] = low_conf_count
                entry['min_conf'] = round(float(scores.min()), 3)
                analysis['low_confidence'].append(entry.copy())

        # Possible false positives candidate
        # Heuristic: if we detect significantly more than GT, some may be FPs
        if gt is not None and count > gt + 3:
            entry['excess_detections'] = count - gt
            entry['note'] = ("Diagnostic candidate: more detections than GT. "
                             "Requires visual review to confirm FPs.")
            analysis['possible_false_positives'].append(entry.copy())

    # Sort each category
    for key in analysis:
        if analysis[key] and 'signed_error' in analysis[key][0]:
            analysis[key].sort(key=lambda x: abs(x.get('signed_error', 0)),
                               reverse=True)

    return analysis


def _max_pairwise_iou(boxes):
    """Compute maximum IoU among all pairs of boxes."""
    n = len(boxes)
    if n < 2:
        return 0.0

    max_iou = 0.0
    for i in range(n):
        for j in range(i + 1, min(n, i + 20)):  # limit computation
            iou = _compute_iou(boxes[i], boxes[j])
            max_iou = max(max_iou, iou)

    return max_iou


def _compute_iou(box1, box2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def run_error_analysis(config=None, custom_model_path=None,
                       split='test', max_vis=20):
    """Run full error analysis pipeline.

    1. Run detector on all images in split
    2. Categorize diagnostic cases
    3. Save visualizations for worst errors
    4. Save error analysis report

    Args:
        config: Config dict.
        custom_model_path: Custom model path or None for pretrained.
        split: Dataset split to analyze.
        max_vis: Max number of visualizations to save per category.
    """
    if config is None:
        config = load_config()

    paths = config['paths']
    output_dir = Path(paths['error_analysis_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    split_csv = Path(paths['existing_splits_dir']) / f"{split}.csv"
    split_df = pd.read_csv(split_csv)

    # Initialize detector
    detector = PersonDetector(config=config,
                              custom_model_path=custom_model_path)

    # Run detection
    print(f"Running detection on {len(split_df)} {split} images...")
    image_paths = split_df['image'].tolist()
    gt_counts = split_df['count'].tolist()
    detections_list = []

    for i, img_path in enumerate(image_paths):
        det = detector.detect(img_path)
        detections_list.append(det)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(image_paths)}")

    # Analyze
    print("\nAnalyzing detections...")
    analysis = analyze_detections(detections_list, image_paths, gt_counts,
                                  config)

    # Report
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS REPORT ({split} set)")
    print(f"{'='*60}")

    for category, items in analysis.items():
        print(f"\n{category}: {len(items)} images")
        if items:
            for item in items[:5]:
                gt_str = f"GT={item['gt_count']}" if item.get('gt_count') is not None else ""
                pred_str = f"Pred={item['pred_count']}"
                extra = ""
                if 'signed_error' in item:
                    extra = f" Error={item['signed_error']:+d}"
                print(f"  {item['image']}: {gt_str} {pred_str}{extra}")

    print(f"\nNOTE: Without bounding-box ground truth, individual missed people,")
    print(f"false positives, and duplicate detections cannot be confirmed from")
    print(f"count differences alone. These are diagnostic candidates for review.")

    # Save analysis JSON
    report_path = output_dir / f"{split}_error_analysis.json"
    # Make serializable (remove numpy types)
    serializable = {}
    for key, items in analysis.items():
        serializable[key] = []
        for item in items:
            clean = {k: (int(v) if isinstance(v, (np.integer, np.int32, np.int64))
                         else float(v) if isinstance(v, (np.floating,))
                         else v)
                     for k, v in item.items()}
            serializable[key].append(clean)

    with open(report_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nAnalysis saved to {report_path}")

    # Save visualizations for worst errors
    print(f"\nSaving error visualizations...")

    # Worst undercounts
    undercount_dir = output_dir / "undercount"
    undercount_dir.mkdir(exist_ok=True)
    for item in analysis['large_undercount'][:max_vis]:
        idx = image_paths.index(item['image_path'])
        draw_detections(
            item['image_path'], detections_list[idx],
            output_path=str(undercount_dir / f"undercount_{item['image']}"),
            gt_count=item['gt_count'],
        )

    # Worst overcounts
    overcount_dir = output_dir / "overcount"
    overcount_dir.mkdir(exist_ok=True)
    for item in analysis['large_overcount'][:max_vis]:
        idx = image_paths.index(item['image_path'])
        draw_detections(
            item['image_path'], detections_list[idx],
            output_path=str(overcount_dir / f"overcount_{item['image']}"),
            gt_count=item['gt_count'],
        )

    # Error grid
    if gt_counts:
        create_error_grid(
            image_paths, detections_list, gt_counts,
            output_path=str(output_dir / f"{split}_error_grid.jpg"),
            title=f"Worst Counting Errors ({split} set)",
        )

    print(f"\nError analysis complete. Results in {output_dir}")

    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run error analysis")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--model", default=None, help="Custom model path")
    args = parser.parse_args()

    run_error_analysis(split=args.split, custom_model_path=args.model)
