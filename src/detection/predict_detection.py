"""
Detection-based inference entry point.

Accepts an image and returns bounding boxes, confidence scores, class IDs,
and a person count. Can use either the pretrained COCO detector or a
custom-trained model.

Pipeline:
    Image → Detector → Confidence Filtering → Person Class Filtering →
    NMS → Final Person Detections → Count

The final count is: count = len(person_detections)

YOLOv8 performs confidence filtering, class filtering, and NMS internally.
This script configures those thresholds; it does NOT apply them twice.
"""

import argparse
import json
from pathlib import Path

from src.detection.config import load_config
from src.detection.detector import PersonDetector
from src.detection.visualize import draw_detections


def predict_detection(image_path, config=None, custom_model_path=None,
                      conf=None, iou=None, output_path=None,
                      visualize=True, gt_count=None):
    """Run person detection on a single image.

    Args:
        image_path: Path to the image file.
        config: Config dict or None.
        custom_model_path: Path to custom model, or None for pretrained.
        conf: Override confidence threshold.
        iou: Override NMS IoU threshold.
        output_path: Save visualization here.
        visualize: Whether to save a visualized image.
        gt_count: Optional ground-truth count for display.

    Returns:
        dict with bounding_boxes, confidence_scores, class_ids, count.
    """
    if config is None:
        config = load_config()

    detector = PersonDetector(config=config,
                              custom_model_path=custom_model_path)

    # Run detection
    result = detector.detect(image_path, conf=conf, iou=iou)

    # Print results
    img_name = Path(image_path).name
    print(f"\nImage: {img_name}")
    print(f"Detected People: {result['count']}")

    if gt_count is not None:
        error = result['count'] - gt_count
        sign = "+" if error > 0 else ""
        print(f"Ground Truth:    {gt_count}")
        print(f"Error:           {sign}{error}")

    if result['count'] > 0:
        print(f"Confidence range: [{result['confidence_scores'].min():.3f}, "
              f"{result['confidence_scores'].max():.3f}]")

    # Visualize
    if visualize:
        if output_path is None:
            vis_dir = Path(config['paths']['visualization_dir'])
            vis_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(vis_dir / f"det_{img_name}")

        draw_detections(image_path, result,
                        output_path=output_path,
                        gt_count=gt_count)
        print(f"Visualization saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Person detection inference",
        epilog="""
Threshold guidance:
  --conf  Low values (0.1-0.2) → more detections but more false positives.
          High values (0.5-0.8) → fewer detections, may miss people.
  --iou   Low values (0.3) → aggressive NMS, removes overlapping real people.
          High values (0.7) → permissive NMS, may retain duplicate detections.
  --max-det  Low limits may cap counts in dense scenes.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image file")
    parser.add_argument("--model", type=str, default=None,
                        help="Custom-trained model path (default: pretrained COCO)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold (default: from config)")
    parser.add_argument("--iou", type=float, default=None,
                        help="NMS IoU threshold (default: from config)")
    parser.add_argument("--max-det", type=int, default=None,
                        help="Max detections per image")
    parser.add_argument("--output", type=str, default=None,
                        help="Output visualization path")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--gt-count", type=int, default=None,
                        help="Ground-truth count for comparison")
    parser.add_argument("--json", action="store_true",
                        help="Print result as JSON")

    args = parser.parse_args()

    result = predict_detection(
        image_path=args.image,
        custom_model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        output_path=args.output,
        visualize=not args.no_vis,
        gt_count=args.gt_count,
    )

    if args.json:
        output = {
            'count': result['count'],
            'bounding_boxes': result['bounding_boxes'].tolist(),
            'confidence_scores': result['confidence_scores'].tolist(),
            'class_ids': result['class_ids'].tolist(),
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
