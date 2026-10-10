"""
Detection visualization module.

Draws bounding boxes, confidence scores, person indices, and count comparisons
on images. Supports ground-truth overlays and error analysis grids.

Color conventions (BGR):
    - Predictions: Blue (255, 0, 0)
    - Ground truth: Green (0, 255, 0)
    - Text: White on semi-transparent dark background
"""

import cv2
import numpy as np
import math
from pathlib import Path


# BGR colors
PRED_COLOR = (255, 0, 0)     # Blue
GT_COLOR = (0, 255, 0)       # Green
TEXT_COLOR = (255, 255, 255)  # White
BG_COLOR = (0, 0, 0)         # Black


def _load_image(image_input):
    """Load image as BGR numpy array."""
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Could not read image at {image_input}")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    else:
        raise TypeError("image_input must be a file path or numpy array (BGR)")


def _draw_text_bg(img, text, position, font_scale=0.5, thickness=1,
                  text_color=TEXT_COLOR, bg_color=BG_COLOR, alpha=0.6):
    """Draw text with semi-transparent background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    pad = 4

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Text
    cv2.putText(img, text, (x, y), font, font_scale,
                text_color, thickness, cv2.LINE_AA)
    return img


def draw_detections(image, detections, output_path=None,
                    show_indices=True, gt_count=None):
    """Draw detection bounding boxes on an image.

    Args:
        image: File path or BGR numpy array.
        detections: Dict from PersonDetector.detect() with keys:
            'bounding_boxes': (N,4) array [x1,y1,x2,y2]
            'confidence_scores': (N,) array
            'count': int
        output_path: Save annotated image here (optional).
        show_indices: Show person index numbers on boxes.
        gt_count: Ground-truth count for comparison overlay.

    Returns:
        Annotated BGR numpy array.
    """
    img = _load_image(image)

    boxes = detections.get('bounding_boxes', np.empty((0, 4)))
    scores = detections.get('confidence_scores', np.empty(0))
    count = detections.get('count', len(boxes))

    # Draw each detection box
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)

        label = f"{scores[i]:.2f}"
        if show_indices:
            label = f"#{i+1} {label}"

        label_y = max(20, y1 - 5)
        _draw_text_bg(img, label, (x1, label_y), font_scale=0.4, thickness=1)

    # Count overlay at top
    count_text = f"Detected: {count}"
    if gt_count is not None:
        error = count - gt_count
        sign = "+" if error > 0 else ""
        count_text += f"  |  GT: {gt_count}  |  Error: {sign}{error}"

    _draw_text_bg(img, count_text, (10, 30),
                  font_scale=0.7, thickness=2)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)

    return img


def draw_gt_comparison(image, detections, gt_boxes=None, gt_count=None,
                       output_path=None):
    """Draw predictions (blue) and ground-truth boxes (green) together.

    Args:
        image: File path or BGR numpy array.
        detections: Dict from PersonDetector.detect().
        gt_boxes: Optional (M,4) array of GT boxes [x1,y1,x2,y2].
        gt_count: Ground-truth count for text overlay.
        output_path: Save here (optional).

    Returns:
        Annotated BGR numpy array.
    """
    img = _load_image(image)

    # Draw GT boxes first (underneath)
    if gt_boxes is not None:
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
            _draw_text_bg(img, "GT", (x1, max(20, y1 - 5)),
                          font_scale=0.35, thickness=1,
                          text_color=GT_COLOR)

    # Draw predictions on top
    img = draw_detections(img, detections, output_path=None,
                          show_indices=False, gt_count=gt_count)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)

    return img


def visualize_batch(image_paths, detector, output_dir,
                    gt_counts=None, max_images=20):
    """Run detection and save visualizations for a batch of images.

    Args:
        image_paths: List of image file paths.
        detector: PersonDetector instance.
        output_dir: Directory to save annotated images.
        gt_counts: Optional list/dict of ground-truth counts.
            If dict, keys should be image stems.
        max_images: Maximum images to process.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(image_paths), max_images)
    for i in range(n):
        img_path = Path(image_paths[i])

        det = detector.detect(str(img_path))

        # Resolve GT count
        gt_c = None
        if isinstance(gt_counts, dict):
            gt_c = gt_counts.get(img_path.stem)
        elif gt_counts is not None and i < len(gt_counts):
            gt_c = gt_counts[i]

        out_path = output_dir / f"det_{img_path.name}"
        draw_detections(str(img_path), det, output_path=str(out_path),
                        gt_count=gt_c)

    print(f"Saved {n} visualizations to {output_dir}")


def create_error_grid(image_paths, detections_list, gt_counts,
                      output_path, title="Error Analysis", n_cols=4,
                      max_cells=16):
    """Create a grid image showing the worst counting errors.

    Args:
        image_paths: List of image paths.
        detections_list: List of detection dicts (aligned with image_paths).
        gt_counts: List of ground-truth counts (aligned with image_paths).
        output_path: Where to save the grid image.
        title: Title text.
        n_cols: Number of columns in the grid.
        max_cells: Maximum number of images in the grid.
    """
    # Compute errors
    errors = []
    for i, (det, gt) in enumerate(zip(detections_list, gt_counts)):
        pred_count = det.get('count', 0)
        abs_err = abs(pred_count - gt)
        errors.append((abs_err, i, pred_count, gt))

    errors.sort(reverse=True, key=lambda x: x[0])
    top = errors[:max_cells]

    if not top:
        print("No errors to display.")
        return

    # Read first image for dimensions
    sample = _load_image(image_paths[top[0][1]])
    cell_h, cell_w = sample.shape[:2]

    n_images = len(top)
    n_rows = math.ceil(n_images / n_cols)
    title_h = 50

    grid = np.zeros((title_h + n_rows * cell_h, n_cols * cell_w, 3),
                    dtype=np.uint8)

    # Title
    _draw_text_bg(grid, title, (20, 35), font_scale=1.0, thickness=2)

    for idx, (abs_err, orig_i, pred_c, gt_c) in enumerate(top):
        row, col = divmod(idx, n_cols)
        y0 = title_h + row * cell_h
        x0 = col * cell_w

        img = _load_image(image_paths[orig_i])
        det = detections_list[orig_i]
        annotated = draw_detections(img, det, show_indices=False,
                                    gt_count=gt_c)

        # Handle size mismatches
        if annotated.shape[:2] != (cell_h, cell_w):
            annotated = cv2.resize(annotated, (cell_w, cell_h))

        grid[y0:y0 + cell_h, x0:x0 + cell_w] = annotated

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    print(f"Error grid saved to {output_path}")
