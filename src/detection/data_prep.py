"""
Data preparation and validation for person detection.

Handles:
- Dataset inventory and annotation inspection
- YOLO label validation (format, coordinates, bounds)
- Annotation status tracking (annotated / empty / unannotated)
- Detection-compatible train/val/test splitting with temporal grouping
- Dataset YAML generation for YOLOv8 training
- Ground-truth visualization sampling
"""

import json
import re
import shutil
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Dataset Inventory
# ---------------------------------------------------------------------------

def inventory_dataset(frames_dir, labels_dir=None, count_csv=None):
    """Scan the dataset and report what exists.

    Returns:
        dict with keys:
            total_images, image_files, images_with_labels, images_without_labels,
            images_with_counts, count_stats, label_stats, warnings
    """
    frames_dir = Path(frames_dir)
    labels_dir = Path(labels_dir) if labels_dir else None
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted(
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in image_exts
    )
    
    result = {
        'total_images': len(image_files),
        'image_files': [str(f) for f in image_files],
        'images_with_labels': [],
        'images_without_labels': [],
        'label_files': [],
        'warnings': [],
    }
    
    # Check for YOLO label files
    if labels_dir and labels_dir.exists():
        for img_path in image_files:
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                result['images_with_labels'].append(str(img_path))
                result['label_files'].append(str(label_path))
            else:
                result['images_without_labels'].append(str(img_path))
    else:
        result['images_without_labels'] = [str(f) for f in image_files]
        result['warnings'].append(
            f"Label directory '{labels_dir}' does not exist. "
            "No bounding-box annotations found."
        )
    
    # Load count labels if available
    if count_csv and Path(count_csv).exists():
        df = pd.read_csv(count_csv)
        result['count_stats'] = {
            'total_labeled': len(df),
            'min': int(df['count'].min()),
            'max': int(df['count'].max()),
            'mean': float(df['count'].mean()),
            'median': float(df['count'].median()),
            'std': float(df['count'].std()),
        }
    
    # Summary
    n_labeled = len(result['images_with_labels'])
    n_unlabeled = len(result['images_without_labels'])
    result['summary'] = (
        f"Dataset: {result['total_images']} images, "
        f"{n_labeled} with bbox labels, "
        f"{n_unlabeled} without bbox labels."
    )
    
    return result


# ---------------------------------------------------------------------------
# 2. Annotation Status Tracking
# ---------------------------------------------------------------------------

def load_annotation_status(status_file):
    """Load annotation status JSON.

    Format: {filename_stem: "annotated"|"empty"|"unannotated"|"skipped"}
    - "annotated": label file exists and contains >= 1 box
    - "empty": label file exists but is empty (reviewed, zero people)
    - "unannotated": no label file, not yet reviewed
    - "skipped": deliberately excluded from annotation
    """
    status_file = Path(status_file)
    if status_file.exists():
        with open(status_file) as f:
            return json.load(f)
    return {}


def save_annotation_status(status_file, status_dict):
    """Save annotation status JSON."""
    status_file = Path(status_file)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, 'w') as f:
        json.dump(status_dict, f, indent=2)


def update_annotation_status(frames_dir, labels_dir, status_file):
    """Scan labels directory and update annotation status.

    Distinguishes:
    - Missing label file → "unannotated"
    - Empty label file → "empty" (reviewed, zero people in image)
    - Non-empty label file → "annotated"
    """
    frames_dir = Path(frames_dir)
    labels_dir = Path(labels_dir)
    
    status = load_annotation_status(status_file)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for img in frames_dir.iterdir():
        if img.suffix.lower() not in image_exts:
            continue
        stem = img.stem
        label_path = labels_dir / (stem + '.txt')
        
        if stem in status and status[stem] == 'skipped':
            continue  # preserve manual skip decisions
        
        if label_path.exists():
            content = label_path.read_text().strip()
            if content:
                status[stem] = 'annotated'
            else:
                status[stem] = 'empty'
        else:
            if stem not in status:
                status[stem] = 'unannotated'
    
    save_annotation_status(status_file, status)
    
    counts = Counter(status.values())
    print(f"Annotation status updated: {dict(counts)}")
    return status


# ---------------------------------------------------------------------------
# 3. YOLO Label Validation
# ---------------------------------------------------------------------------

def validate_yolo_label(label_path, image_width=None, image_height=None,
                        valid_class_ids=None):
    """Validate a single YOLO format label file.

    Checks:
    - Line format: class_id x_center y_center width height
    - class_id is integer and in valid_class_ids (if provided)
    - Coordinates are finite floats in [0, 1]
    - Width and height are positive
    - Box does not extend outside [0, 1] bounds (warning, not error)

    Returns:
        dict with: valid (bool), boxes (list), errors (list), warnings (list)
    """
    label_path = Path(label_path)
    result = {'valid': True, 'boxes': [], 'errors': [], 'warnings': []}
    
    if not label_path.exists():
        result['valid'] = False
        result['errors'].append(f"Label file not found: {label_path}")
        return result
    
    content = label_path.read_text().strip()
    if not content:
        # Empty file = reviewed image with zero people (valid)
        return result
    
    for i, line in enumerate(content.split('\n'), 1):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) != 5:
            result['errors'].append(
                f"Line {i}: expected 5 values, got {len(parts)}: '{line}'"
            )
            result['valid'] = False
            continue
        
        try:
            cls_id = int(parts[0])
            x_c, y_c, w, h = [float(v) for v in parts[1:]]
        except ValueError:
            result['errors'].append(
                f"Line {i}: cannot parse values: '{line}'"
            )
            result['valid'] = False
            continue
        
        # Check finite
        if not all(np.isfinite([x_c, y_c, w, h])):
            result['errors'].append(f"Line {i}: non-finite coordinates")
            result['valid'] = False
            continue
        
        # Check class ID
        if valid_class_ids is not None and cls_id not in valid_class_ids:
            result['errors'].append(
                f"Line {i}: class_id {cls_id} not in {valid_class_ids}"
            )
            result['valid'] = False
        
        # Check coordinate ranges
        for name, val in [('x_center', x_c), ('y_center', y_c),
                          ('width', w), ('height', h)]:
            if val < 0 or val > 1:
                result['warnings'].append(
                    f"Line {i}: {name}={val:.4f} outside [0,1]"
                )
        
        # Check positive dimensions
        if w <= 0 or h <= 0:
            result['errors'].append(
                f"Line {i}: non-positive dimensions w={w:.4f}, h={h:.4f}"
            )
            result['valid'] = False
            continue
        
        # Check box bounds
        x_min = x_c - w / 2
        x_max = x_c + w / 2
        y_min = y_c - h / 2
        y_max = y_c + h / 2
        
        if x_min < -0.01 or x_max > 1.01 or y_min < -0.01 or y_max > 1.01:
            result['warnings'].append(
                f"Line {i}: box extends outside image bounds"
            )
        
        result['boxes'].append({
            'class_id': cls_id,
            'x_center': x_c, 'y_center': y_c,
            'width': w, 'height': h,
            'line': i,
        })
    
    return result


def validate_all_labels(frames_dir, labels_dir, valid_class_ids=None,
                        report_path=None):
    """Validate all label files and generate a report.

    Returns:
        dict with: total, valid, invalid, empty, missing,
                   errors (list), warnings (list), box_count_stats
    """
    frames_dir = Path(frames_dir)
    labels_dir = Path(labels_dir)
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted(
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in image_exts
    )
    
    report = {
        'total_images': len(images),
        'valid': 0, 'invalid': 0, 'empty': 0, 'missing': 0,
        'total_boxes': 0,
        'errors': [], 'warnings': [],
        'per_image': [],
    }
    
    box_counts = []
    
    for img_path in images:
        label_path = labels_dir / (img_path.stem + '.txt')
        
        if not label_path.exists():
            report['missing'] += 1
            report['per_image'].append({
                'image': img_path.name,
                'status': 'missing',
                'boxes': 0,
            })
            continue
        
        val = validate_yolo_label(
            label_path,
            valid_class_ids=valid_class_ids,
        )
        
        n_boxes = len(val['boxes'])
        box_counts.append(n_boxes)
        report['total_boxes'] += n_boxes
        
        if not val['valid']:
            report['invalid'] += 1
            for err in val['errors']:
                report['errors'].append(f"{img_path.name}: {err}")
        elif n_boxes == 0:
            report['empty'] += 1
            report['valid'] += 1
        else:
            report['valid'] += 1
        
        for warn in val['warnings']:
            report['warnings'].append(f"{img_path.name}: {warn}")
        
        report['per_image'].append({
            'image': img_path.name,
            'status': 'valid' if val['valid'] else 'invalid',
            'boxes': n_boxes,
        })
    
    if box_counts:
        report['box_count_stats'] = {
            'min': int(np.min(box_counts)),
            'max': int(np.max(box_counts)),
            'mean': float(np.mean(box_counts)),
            'median': float(np.median(box_counts)),
        }
    
    # Save report
    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # Save summary (exclude per_image for readability)
        summary = {k: v for k, v in report.items() if k != 'per_image'}
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Validation report saved to {report_path}")
    
    return report


# ---------------------------------------------------------------------------
# 4. Reject unsupported count-to-box conversion
# ---------------------------------------------------------------------------

def reject_count_to_box_fabrication():
    """Explicitly reject fabricating bounding boxes from count labels.

    This function exists to document and enforce the constraint that
    total count labels cannot be meaningfully converted to bounding boxes.
    Calling any conversion function will raise an error.
    """
    raise ValueError(
        "Cannot convert total count labels to bounding boxes. "
        "Count labels indicate how many people are in an image, but do not "
        "provide spatial location information. Bounding-box annotations must "
        "be created through manual annotation or semi-automatic annotation "
        "(pretrained detector proposals + human review). "
        "See the annotation workflow in the project documentation."
    )


# ---------------------------------------------------------------------------
# 5. Duplicate Detection
# ---------------------------------------------------------------------------

def find_duplicate_images(frames_dir, method='hash'):
    """Find exact duplicate images by content hash.

    Args:
        frames_dir: Directory containing images.
        method: 'hash' for MD5 content hash.

    Returns:
        List of duplicate groups (each group is a list of filenames).
    """
    frames_dir = Path(frames_dir)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    hash_to_files = defaultdict(list)
    
    for img_path in sorted(frames_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        with open(img_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        hash_to_files[file_hash].append(img_path.name)
    
    duplicates = [
        files for files in hash_to_files.values()
        if len(files) > 1
    ]
    
    if duplicates:
        print(f"Found {len(duplicates)} duplicate groups:")
        for group in duplicates:
            print(f"  {group}")
    else:
        print("No exact duplicate images found.")
    
    return duplicates


# ---------------------------------------------------------------------------
# 6. Leakage-Aware Splitting
# ---------------------------------------------------------------------------

def extract_sequence_id(filename):
    """Extract numeric sequence ID from filename like 'seq_001234.jpg'."""
    match = re.search(r'\d+', Path(filename).stem)
    return int(match.group()) if match else None


def group_by_temporal_proximity(image_files, gap_threshold=30):
    """Group sequential frames into scenes/segments.

    Consecutive frames within `gap_threshold` of each other are grouped together.
    This prevents temporal leakage: adjacent frames from the same camera view
    are visually near-identical and must not be split across train/val/test.

    Args:
        image_files: List of image paths or filenames.
        gap_threshold: Maximum ID gap to consider frames as same group.

    Returns:
        List of groups, where each group is a list of filenames.
    """
    items = []
    for f in image_files:
        seq_id = extract_sequence_id(f)
        if seq_id is not None:
            items.append((seq_id, str(f)))
    
    items.sort(key=lambda x: x[0])
    
    groups = []
    current_group = [items[0]]
    
    for i in range(1, len(items)):
        if items[i][0] - items[i - 1][0] <= gap_threshold:
            current_group.append(items[i])
        else:
            groups.append(current_group)
            current_group = [items[i]]
    
    if current_group:
        groups.append(current_group)
    
    return groups


def create_detection_splits(frames_dir, labels_dir, output_dir,
                            train_ratio=0.7, val_ratio=0.15,
                            gap_threshold=30, seed=42,
                            require_labels=True):
    """Create train/val/test splits with temporal grouping.

    Groups consecutive frames into segments to prevent temporal leakage,
    then splits at the group level.

    Args:
        frames_dir: Directory of images.
        labels_dir: Directory of YOLO label files.
        output_dir: Where to write split files (train.txt, val.txt, test.txt).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation. Test = 1 - train - val.
        gap_threshold: Max sequence gap for grouping.
        seed: Random seed.
        require_labels: If True, only include images with label files.

    Returns:
        dict with train/val/test file lists.
    """
    frames_dir = Path(frames_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = sorted(
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in image_exts
    )
    
    # Filter to only annotated images if required
    if require_labels:
        annotated = []
        for img in all_images:
            label_path = labels_dir / (img.stem + '.txt')
            if label_path.exists():
                annotated.append(img)
        
        if not annotated:
            raise FileNotFoundError(
                f"No label files found in {labels_dir}. "
                "Cannot create detection splits without bounding-box annotations. "
                "Please complete the annotation workflow first."
            )
        
        images = annotated
        print(f"Found {len(images)} annotated images out of {len(all_images)} total.")
    else:
        images = all_images
    
    # Group by temporal proximity
    groups = group_by_temporal_proximity(
        [f.name for f in images], gap_threshold=gap_threshold
    )
    print(f"Formed {len(groups)} temporal groups from {len(images)} images.")
    
    # Shuffle and split groups
    rng = np.random.RandomState(seed)
    group_indices = list(range(len(groups)))
    rng.shuffle(group_indices)
    
    n_train = int(len(groups) * train_ratio)
    n_val = int(len(groups) * val_ratio)
    
    train_groups = [groups[i] for i in group_indices[:n_train]]
    val_groups = [groups[i] for i in group_indices[n_train:n_train + n_val]]
    test_groups = [groups[i] for i in group_indices[n_train + n_val:]]
    
    def flatten_groups(grps):
        return [item[1] for g in grps for item in g]
    
    splits = {
        'train': sorted(flatten_groups(train_groups)),
        'val': sorted(flatten_groups(val_groups)),
        'test': sorted(flatten_groups(test_groups)),
    }
    
    # Save split files
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, files in splits.items():
        split_file = output_dir / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for filename in files:
                # Write absolute path for YOLO compatibility
                img_path = frames_dir / filename
                f.write(str(img_path.resolve()) + '\n')
        print(f"{split_name}: {len(files)} images")
    
    # Verify no leakage
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    test_set = set(splits['test'])
    
    assert not (train_set & val_set), "Train-val overlap detected!"
    assert not (train_set & test_set), "Train-test overlap detected!"
    assert not (val_set & test_set), "Val-test overlap detected!"
    
    # Check temporal leakage
    train_ids = {extract_sequence_id(f) for f in splits['train']}
    val_ids = {extract_sequence_id(f) for f in splits['val']}
    test_ids = {extract_sequence_id(f) for f in splits['test']}
    
    val_adjacent = sum(
        1 for v in val_ids
        if (v - 1 in train_ids) or (v + 1 in train_ids)
    )
    test_adjacent = sum(
        1 for t in test_ids
        if (t - 1 in train_ids) or (t + 1 in train_ids)
    )
    
    print(f"\nTemporal leakage check:")
    print(f"  Val frames with adjacent-frame neighbor in train: "
          f"{val_adjacent}/{len(val_ids)}")
    print(f"  Test frames with adjacent-frame neighbor in train: "
          f"{test_adjacent}/{len(test_ids)}")
    
    return splits


# ---------------------------------------------------------------------------
# 7. Dataset YAML Generation
# ---------------------------------------------------------------------------

def generate_dataset_yaml(frames_dir, labels_dir, splits_dir, output_path,
                          class_names=None):
    """Generate a YOLO dataset.yaml for training.

    Args:
        frames_dir: Directory containing images.
        labels_dir: Directory containing YOLO label .txt files.
        splits_dir: Directory containing train.txt, val.txt, test.txt.
        output_path: Where to write dataset.yaml.
        class_names: Dict of {class_id: name}. Default: {0: 'person'}.
    """
    if class_names is None:
        class_names = {0: 'person'}
    
    frames_dir = Path(frames_dir).resolve()
    labels_dir = Path(labels_dir).resolve()
    splits_dir = Path(splits_dir).resolve()
    output_path = Path(output_path)
    
    # Build YAML content
    lines = [
        f"# Auto-generated dataset config for person detection",
        f"# Generated for YOLOv8 training",
        f"",
        f"path: {frames_dir.parent}  # dataset root dir",
        f"train: {splits_dir / 'train.txt'}",
        f"val: {splits_dir / 'val.txt'}",
        f"test: {splits_dir / 'test.txt'}",
        f"",
        f"# Classes",
        f"nc: {len(class_names)}",
        f"names:",
    ]
    for cls_id in sorted(class_names.keys()):
        lines.append(f"  {cls_id}: {class_names[cls_id]}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f"Dataset YAML saved to {output_path}")


# ---------------------------------------------------------------------------
# 8. Count Label Comparison Utility
# ---------------------------------------------------------------------------

def load_count_labels(count_csv):
    """Load image-level count labels from CSV.

    Returns:
        dict mapping image_stem (str) → count (int)
    """
    df = pd.read_csv(count_csv)
    label_map = {}
    for _, row in df.iterrows():
        # Handle both 'id' column (numeric) and 'image' column (path)
        if 'id' in df.columns:
            key = f"seq_{int(row['id']):06d}"
        elif 'image' in df.columns:
            key = Path(row['image']).stem
        label_map[key] = int(row['count'])
    return label_map


# ---------------------------------------------------------------------------
# Entry point for data preparation
# ---------------------------------------------------------------------------

def prepare_data(config=None):
    """Run full data preparation pipeline.

    Steps:
    1. Inventory dataset
    2. Check for duplicates
    3. Validate any existing labels
    4. Update annotation status
    5. Create detection splits (if labels exist)
    6. Generate dataset YAML (if labels exist)
    """
    if config is None:
        from src.detection.config import load_config
        config = load_config()
    
    paths = config['paths']
    frames_dir = paths['frames_dir']
    labels_dir = paths['bbox_labels_dir']
    count_csv = paths['count_labels']
    status_file = paths['annotation_status_file']
    splits_dir = paths['detection_splits_dir']
    dataset_yaml = paths['detection_dataset_yaml']
    
    print("=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)
    
    # 1. Inventory
    print("\n--- Dataset Inventory ---")
    inv = inventory_dataset(frames_dir, labels_dir, count_csv)
    print(inv['summary'])
    if inv.get('count_stats'):
        cs = inv['count_stats']
        print(f"Count labels: {cs['total_labeled']} images, "
              f"range [{cs['min']}, {cs['max']}], "
              f"mean={cs['mean']:.1f}, std={cs['std']:.1f}")
    for w in inv['warnings']:
        print(f"  WARNING: {w}")
    
    # 2. Duplicates
    print("\n--- Duplicate Check ---")
    dupes = find_duplicate_images(frames_dir)
    
    # 3. Validate labels (if any exist)
    if inv['images_with_labels']:
        print("\n--- Label Validation ---")
        report = validate_all_labels(
            frames_dir, labels_dir,
            valid_class_ids={0},
            report_path=str(Path(paths['detection_outputs_dir']) / 'label_validation.json'),
        )
        print(f"Valid: {report['valid']}, Invalid: {report['invalid']}, "
              f"Empty: {report['empty']}, Missing: {report['missing']}")
        if report['errors']:
            print(f"Errors ({len(report['errors'])}):")
            for err in report['errors'][:10]:
                print(f"  {err}")
    
    # 4. Annotation status
    print("\n--- Annotation Status ---")
    labels_path = Path(labels_dir)
    if labels_path.exists():
        update_annotation_status(frames_dir, labels_dir, status_file)
    else:
        print(f"Labels directory {labels_dir} does not exist yet.")
        print("Create it and add YOLO format .txt files after annotation.")
    
    # 5. Create splits (only if labels exist)
    n_labeled = len(inv['images_with_labels'])
    if n_labeled > 0:
        print(f"\n--- Creating Detection Splits ({n_labeled} labeled images) ---")
        create_detection_splits(
            frames_dir, labels_dir, splits_dir,
            seed=config['training']['seed'],
        )
        
        # 6. Generate dataset YAML
        print("\n--- Generating Dataset YAML ---")
        generate_dataset_yaml(
            frames_dir, labels_dir, splits_dir, dataset_yaml
        )
    else:
        print("\n--- Skipping Split Creation ---")
        print("No bounding-box annotations found. Splits will be created")
        print("after the annotation workflow is completed.")
        print(f"Expected label location: {labels_dir}/")
        print("Format: one .txt file per image, YOLO normalized format:")
        print("  class_id x_center y_center width height")
        print("  (class 0 = person)")
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    
    return inv


if __name__ == "__main__":
    prepare_data()
