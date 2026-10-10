"""
Custom detector training entry point.

Trains a YOLOv8 person detector on the custom dataset, using:
- Pretrained COCO weights for initialization
- Validated YOLO-format bounding-box labels
- Configuration from configs/detection_config.yaml
- Temporal-aware train/val/test splits

Prerequisites:
    - Bounding-box annotations must exist in data/raw/labels/
    - Detection splits must be created (run data_prep.py first)
    - Dataset YAML must be generated (run data_prep.py first)

This script will FAIL CLEARLY if annotations are missing or incomplete.
It never silently substitutes empty labels or pretrained predictions.
"""

import sys
from pathlib import Path

from ultralytics import YOLO

from src.detection.config import load_config
from src.detection.data_prep import (
    validate_all_labels,
    inventory_dataset,
    update_annotation_status,
)


def validate_prerequisites(config):
    """Check that all prerequisites for training are met.

    Fails with clear error messages if:
    - No label files exist
    - Dataset YAML is missing
    - Labels are invalid
    """
    paths = config['paths']
    labels_dir = Path(paths['bbox_labels_dir'])
    dataset_yaml = Path(paths['detection_dataset_yaml'])
    frames_dir = Path(paths['frames_dir'])

    errors = []

    # Check labels directory
    if not labels_dir.exists():
        errors.append(
            f"Labels directory does not exist: {labels_dir}\n"
            f"  You must complete the annotation workflow before training.\n"
            f"  Each image needs a corresponding .txt file in YOLO format:\n"
            f"    class_id x_center y_center width height\n"
            f"  Run: python -m src.detection.data_prep  (after annotation)"
        )
    else:
        label_files = list(labels_dir.glob('*.txt'))
        if not label_files:
            errors.append(
                f"No .txt label files found in {labels_dir}\n"
                f"  Cannot train a detector without bounding-box annotations.\n"
                f"  Complete the annotation workflow first."
            )
        else:
            # Validate labels
            report = validate_all_labels(
                frames_dir, labels_dir,
                valid_class_ids={config['model']['custom_person_class_id']},
            )
            if report['invalid'] > 0:
                errors.append(
                    f"{report['invalid']} label files have validation errors.\n"
                    f"  Fix errors before training. Details:\n"
                    + "\n".join(f"    {e}" for e in report['errors'][:10])
                )

            # Check minimum annotation count
            n_annotated = report['valid']
            if n_annotated < 50:
                errors.append(
                    f"Only {n_annotated} valid label files found.\n"
                    f"  Minimum recommended: 50 images (ideally 200+).\n"
                    f"  Continue annotating before training."
                )

    # Check dataset YAML
    if not dataset_yaml.exists():
        errors.append(
            f"Dataset YAML not found: {dataset_yaml}\n"
            f"  Run: python -m src.detection.data_prep  (after annotation)"
        )

    if errors:
        print("=" * 60)
        print("TRAINING PREREQUISITES NOT MET")
        print("=" * 60)
        for i, err in enumerate(errors, 1):
            print(f"\n[{i}] {err}")
        print("\n" + "=" * 60)
        print("Training cannot proceed. Complete the steps above first.")
        print("=" * 60)
        return False

    return True


def train_detector(config=None):
    """Train a custom YOLOv8 person detector.

    Uses pretrained COCO weights as initialization and fine-tunes on
    the annotated custom dataset.

    All hyperparameters are loaded from the config file.
    """
    if config is None:
        config = load_config()

    # Validate prerequisites
    if not validate_prerequisites(config):
        sys.exit(1)

    paths = config['paths']
    training = config['training']

    # Load pretrained model
    model_checkpoint = config['model']['checkpoint']
    print(f"\nInitializing from pretrained: {model_checkpoint}")
    model = YOLO(model_checkpoint)

    # Training output directory
    output_dir = Path(paths['training_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset YAML path
    dataset_yaml = str(Path(paths['detection_dataset_yaml']).resolve())

    # Augmentation settings
    aug = config.get('augmentation', {})

    print(f"\nStarting training:")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Epochs: {training['epochs']}")
    print(f"  Batch size: {training['batch_size']}")
    print(f"  Image size: {training['image_size']}")
    print(f"  Learning rate: {training['learning_rate']}")
    print(f"  Patience: {training['patience']}")
    print(f"  Device: cuda (auto)")
    print(f"  Output: {output_dir}")

    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=training['epochs'],
        batch=training['batch_size'],
        imgsz=training['image_size'],
        lr0=training['learning_rate'],
        optimizer=training['optimizer'],
        patience=training['patience'],
        seed=training['seed'],
        workers=training['workers'],
        resume=training['resume'],
        freeze=training['freeze'],
        close_mosaic=training['close_mosaic'],

        # Augmentation
        hsv_h=aug.get('hsv_h', 0.015),
        hsv_s=aug.get('hsv_s', 0.5),
        hsv_v=aug.get('hsv_v', 0.3),
        degrees=aug.get('degrees', 5.0),
        translate=aug.get('translate', 0.05),
        scale=aug.get('scale', 0.3),
        shear=aug.get('shear', 2.0),
        flipud=aug.get('flipud', 0.0),
        fliplr=aug.get('fliplr', 0.5),
        mosaic=aug.get('mosaic', 0.5),
        mixup=aug.get('mixup', 0.0),
        copy_paste=aug.get('copy_paste', 0.0),

        # Output
        project=str(output_dir),
        name="person_detector",
        exist_ok=True,
        save=True,
        save_period=-1,  # save only best
        plots=True,
        verbose=True,
    )

    # Report results
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")

    best_model = output_dir / "person_detector" / "weights" / "best.pt"
    last_model = output_dir / "person_detector" / "weights" / "last.pt"

    if best_model.exists():
        print(f"Best model: {best_model}")
    if last_model.exists():
        print(f"Last model: {last_model}")

    print(f"\nTo evaluate: python -m src.detection.evaluate --model {best_model}")
    print(f"To predict:  python -m src.detection.predict_detection --image <path> --model {best_model}")

    return results


if __name__ == "__main__":
    train_detector()
