# Crowd-Counting: Person Detection & Direct Count Regression

A comprehensive PyTorch-based crowd-counting framework for surveillance footage. This project supports two distinct counting paradigms:

1. **Direct Count Regression (ResNet-50)**: Directly maps an image to a continuous scalar count.
2. **Person Detection & Box-Based Counting (YOLOv8)**: Detects individual person bounding boxes and computes the total count as $N = \text{len}(\text{detections})$.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Environment Setup & Installation](#environment-setup--installation)
- [Dataset Specifications](#dataset-specifications)
- [Quick Start](#quick-start)
- [Pipeline 1: Direct Count Regression (ResNet-50)](#pipeline-1-direct-count-regression-resnet-50)
- [Pipeline 2: Person Detection & Counting (YOLOv8)](#pipeline-2-person-detection--counting-yolov8)
- [Why Pretrained Detectors Undercount & Threshold Guidance](#why-pretrained-detectors-undercount--threshold-guidance)
- [Bounding-Box Annotation Workflow](#bounding-box-annotation-workflow)
- [Configuration Reference](#configuration-reference)
- [Evaluation Metrics & Error Analysis](#evaluation-metrics--error-analysis)
- [Architectural Comparison: Regression vs. Detection vs. Density Maps](#architectural-comparison-regression-vs-detection-vs-density-maps)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview

Crowd counting is critical for safety monitoring, facility management, and surveillance. Traditional direct regression models predict an aggregate count from deep feature maps but lack spatial interpretability. Object detection pipelines localize every individual with a bounding box, providing visual transparency, localization, and verifiable counts.

This repository integrates both approaches cleanly without breaking legacy models or workflows:

```
                          ┌─── [Pipeline 1: Direct Regression] ───► ResNet-50 ──► Linear(2048, 1) ──► Continuous Count
                          │
Input Frame (640 × 480) ──┤
                          │
                          └─── [Pipeline 2: Detection Pipeline]  ──► YOLOv8 ──► Filter Person (Class 0) ──► NMS ──► Bounding Boxes ──► Count = len(boxes)
```

---

## Repository Structure

```
Crowd-Counting/
├── configs/
│   └── detection_config.yaml         # Central YAML configuration for detection & training
├── data/
│   ├── raw/
│   │   ├── count/
│   │   │   └── labels.csv             # Ground-truth count annotations (id, count)
│   │   ├── frames/                    # 2,000 raw video frames (seq_000001.jpg - seq_002000.jpg)
│   │   └── labels/                    # [After annotation] YOLO normalized .txt bbox labels
│   ├── splits/                        # Regression splits: train.csv, val.csv, test.csv
│   └── detection_splits/              # Temporal-aware detection splits (train.txt, val.txt, test.txt)
├── models/
│   └── checkpoints/
│       └── best_model.pth             # Trained ResNet-50 regression checkpoint
├── outputs/
│   ├── metrics/
│   │   └── test_predictions.csv      # Regression test predictions & errors
│   └── detection/                     # Detection outputs
│       ├── visualizations/            # Annotated images with bounding boxes & count overlays
│       ├── error_analysis/            # Diagnostic reports, worst-case grids, under/overcount images
│       └── training/                  # Custom detector training runs & weights
├── src/
│   ├── data/                          # Regression data handling
│   │   ├── dataset.py                 # PeopleCountingDataset (PyTorch Dataset)
│   │   ├── preprocessing.py           # Augmentations & transforms (384x384)
│   │   └── split.py                   # Train/Val/Test splitting logic
│   ├── detection/                     # Detection & Box-Counting pipeline
│   │   ├── __init__.py
│   │   ├── config.py                  # YAML configuration parser & fallback defaults
│   │   ├── detector.py                # PersonDetector (YOLOv8 wrapper, class filtering)
│   │   ├── data_prep.py               # Dataset inventory, YOLO validation, temporal splitting
│   │   ├── evaluate.py                # Counting metrics (MAE, RMSE, bias) & comparison tables
│   │   ├── error_analysis.py          # Diagnostic grouping (dense scenes, small boxes, overlaps)
│   │   ├── predict_detection.py       # Single-image detection inference CLI
│   │   ├── train_detector.py          # Custom YOLOv8 training entry point with validation checks
│   │   └── visualize.py               # OpenCV visualization (boxes, confidence, indices, error grids)
│   ├── evaluation/                    # Regression evaluation
│   │   ├── evaluate.py                # Model evaluation over test set
│   │   └── metrics.py                 # MAE and RMSE calculation
│   ├── inference/                     # Regression inference
│   │   └── predict.py                 # Single-image regression script
│   ├── models/                        # Regression neural network architectures
│   │   └── cnn.py                     # ResNet-50 PeopleCountingModel
│   └── training/                      # Regression training loop
│       └── train.py                   # PyTorch training loop with SmoothL1Loss
├── main.py                            # Unified CLI entry point for all workflows
├── run_baseline.py                    # Automated runner for pretrained detection baseline
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## Environment Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- NVIDIA GPU (4 GB+ VRAM recommended, e.g., RTX 3050 Laptop / Desktop) or CPU
- CUDA 12.x compatible drivers

### 2. Clone and Setup Environment

```bash
git clone https://github.com/Ali-Ehab-pixel/Crowd-Counting.git
cd Crowd-Counting
```

Create and activate a virtual environment:

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install PyTorch with CUDA support matching your system, then install requirements:

```bash
pip install -r requirements.txt
```

Verify your installation:

```bash
python -c "import torch, ultralytics, cv2; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Ultralytics:', ultralytics.__version__)"
```

---

## Dataset Specifications

- **Total Frames**: 2,000 video frames (`seq_000001.jpg` through `seq_002000.jpg`).
- **Resolution**: $640 \times 480$ pixels (RGB).
- **Environment**: Overhead indoor shopping mall surveillance footage (central kiosk, decorative palm tree, perimeter storefronts).
- **Annotation Format**:
  - Existing: Image-level count CSV (`data/raw/count/labels.csv`) containing `id` (frame number) and `count` (number of persons).
  - Person count range: **13 to 53 people** per frame ($\mu = 31.16$, $\sigma = 6.95$).
- **Bounding Boxes**: The initial dataset contains **no bounding-box annotations**. Ground-truth bounding boxes must be annotated before training a custom detector (see [Bounding-Box Annotation Workflow](#bounding-box-annotation-workflow)).

> [!WARNING]
> **Temporal Leakage in Frame Splits**: Consecutive surveillance frames are visually near-identical. Splitting frames randomly using `train_test_split` results in 88.7% of validation frames having an immediate neighbor in the training set. For detector training, use `python main.py data-prep` which performs **temporal-aware group splitting** to prevent data leakage.

---

## Quick Start

### 1. Detect People & Count with YOLOv8 (Immediate Pretrained Baseline)

```bash
python main.py detect --image data/raw/frames/seq_000001.jpg
```
Output:
- Prints detected count and confidence ranges.
- Annotates image with boxes, confidence scores, and person indices.
- Saves visualization to `outputs/detection/visualizations/det_seq_000001.jpg`.

### 2. Predict Count with ResNet-50 Regression

```bash
python main.py predict --image data/raw/frames/seq_000001.jpg
```
Output:
- Prints estimated continuous count and rounded integer count.

### 3. Run the Full Pretrained Detection Baseline

```bash
python run_baseline.py
```
This automatically runs data validation, evaluates counting on the test set, compares metrics against the ResNet-50 model, generates batch visualizations, and saves error analysis grids.

---

## Pipeline 1: Direct Count Regression (ResNet-50)

### Architecture
- **Backbone**: Pretrained `resnet50(weights=ResNet50_Weights.DEFAULT)`.
- **Head**: Replaces `model.fc` with `nn.Linear(2048, 1)`.
- **Activation**: Linear (unbounded continuous regression).
- **Loss Function**: `SmoothL1Loss` (Huber loss, robust to count outliers).
- **Optimizer**: Adam ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 10^{-4}$).

### Input Preprocessing & Augmentation
- Resized to $384 \times 384$ pixels.
- Augmentations (training):
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomAffine(degrees=10, translate=(0.02, 0.02), shear=4)`
  - `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)`
  - `GaussianBlur(kernel_size=(3,3), p=0.3)` and `RandomAdjustSharpness(p=0.3)`
  - ImageNet normalization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

### Commands

| Task | Command | Description |
|------|---------|-------------|
| Split Data | `python main.py split` | Generates 70/15/15 train/val/test splits in `data/splits/` |
| Train Model | `python main.py train` | Trains for 20 epochs, saves best checkpoint to `models/checkpoints/best_model.pth` |
| Evaluate | `python main.py evaluate` | Evaluates on test set, prints MAE/RMSE, saves `outputs/metrics/test_predictions.csv` |
| Predict Single | `python main.py predict --image <path>` | Predicts count for a single image |

---

## Pipeline 2: Person Detection & Counting (YOLOv8)

### Inference Workflow
```
Input Image 
    │
    ▼
YOLOv8 Feature Extraction & Anchor-Free Head (Letterbox 640×640)
    │
    ▼
Confidence Threshold Filtering (conf >= 0.25)
    │
    ▼
Person Class Filtering (COCO Class 0 / Custom Class 0)
    │
    ▼
Non-Maximum Suppression (IoU >= 0.45)
    │
    ▼
Final Person Detections (xyxy in original coordinates)
    │
    ▼
Count = len(person_detections)
```

### Commands

| Task | Command | Description |
|------|---------|-------------|
| **Single Detection** | `python main.py detect --image data/raw/frames/seq_000001.jpg` | Runs detection, displays count, saves annotated image |
| **Custom Threshold** | `python main.py detect --image <path> --conf 0.15` | Runs detection with custom confidence threshold |
| **Evaluate Split** | `python main.py detect-eval --split test` | Evaluates detection-based counting over test set |
| **Compare Models** | `python main.py detect-eval --split test --compare` | Side-by-side comparison with ResNet-50 regression |
| **Error Analysis** | `python main.py detect-errors --split test` | Categorizes undercounts, overcounts, small boxes, generates grid |
| **Prepare Data** | `python main.py data-prep` | Inventories dataset, checks labels, creates temporal-aware splits |
| **Train Custom** | `python main.py detect-train` | Trains custom YOLOv8 on reviewed bbox annotations |

---

## Why Pretrained Detectors Undercount & Threshold Guidance

### The Domain Gap
When running the off-the-shelf COCO-pretrained YOLOv8 on this surveillance dataset, you will observe systematic undercounting (e.g., detecting ~20 people on an image with a ground truth of 35):

1. **Distant & Small People**: COCO is predominantly composed of eye-level, prominent objects. In an overhead surveillance view, people in the background occupy as few as $15 \times 25$ pixels.
2. **Severe Occlusions**: People standing behind the central kiosk or palm tree planters only have their heads/shoulders visible.
3. **Overlapping Crowds**: Pedestrians walking in pairs or groups can have overlapping bodies that standard NMS may merge.
4. **Lighting & Contrast**: Mall floors reflect overhead ambient lighting, reducing contrast between distant pedestrians and the background.

### Confidence Threshold Sensitivity

Testing on validation samples shows the following trade-offs:

| Confidence (`--conf`) | Avg Predicted | Avg Ground Truth | MAE | Behavior |
|:---:|:---:|:---:|:---:|---|
| **0.50** | 13.6 | 36.1 | 22.5 | Heavy undercounting; only detects prominent foreground persons. |
| **0.25** (default) | 20.2 | 36.1 | 15.9 | Detects clear bodies; misses distant and partially occluded persons. |
| **0.15** | 22.6 | 36.1 | 13.5 | Detects more background pedestrians; small risk of false positives. |
| **0.10** | 25.1 | 36.1 | 11.0 | Catches faint distant figures; may detect mannequins in shop windows. |

> [!TIP]
> **Threshold Tuning Guidelines**:
> - **Lowering `--conf`** (e.g. `0.10 - 0.15`) increases recall on distant persons, but can introduce false positives on mannequins and shop displays.
> - **Raising `--conf`** (e.g. `0.40 - 0.60`) guarantees high-precision foreground detections but will miss smaller subjects.
> - **NMS IoU (`--iou`)**: A lower value (`0.30`) aggressively eliminates duplicates but may suppress two people walking closely together. A higher value (`0.60`) preserves adjacent individuals.
> - **Tuning Strategy**: Always tune thresholds on the **validation split** (`data/splits/val.csv`), never on the test split.

---

## Bounding-Box Annotation Workflow

To achieve accurate counts that match or exceed the ResNet-50 regression model, YOLOv8 must be fine-tuned on **scene-specific bounding boxes**.

### 1. Label Format: Normalized YOLO Format

Each image in `data/raw/frames/<filename>.jpg` must have a corresponding text file `data/raw/labels/<filename>.txt`:

```text
data/raw/frames/seq_000001.jpg  ──►  data/raw/labels/seq_000001.txt
```

Each line in the file defines a single person bounding box:
```text
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized floats in $[0.0, 1.0]$:
$$\begin{aligned}
x_{\text{center}} &= \frac{\text{box center } x}{\text{image width}} = \frac{x_{\text{min}} + x_{\text{max}}}{2 \times 640} \\
y_{\text{center}} &= \frac{\text{box center } y}{\text{image height}} = \frac{y_{\text{min}} + y_{\text{max}}}{2 \times 480} \\
\text{width} &= \frac{x_{\text{max}} - x_{\text{min}}}{640} \\
\text{height} &= \frac{y_{\text{max}} - y_{\text{min}}}{480}
\end{aligned}$$

For custom single-class training:
- **`class_id` = 0** (person).

### 2. Annotation Policy

To ensure high consistency between your bounding boxes and count ground truth:

| Scenario | Annotation Rule | Rationale |
|----------|-----------------|-----------|
| **Full Body** | Draw a tight bounding box from head to feet. | Standard object detection label. |
| **Partially Occluded Body ($\ge 50\%$)** | Box the entire visible extent of the person. | Preserves detector ability to learn torso/legs. |
| **Head/Shoulders Only ($< 50\%$)** | Box visible head/shoulders if clearly a person. | Distant/seated persons behind planters must be detected to match counts. |
| **People at Image Boundaries** | Box visible portion; clip tightly to border. | Avoids penalizing truncated detections. |
| **Distant People ($< 25\text{ px}$ height)** | Box if identifiable as a human figure. | Eliminates systematic distant undercounting. |
| **Mannequins & Store Displays** | **DO NOT ANNOTATE.** | Prevents false positives from retail displays. |
| **Mirrors & Reflections** | **DO NOT ANNOTATE.** | Mall floors/windows reflect figures; do not label reflections. |
| **Ambiguous Blobs** | **DO NOT ANNOTATE.** | Unidentifiable shapes degrade gradient stability. |

### 3. Semi-Automated Annotation (Proposals + Human Review)

Do not label completely from scratch. Use the pretrained detector to generate candidate proposals:

```bash
# Output detections as JSON for review tools (CVAT, Label Studio, Roboflow)
python -m src.detection.predict_detection --image data/raw/frames/seq_000001.jpg --conf 0.15 --json
```

> [!CAUTION]
> **Never treat model predictions as ground truth!** Pretrained predictions must always be reviewed, adjusted, and confirmed by a human annotator before being moved to `data/raw/labels/`.

### 4. Annotation Status & Verification

The status of all images is tracked in `data/raw/labels/annotation_status.json`:
- `annotated`: Label file exists and contains $\ge 1$ boxes.
- `empty`: Label file exists and is empty ($0$ bytes, verified empty frame).
- `unannotated`: Missing label file.

Run data preparation to validate all boxes and ensure zero format errors:

```bash
python main.py data-prep
```
The validator checks:
- Exact 5 columns per line.
- Valid class ID (0).
- Finite floats in $[0, 1]$.
- Strictly positive width and height.

### 5. Train Custom Detector

Once at least 50–200 representative frames are annotated:

```bash
python main.py detect-train
```
This fine-tunes YOLOv8n, employs early stopping, evaluates on validation splits, and saves checkpoints to `outputs/detection/training/person_detector/weights/best.pt`.

---

## Configuration Reference

All detection, evaluation, augmentation, and training parameters are centralized in [`configs/detection_config.yaml`](file:///c:/Users/hp/Desktop/Crowd-Counting/configs/detection_config.yaml):

```yaml
# Model Architecture & Checkpoints
model:
  checkpoint: "yolov8n.pt"          # Model size: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt
  coco_person_class_id: 0          # COCO class index for person
  custom_person_class_id: 0        # Custom dataset class index

# Inference Thresholds
inference:
  image_size: 640                  # Input resolution (letterbox preserves 4:3 aspect ratio)
  confidence_threshold: 0.25       # Minimum confidence to accept box
  nms_iou_threshold: 0.45          # IoU threshold for duplicate removal
  max_detections: 300              # Cap on detections per image
  device: "cuda"                   # "cuda", "cpu", or "cuda:0"

# Custom Detector Training
training:
  epochs: 100
  batch_size: 8                    # Fits comfortably in 4GB GPU VRAM
  image_size: 640
  learning_rate: 0.01
  optimizer: "auto"
  patience: 15                     # Early stopping patience
  seed: 42
  workers: 2
  close_mosaic: 10                 # Disables mosaic 10 epochs before end for cleaner boxes

# Detection-Specific Augmentations
augmentation:
  hsv_h: 0.015                     # Hue shift
  hsv_s: 0.5                       # Saturation shift
  hsv_v: 0.3                       # Brightness shift
  degrees: 5.0                     # Subtle rotation (overhead camera angle)
  translate: 0.05                  # Translation shift
  scale: 0.3                       # Scale jittering
  shear: 2.0
  fliplr: 0.5                      # Horizontal flip
  mosaic: 0.5                      # Reduced from 1.0 for dense crowd stability
```

---

## Evaluation Metrics & Error Analysis

### 1. Counting Metrics

When evaluating counting predictions against ground truth counts $y_i$ and $\hat{y}_i$:

- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
- **Mean Signed Error (Bias)**:
  $$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
  *(Positive indicates systematic overcounting; negative indicates systematic undercounting).*
- **Mean Absolute Percentage Error (MAPE)**:
  $$\text{MAPE} = \frac{100\%}{N_{\text{nonzero}}} \sum_{y_i > 0} \frac{|y_i - \hat{y}_i|}{y_i}$$
- **Accuracy Within Tolerance**:
  $$\text{Acc}_{\pm k} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(|y_i - \hat{y}_i| \le k) \times 100\% \quad (k \in \{1, 2, 5\})$$

### 2. Detection Metrics (Available after BBox Ground Truth)

- **Precision, Recall, F1** evaluated at IoU $\ge 0.50$.
- **mAP@50**: Mean Average Precision at IoU threshold 0.50.
- **mAP@50:95**: Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 (COCO standard).

### 3. Automated Error Analysis

Run the error analysis diagnostic tool:

```bash
python main.py detect-errors --split test
```

This generates:
- `outputs/detection/error_analysis/test_error_analysis.json`: Categorized breakdowns of:
  - **Large undercounting** ($y_i - \hat{y}_i \ge 5$)
  - **Large overcounting** ($\hat{y}_i - y_i \ge 5$)
  - **Dense scenes** ($\ge 40$ people detected)
  - **Small people detection** (boxes $< 1000 \text{ px}^2$)
  - **Overlapping detections** (pairwise IoU $> 0.5$)
- `outputs/detection/error_analysis/test_error_grid.jpg`: A visual montage of worst-error images with overlaid predictions and ground-truth comparisons.

---

## Architectural Comparison: Regression vs. Detection vs. Density Maps

| Property | Direct Count Regression (ResNet-50) | Person Detection (YOLOv8) | Density-Map Estimation (CSRNet / MCNN) |
|---|---|---|---|
| **Pipeline** | Image $\to$ Backbone $\to$ Linear $\to$ Count | Image $\to$ YOLO $\to$ Boxes $\to$ $N = \text{len}(\text{boxes})$ | Image $\to$ CNN $\to$ Density Map $\to \int \text{density}$ |
| **Annotation Requirement** | Total count only | Full bounding boxes ($x, y, w, h$) | Point annotations (head/body centers) |
| **Spatial Localization** | None (black-box scalar) | **Full bounding box coordinates** | Continuous spatial heatmap |
| **Individual Verification** | Impossible to visually verify | **Directly inspectable & debuggable** | Qualitative density blobs |
| **Performance in Extreme Crowds** ($> 200/\text{frame}$) | Poor (severe scale variance) | Struggles due to severe occlusion | **Best suited** (standard for stadium/concerts) |
| **Performance in Moderate Crowds** ($10-60/\text{frame}$) | Good count accuracy | **Ideal** (each person is a distinct object) | Overkill; produces blurry density blobs |
| **Applicability to this Dataset** | Functional; already trained ($\text{MAE} \approx 1.5$) | **Implemented**; optimal once custom-trained | Incompatible without point annotations |

---

## Troubleshooting & FAQ

### 1. `CUDA out of memory` during training or inference
- Reduce `batch_size` in `configs/detection_config.yaml` (e.g. from `8` to `4` or `2`).
- Reduce `inference.image_size` from `640` to `512` or `480`.

### 2. Detections miss people in shop windows or background
- Test with lower confidence threshold:
  ```bash
  python main.py detect --image <path> --conf 0.12
  ```
- Fine-tune custom YOLOv8 on images where distant people are explicitly labeled.

### 3. Too many duplicate boxes on one person
- Lower the NMS IoU threshold in `configs/detection_config.yaml` from `0.45` to `0.35`.

### 4. `Labels directory does not exist` when running `detect-train`
- Supervised detector training cannot run without ground-truth boxes.
- Create `data/raw/labels/`, place your `.txt` annotation files there, and run `python main.py data-prep` before calling `detect-train`.

---

## License

This project is licensed under the Apache License 2.0. Note that the Ultralytics YOLOv8 library is licensed under AGPL-3.0 for open-source use.