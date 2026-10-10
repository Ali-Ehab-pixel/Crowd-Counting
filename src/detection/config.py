import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    'model': {
        'checkpoint': 'yolov8n.pt',
        'coco_person_class_id': 0,
        'custom_person_class_id': 0,
    },
    'inference': {
        'image_size': 640,
        'confidence_threshold': 0.25,
        'nms_iou_threshold': 0.45,
        'max_detections': 300,
        'device': 'cuda',
    },
    'training': {
        'epochs': 100,
        'batch_size': 8,
        'image_size': 640,
        'learning_rate': 0.01,
        'optimizer': 'auto',
        'patience': 15,
        'seed': 42,
        'workers': 2,
        'resume': False,
        'freeze': 0,
        'close_mosaic': 10,
    },
    'paths': {
        'frames_dir': 'data/raw/frames',
        'count_labels': 'data/raw/count/labels.csv',
        'existing_splits_dir': 'data/splits',
        'bbox_labels_dir': 'data/raw/labels',
        'detection_dataset_yaml': 'configs/dataset.yaml',
        'detection_splits_dir': 'data/detection_splits',
        'detection_outputs_dir': 'outputs/detection',
        'visualization_dir': 'outputs/detection/visualizations',
        'error_analysis_dir': 'outputs/detection/error_analysis',
        'training_output_dir': 'outputs/detection/training',
        'annotation_status_file': 'data/raw/labels/annotation_status.json',
    },
    'evaluation': {
        'match_iou_threshold': 0.5,
        'compare_with_regression': True,
    },
}

def load_config(config_path='configs/detection_config.yaml'):
    """Load detection config from YAML, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            user_config = yaml.safe_load(f)
        if user_config:
            for section, values in user_config.items():
                if section in config and isinstance(values, dict):
                    config[section].update(values)
                else:
                    config[section] = values
    return config
