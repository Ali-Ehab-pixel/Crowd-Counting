import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from src.detection.config import load_config


class PersonDetector:
    """Person detector using YOLOv8.
    
    Wraps a pretrained COCO YOLOv8 model or a custom-trained model.
    Returns bounding boxes in original image coordinates.
    """
    
    def __init__(self, config=None, custom_model_path=None):
        """Initialize detector.
        
        Args:
            config: Config dict or None to load from YAML.
            custom_model_path: Path to custom-trained weights. If None,
                uses pretrained COCO model from config.
        """
        if config is None:
            config = load_config()
        self.config = config
        self.is_custom = custom_model_path is not None
        
        model_path = custom_model_path or config['model']['checkpoint']
        self.model = YOLO(model_path)
        
        # Person class ID depends on whether using COCO or custom model
        if self.is_custom:
            self.person_class_id = config['model']['custom_person_class_id']
        else:
            self.person_class_id = config['model']['coco_person_class_id']
        
        self.conf_threshold = config['inference']['confidence_threshold']
        self.iou_threshold = config['inference']['nms_iou_threshold']
        self.max_det = config['inference']['max_detections']
        self.img_size = config['inference']['image_size']
        self.device = config['inference']['device']
    
    def detect(self, image, conf=None, iou=None, max_det=None):
        """Run person detection on a single image.
        
        Args:
            image: Image path (str/Path), PIL Image, or numpy array (BGR or RGB).
            conf: Override confidence threshold.
            iou: Override NMS IoU threshold.
            max_det: Override max detections.
            
        Returns:
            dict with keys:
                'bounding_boxes': np.ndarray shape (N, 4) in [x1, y1, x2, y2] 
                    format, original image coordinates
                'confidence_scores': np.ndarray shape (N,)
                'class_ids': np.ndarray shape (N,) of int
                'count': int, number of person detections
        """
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold
        max_det = max_det if max_det is not None else self.max_det
        
        # YOLOv8 handles preprocessing (letterbox resize preserving aspect ratio),
        # inference, NMS, and coordinate scaling back to original image internally.
        # We do NOT apply separate NMS or normalization.
        results = self.model(
            image,
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
            classes=[self.person_class_id],  # filter to person class only
        )
        
        result = results[0]  # single image
        boxes = result.boxes
        
        if len(boxes) == 0:
            return {
                'bounding_boxes': np.empty((0, 4), dtype=np.float32),
                'confidence_scores': np.empty(0, dtype=np.float32),
                'class_ids': np.empty(0, dtype=np.int32),
                'count': 0,
            }
        
        # boxes.xyxy: (N, 4) tensor in original image coordinates
        bboxes = boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = boxes.conf.cpu().numpy().astype(np.float32)
        cls_ids = boxes.cls.cpu().numpy().astype(np.int32)
        
        return {
            'bounding_boxes': bboxes,
            'confidence_scores': confs,
            'class_ids': cls_ids,
            'count': len(bboxes),
        }
    
    def detect_batch(self, images, conf=None, iou=None, max_det=None):
        """Run detection on multiple images.
        
        Args:
            images: List of image paths, PIL images, or numpy arrays.
            
        Returns:
            List of detection dicts.
        """
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold
        max_det = max_det if max_det is not None else self.max_det
        
        results = self.model(
            images,
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
            classes=[self.person_class_id],
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                detections.append({
                    'bounding_boxes': np.empty((0, 4), dtype=np.float32),
                    'confidence_scores': np.empty(0, dtype=np.float32),
                    'class_ids': np.empty(0, dtype=np.int32),
                    'count': 0,
                })
            else:
                bboxes = boxes.xyxy.cpu().numpy().astype(np.float32)
                confs = boxes.conf.cpu().numpy().astype(np.float32)
                cls_ids = boxes.cls.cpu().numpy().astype(np.int32)
                detections.append({
                    'bounding_boxes': bboxes,
                    'confidence_scores': confs,
                    'class_ids': cls_ids,
                    'count': len(bboxes),
                })
        
        return detections
