import os
import torch
from ultralytics import YOLO
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from vision_explanation_methods.explanations.drise import DRISE_saliency
from vision_explanation_methods.explanations.common import DetectionRecord, GeneralObjectDetectionModelWrapper
from vision_explanation_methods.explanations.drise import generate_mask, fuse_mask, compute_affinity_scores, saliency_fusion, MaskAffinityRecord
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging
import tempfile
import pickle
import random
logging.getLogger("ultralytics").setLevel(logging.ERROR)


class YOLOv8Wrapper(GeneralObjectDetectionModelWrapper):
    """Wrapper for YOLOv8 to make it compatible with D-RISE."""
    def __init__(self, model_path: str):
        self.model = YOLO(model_path, verbose=False)
        self.model.to(device="cuda" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available
        self.num_classes = self.model.model.yaml['nc']  # Number of classes in the model

    def predict(self, x: torch.Tensor, conf_threshold: float = 0.2) -> list:
        """
        Run predictions and return detections in DetectionRecord format.
        
        Args:
            x (torch.Tensor): Input image tensor.
            conf_threshold (float): Confidence threshold for predictions.


        Returns:
            list: List of DetectionRecord objects containing bounding boxes, scores, and class scores.
        """

        # Run YOLOv8 predictions
        results = self.model.predict(source=x, conf=conf_threshold, verbose=False)
        # Convert results to DetectionRecord format
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.clone().detach().to(dtype=torch.float32)  # [N, 4]
            scores = result.boxes.conf.clone().detach().to(dtype=torch.float32)  # [N]
            labels = result.boxes.cls.clone().detach().to(dtype=torch.int64)  # [N]
            
            class_scores = torch.zeros((len(labels), self.num_classes), dtype=torch.float32)
            for i, label in enumerate(labels):
                class_scores[i, label] = scores[i]
            detections.append(DetectionRecord(bounding_boxes=boxes, objectness_scores=scores, class_scores=class_scores))
        return detections
