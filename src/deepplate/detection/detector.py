from __future__ import annotations
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from ..utils import get_logger

logger = get_logger("detection")


class PlateDetector:
    """Wraps a YOLOv8 model to locate license plate regions."""

    def __init__(self, model_path: str | Path, confidence: float = 0.45,
                 iou: float = 0.5, device: str = "cpu") -> None:
        self.conf = confidence
        self.iou = iou
        self.device = device
        logger.info(f"Loading detection model from {model_path}")
        self.model = YOLO(str(model_path))

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) bounding boxes."""
        results = self.model.predict(
            image, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False
        )
        boxes = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy().astype(int):
                boxes.append(tuple(box))
        return boxes

    def crop_plates(self, image: np.ndarray) -> list[np.ndarray]:
        """Return cropped plate regions from an image."""
        crops = []
        for x1, y1, x2, y2 in self.detect(image):
            # Add small padding without going out of bounds
            h, w = image.shape[:2]
            pad = 4
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            crops.append(image[y1:y2, x1:x2])
        return crops
