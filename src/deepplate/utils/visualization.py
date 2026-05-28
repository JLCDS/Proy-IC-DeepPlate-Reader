from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class DetectionResult:
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    plate_text: str
    confidence: float
    ocr_confidence: float


def draw_results(frame: np.ndarray, results: list[DetectionResult]) -> np.ndarray:
    out = frame.copy()
    for r in results:
        x1, y1, x2, y2 = r.bbox
        color = (0, 255, 0) if r.ocr_confidence >= 0.7 else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{r.plate_text}  [{r.ocr_confidence:.2f}]"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out
