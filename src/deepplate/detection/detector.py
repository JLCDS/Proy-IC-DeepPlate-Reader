from __future__ import annotations
import cv2
import numpy as np
from ..utils import get_logger

logger = get_logger("detection")


class PlateDetector:

    def __init__(
        self,
        min_area_ratio: float = 0.002,
        max_area_ratio: float = 0.15,
        min_aspect: float = 1.8,
        max_aspect: float = 6.0,
        canny_low: int = 50,
        canny_high: int = 200,
        blur_ksize: int = 5,
    ) -> None:
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_ksize = blur_ksize
        logger.info("PlateDetector initialized (classical CV mode)")

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        total_area = h * w

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, self.canny_low, self.canny_high)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[int, int, int, int]] = []
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
            area = cv2.contourArea(cnt)
            if not (total_area * self.min_area_ratio <= area <= total_area * self.max_area_ratio):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh > 0 else 0
            if self.min_aspect <= aspect <= self.max_aspect:
                candidates.append((x, y, x + bw, y + bh))

        return self._nms(candidates)

    def crop_plates(self, image: np.ndarray) -> list[np.ndarray]:
        h, w = image.shape[:2]
        crops = []
        for x1, y1, x2, y2 in self.detect(image):
            pad = 4
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            crops.append(image[y1:y2, x1:x2])
        return crops

    @staticmethod
    def _nms(
        boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.3
    ) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []
        boxes_arr = np.array(boxes, dtype=float)
        x1, y1, x2, y2 = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou < iou_threshold]
        return [boxes[k] for k in keep]
