from __future__ import annotations
import re
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog

from ..utils import get_logger

logger = get_logger("ocr")

# Colombian car plate: ABC123 (3 letters + 3 digits)
_PLATE_RE = re.compile(r"[A-Z]{3}\d{3}", re.IGNORECASE)
TARGET_SIZE = (32, 32)


def _prepare_char(img: np.ndarray) -> np.ndarray:
    """
    Binarize → pad to square preserving aspect ratio → resize to TARGET_SIZE.

    Binarizing first removes plate-color information (yellow/white background) and
    eliminates artificial gradients that appear when gray-scale background values
    differ from the 255 padding, which would otherwise confuse HOG features.
    """
    # 1. Binarize: character pixels → 0 (black), background → 255 (white)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # After Otsu, background is usually brighter; invert so character is dark (0)
    if binary.mean() > 128:
        binary = 255 - binary

    # 2. Pad to square so aspect ratio is preserved (no stretching distortion)
    h, w = binary.shape
    side = max(h, w)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off: y_off + h, x_off: x_off + w] = binary

    # 3. Resize to fixed size for HOG
    return cv2.resize(canvas, TARGET_SIZE, interpolation=cv2.INTER_AREA)


class PlateOCR:
    """HOG feature extraction + pre-trained SVM classifier for alphanumeric characters."""

    def __init__(
        self,
        model_path: str | Path,
        label_encoder_path: str | Path,
        min_chars: int = 4,
        min_confidence: float = 0.4,
    ) -> None:
        model_path = Path(model_path)
        le_path = Path(label_encoder_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"OCR model not found at {model_path}. "
                "Run 'python scripts/train.py' first."
            )
        self.clf = joblib.load(model_path)
        self.le = joblib.load(le_path)
        self.min_chars = min_chars
        self.min_conf = min_confidence
        logger.info(f"OCR model loaded from {model_path}")

    def read(self, char_crops: list[np.ndarray]) -> tuple[str, float]:
        """
        Classify each character crop and assemble the plate string.

        Returns (plate_text, mean_confidence). Empty string if recognition fails.
        """
        if len(char_crops) < self.min_chars:
            logger.debug(f"Too few chars ({len(char_crops)} < {self.min_chars}), skipping")
            return "", 0.0

        features = np.array([self._extract_hog(c) for c in char_crops], dtype=np.float32)
        probs = self.clf.predict_proba(features)   # shape (n_chars, n_classes)
        label_ids = probs.argmax(axis=1)
        chars = self.le.inverse_transform(label_ids)
        confidences = probs.max(axis=1)

        mean_conf = float(confidences.mean())
        if mean_conf < self.min_conf:
            return "", mean_conf

        raw = "".join(chars).upper()
        plate = self._postprocess(raw)
        logger.debug(f"OCR raw='{raw}'  post='{plate}'  conf={mean_conf:.2f}")
        return plate, mean_conf

    @staticmethod
    def _extract_hog(img: np.ndarray) -> np.ndarray:
        resized = _prepare_char(img)
        return hog(resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)

    @staticmethod
    def _postprocess(raw: str) -> str:
        cleaned = re.sub(r"[^A-Z0-9]", "", raw)
        match = _PLATE_RE.search(cleaned)
        return match.group() if match else cleaned
