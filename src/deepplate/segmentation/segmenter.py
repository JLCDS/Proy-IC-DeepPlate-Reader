from __future__ import annotations
import cv2
import numpy as np
from ..utils import get_logger

logger = get_logger("segmentation")

PLATE_STD_HEIGHT = 80  # pixels — normalize plate to this height before segmenting


class CharacterSegmenter:

    def __init__(
        self,
        min_char_height_ratio: float = 0.25,
        max_char_height_ratio: float = 0.98,
        min_char_aspect: float = 0.1,
        max_char_aspect: float = 1.1,
        expected_chars: int = 6,
    ) -> None:
        self.min_h = min_char_height_ratio
        self.max_h = max_char_height_ratio
        self.min_asp = min_char_aspect
        self.max_asp = max_char_aspect
        self.expected = expected_chars

    BORDER_TRIM = 0.06

    def segment(self, plate_crop: np.ndarray) -> list[np.ndarray]:
        gray = self._to_gray(plate_crop)
        gray = self._resize_to_std(gray)
        gray = self._trim_border(gray)
        binary = self._binarize(gray)

        plate_h, plate_w = binary.shape
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[int, np.ndarray]] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if x <= 1 or y <= 1 or x + cw >= plate_w - 1 or y + ch >= plate_h - 1:
                continue
            h_ratio = ch / plate_h
            aspect = cw / ch if ch > 0 else 0
            if self.min_h <= h_ratio <= self.max_h and self.min_asp <= aspect <= self.max_asp:
                candidates.append((x, gray[y: y + ch, x: x + cw]))

        candidates.sort(key=lambda c: c[0])
        chars = [c[1] for c in candidates]

        if len(chars) < self.expected // 2:
            chars = self._projection_segment(gray, binary)

        logger.debug(f"Segmented {len(chars)} character(s) from plate")
        return chars

    def _projection_segment(self, gray: np.ndarray, binary: np.ndarray) -> list[np.ndarray]:
        col_sums = binary.sum(axis=0).astype(float)
        kernel = np.ones(3) / 3
        col_sums = np.convolve(col_sums, kernel, mode="same")

        threshold = col_sums.max() * 0.15
        in_char = False
        segments: list[tuple[int, int]] = []
        start = 0
        for i, v in enumerate(col_sums):
            if not in_char and v > threshold:
                in_char = True
                start = i
            elif in_char and v <= threshold:
                in_char = False
                if i - start >= 4:
                    segments.append((start, i))
        if in_char:
            segments.append((start, len(col_sums)))

        chars = []
        for s, e in segments:
            col = gray[:, s:e]
            h, w = col.shape
            if w > 0 and h > 0:
                chars.append(col)
        return chars

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    @staticmethod
    def _resize_to_std(gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        new_w = int(w * PLATE_STD_HEIGHT / h)
        return cv2.resize(gray, (new_w, PLATE_STD_HEIGHT), interpolation=cv2.INTER_AREA)

    def _trim_border(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        t = max(1, int(h * self.BORDER_TRIM))
        l = max(1, int(w * self.BORDER_TRIM))
        return gray[t: h - t, l: w - l]

    @staticmethod
    def _binarize(gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(binary) > 200:
            binary = cv2.bitwise_not(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary
