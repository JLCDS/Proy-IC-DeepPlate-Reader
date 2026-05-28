from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from ..utils import get_logger

logger = get_logger("enhancement")


@dataclass
class EnhancerConfig:
    denoise: bool = True
    denoise_strength: int = 10
    upscale_factor: int = 2
    contrast_clip_limit: float = 2.0


class ImageEnhancer:
    """Classical CV enhancement pipeline for degraded plate crops."""

    def __init__(self, config: EnhancerConfig | None = None) -> None:
        self.cfg = config or EnhancerConfig()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        img = self._upscale(img)
        img = self._denoise(img)
        img = self._equalize_contrast(img)
        img = self._sharpen(img)
        return img

    # --- private steps ---

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        if self.cfg.upscale_factor <= 1:
            return img
        h, w = img.shape[:2]
        return cv2.resize(img, (w * self.cfg.upscale_factor, h * self.cfg.upscale_factor),
                          interpolation=cv2.INTER_CUBIC)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        if not self.cfg.denoise:
            return img
        h = self.cfg.denoise_strength
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    def _equalize_contrast(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.cfg.contrast_clip_limit, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)
