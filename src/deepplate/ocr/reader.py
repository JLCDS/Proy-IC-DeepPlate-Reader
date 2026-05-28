from __future__ import annotations
import re
import numpy as np
from ..utils import get_logger

logger = get_logger("ocr")

# Colombian plate pattern (adapt as needed): ABC 123
_PLATE_RE = re.compile(r"[A-Z]{3}\s?\d{3}", re.IGNORECASE)


class PlateOCR:
    """Wraps EasyOCR to extract text from plate crops."""

    def __init__(self, languages: list[str] | None = None,
                 min_confidence: float = 0.6) -> None:
        import easyocr
        langs = languages or ["en"]
        logger.info(f"Initializing EasyOCR with languages={langs}")
        self.reader = easyocr.Reader(langs, gpu=self._has_cuda())
        self.min_conf = min_confidence

    def read(self, image: np.ndarray) -> tuple[str, float]:
        """Return (plate_text, confidence). Empty string if nothing found."""
        results = self.reader.readtext(image, detail=1, paragraph=False)
        if not results:
            return "", 0.0
        # Filter low-confidence detections and join remaining tokens
        tokens = [(text, conf) for _, text, conf in results if conf >= self.min_conf]
        if not tokens:
            return "", 0.0
        raw = " ".join(t for t, _ in tokens).upper()
        avg_conf = sum(c for _, c in tokens) / len(tokens)
        plate = self._postprocess(raw)
        return plate, avg_conf

    def _postprocess(self, raw: str) -> str:
        # Remove obvious OCR noise characters
        cleaned = re.sub(r"[^A-Z0-9\s]", "", raw).strip()
        match = _PLATE_RE.search(cleaned)
        if match:
            return match.group().replace(" ", "")
        return cleaned

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
