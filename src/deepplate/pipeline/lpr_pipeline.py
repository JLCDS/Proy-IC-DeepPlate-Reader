from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from ..detection import PlateDetector
from ..enhancement import ImageEnhancer
from ..enhancement.preprocessor import EnhancerConfig
from ..ocr import PlateOCR
from ..utils import get_logger, draw_results
from ..utils.visualization import DetectionResult

logger = get_logger("pipeline")


class LPRPipeline:
    """End-to-end License Plate Recognition pipeline."""

    def __init__(self, config: dict) -> None:
        det_cfg = config["detection"]
        enh_cfg = config.get("enhancement", {})
        ocr_cfg = config.get("ocr", {})

        self.detector = PlateDetector(
            model_path=det_cfg["model_path"],
            confidence=det_cfg.get("confidence", 0.45),
            iou=det_cfg.get("iou_threshold", 0.5),
            device=det_cfg.get("device", "cpu"),
        )
        self.enhancer = ImageEnhancer(EnhancerConfig(
            denoise=enh_cfg.get("denoise", True),
            denoise_strength=enh_cfg.get("denoise_strength", 10),
            upscale_factor=enh_cfg.get("upscale_factor", 2),
            contrast_clip_limit=enh_cfg.get("contrast_clip_limit", 2.0),
        )) if enh_cfg.get("enabled", True) else None
        self.ocr = PlateOCR(
            languages=ocr_cfg.get("languages", ["en"]),
            min_confidence=ocr_cfg.get("min_confidence", 0.6),
        )

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[DetectionResult]]:
        bboxes = self.detector.detect(frame)
        results: list[DetectionResult] = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if self.enhancer:
                crop = self.enhancer.enhance(crop)
            text, conf = self.ocr.read(crop)
            if text:
                results.append(DetectionResult(
                    bbox=bbox, plate_text=text, confidence=1.0, ocr_confidence=conf
                ))
                logger.info(f"Plate detected: {text} ({conf:.2f})")

        annotated = draw_results(frame, results)
        return annotated, results

    def run_on_image(self, image_path: str | Path) -> list[DetectionResult]:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        _, results = self.process_frame(frame)
        return results

    def run_on_video(self, source: str | int, show: bool = True,
                     frame_skip: int = 2, output_path: str | None = None) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        writer = None
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue
                annotated, _ = self.process_frame(frame)
                if writer:
                    writer.write(annotated)
                if show:
                    cv2.imshow("DeepPlate-Reader", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
