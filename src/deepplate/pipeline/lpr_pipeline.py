from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from ..detection import PlateDetector
from ..enhancement import ImageEnhancer
from ..enhancement.preprocessor import EnhancerConfig
from ..segmentation import CharacterSegmenter
from ..ocr import PlateOCR
from ..utils import get_logger, draw_results
from ..utils.visualization import DetectionResult

logger = get_logger("pipeline")


class LPRPipeline:
    """End-to-end License Plate Recognition pipeline (classical CV + HOG + SVM)."""

    def __init__(self, config: dict) -> None:
        det = config.get("detection", {})
        enh = config.get("enhancement", {})
        seg = config.get("segmentation", {})
        ocr = config.get("ocr", {})

        self.detector = PlateDetector(
            min_area_ratio=det.get("min_area_ratio", 0.002),
            max_area_ratio=det.get("max_area_ratio", 0.15),
            min_aspect=det.get("min_aspect", 1.8),
            max_aspect=det.get("max_aspect", 6.0),
            canny_low=det.get("canny_low", 50),
            canny_high=det.get("canny_high", 200),
            blur_ksize=det.get("blur_ksize", 5),
        )
        self.enhancer = (
            ImageEnhancer(EnhancerConfig(
                denoise=enh.get("denoise", True),
                denoise_strength=enh.get("denoise_strength", 10),
                upscale_factor=enh.get("upscale_factor", 2),
                contrast_clip_limit=enh.get("contrast_clip_limit", 2.0),
            ))
            if enh.get("enabled", True)
            else None
        )
        self.segmenter = CharacterSegmenter(
            min_char_height_ratio=seg.get("min_char_height_ratio", 0.25),
            max_char_height_ratio=seg.get("max_char_height_ratio", 0.98),
            min_char_aspect=seg.get("min_char_aspect", 0.1),
            max_char_aspect=seg.get("max_char_aspect", 1.1),
            expected_chars=seg.get("expected_chars", 6),
        )
        self.ocr = PlateOCR(
            model_path=ocr.get("model_path", "models/ocr/ocr_svm.pkl"),
            label_encoder_path=ocr.get("label_encoder_path", "models/ocr/ocr_le.pkl"),
            min_chars=ocr.get("min_chars", 4),
            min_confidence=ocr.get("min_confidence", 0.4),
        )

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[DetectionResult]]:
        bboxes = self.detector.detect(frame)
        results: list[DetectionResult] = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if self.enhancer:
                crop = self.enhancer.enhance(crop)
            char_crops = self.segmenter.segment(crop)
            text, conf = self.ocr.read(char_crops)
            if text:
                results.append(DetectionResult(
                    bbox=bbox, plate_text=text, confidence=1.0, ocr_confidence=conf
                ))
                logger.info(f"Plate: {text}  (conf={conf:.2f})")

        annotated = draw_results(frame, results)
        return annotated, results

    def run_on_image(self, image_path: str | Path) -> list[DetectionResult]:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        _, results = self.process_frame(frame)
        return results

    def run_on_video(
        self,
        source: str | int,
        show: bool = True,
        frame_skip: int = 2,
        output_path: str | None = None,
    ) -> None:
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
