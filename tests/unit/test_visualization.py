import numpy as np
from deepplate.utils.visualization import draw_results, DetectionResult


def test_draw_results_no_detections():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_results(frame, [])
    assert out.shape == frame.shape


def test_draw_results_with_detection():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [DetectionResult(bbox=(10, 10, 200, 60), plate_text="ABC123",
                               confidence=0.9, ocr_confidence=0.85)]
    out = draw_results(frame, results)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
