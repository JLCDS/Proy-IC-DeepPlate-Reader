"""
Diagnóstico rápido del pipeline completo.

Prueba cada etapa por separado y guarda imágenes en outputs/diagnostico/:
  1. Entrenamiento rápido del SVM (50 muestras/clase ~30s)
  2. Detector de placa sobre imágenes del dataset
  3. Segmentador de caracteres sobre los recortes detectados
  4. OCR completo (detección ->segmentación ->clasificación)

Uso:
    python scripts/diagnostico.py
    python scripts/diagnostico.py --image ruta/imagen.jpg
    python scripts/diagnostico.py --skip-train   # si ya tienes modelos/ocr/ entrenado
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "outputs" / "diagnostico"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_IMAGE = (
    ROOT
    / "data/dataset/Proyecto Placas.v1-primera-version.retinanet/test"
    / "125_jpg.rf.7cc4a67aeb3387070646dc003ba89a18.jpg"
)


# ──────────────────────────────────────────────
# 1. Entrenamiento rápido del SVM
# ──────────────────────────────────────────────
def test_entrenamiento(samples: int = 50) -> None:
    print("\n" + "=" * 60)
    print(f"ETAPA 1 — Entrenamiento SVM ({samples} muestras/clase)")
    print("=" * 60)
    from deepplate.ocr.trainer import train_svm
    train_svm(output_dir=ROOT / "models/ocr", samples_per_class=samples)
    print("[OK] Modelo guardado en models/ocr/")


# ──────────────────────────────────────────────
# 2. Detector de placa
# ──────────────────────────────────────────────
def test_detector(image_path: Path) -> list[tuple[int, int, int, int]]:
    print("\n" + "=" * 60)
    print("ETAPA 2 — Detección de placa (CV clásico)")
    print("=" * 60)
    from deepplate.detection import PlateDetector

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [ERR] No se pudo leer {image_path}")
        return []

    detector = PlateDetector()
    bboxes = detector.detect(img)
    print(f"  Imagen: {image_path.name}  ({img.shape[1]}×{img.shape[0]})")
    print(f"  Detecciones: {len(bboxes)}")

    vis = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"#{i}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"    #{i}  bbox=({x1},{y1},{x2},{y2})  "
              f"tamaño={x2-x1}×{y2-y1}")

    out_path = OUT_DIR / "2_detector.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"  [OK] Guardado ->{out_path}")
    return bboxes


# ──────────────────────────────────────────────
# 3. Segmentador de caracteres
# ──────────────────────────────────────────────
def test_segmentador(image_path: Path, bboxes: list) -> list[np.ndarray]:
    print("\n" + "=" * 60)
    print("ETAPA 3 — Segmentación de caracteres")
    print("=" * 60)
    from deepplate.segmentation import CharacterSegmenter
    from deepplate.enhancement import ImageEnhancer

    img = cv2.imread(str(image_path))
    enhancer = ImageEnhancer()
    segmenter = CharacterSegmenter()

    all_chars: list[np.ndarray] = []

    if not bboxes:
        print("  Sin detecciones previas; usando imagen completa como placa")
        bboxes = [(0, 0, img.shape[1], img.shape[0])]

    for i, (x1, y1, x2, y2) in enumerate(bboxes[:3]):
        crop = img[y1:y2, x1:x2]
        enhanced = enhancer.enhance(crop)
        chars = segmenter.segment(enhanced)
        print(f"  Recorte #{i}: {len(chars)} carácter(es) encontrados")

        # Guardar cada carácter como imagen pequeña y un collage
        char_strip = _make_char_strip(chars)
        out_path = OUT_DIR / f"3_segmentacion_placa{i}.jpg"
        cv2.imwrite(str(out_path), char_strip)
        print(f"  [OK] Guardado ->{out_path}")
        all_chars.extend(chars)

    return all_chars


# ──────────────────────────────────────────────
# 4. OCR completo
# ──────────────────────────────────────────────
def test_ocr(image_path: Path) -> None:
    print("\n" + "=" * 60)
    print("ETAPA 4 — OCR completo (HOG + SVM)")
    print("=" * 60)
    from deepplate.detection import PlateDetector
    from deepplate.enhancement import ImageEnhancer
    from deepplate.segmentation import CharacterSegmenter
    from deepplate.ocr import PlateOCR
    from deepplate.utils import draw_results
    from deepplate.utils.visualization import DetectionResult

    model_path = ROOT / "models/ocr/ocr_svm.pkl"
    le_path = ROOT / "models/ocr/ocr_le.pkl"

    if not model_path.exists():
        print("  [ERR] Modelo no encontrado. Ejecuta primero la Etapa 1.")
        return

    img = cv2.imread(str(image_path))
    detector = PlateDetector()
    enhancer = ImageEnhancer()
    segmenter = CharacterSegmenter()
    ocr = PlateOCR(model_path=model_path, label_encoder_path=le_path)

    bboxes = detector.detect(img)
    results: list[DetectionResult] = []

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        enhanced = enhancer.enhance(crop)
        chars = segmenter.segment(enhanced)
        text, conf = ocr.read(chars)
        if text:
            results.append(DetectionResult(
                bbox=bbox, plate_text=text, confidence=1.0, ocr_confidence=conf
            ))
            print(f"  Placa detectada: '{text}'  (confianza={conf:.2f})")

    if not results:
        print("  Sin placas reconocidas (detecciones sin texto válido)")

    annotated = draw_results(img, results)
    out_path = OUT_DIR / "4_ocr_completo.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"  [OK] Guardado ->{out_path}")


# ──────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────
def _make_char_strip(chars: list[np.ndarray], target_h: int = 64) -> np.ndarray:
    """Apila caracteres horizontalmente en una sola imagen para visualizar."""
    if not chars:
        return np.ones((target_h, target_h), dtype=np.uint8) * 200
    strips = []
    for c in chars:
        h, w = c.shape[:2]
        new_w = max(1, int(w * target_h / h))
        resized = cv2.resize(c, (new_w, target_h), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        strips.append(resized)
        strips.append(np.ones((target_h, 4, 3), dtype=np.uint8) * 100)  # separador
    return np.hstack(strips[:-1]) if strips else np.ones((target_h, 10, 3), np.uint8)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=str(SAMPLE_IMAGE),
                   help="Imagen a procesar (default: imagen del dataset de test)")
    p.add_argument("--skip-train", action="store_true",
                   help="Omitir entrenamiento si ya existe models/ocr/ocr_svm.pkl")
    p.add_argument("--samples", type=int, default=50,
                   help="Muestras por clase para entrenamiento rápido (default: 50)")
    args = p.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Imagen no encontrada: {image_path}")
        sys.exit(1)

    model_exists = (ROOT / "models/ocr/ocr_svm.pkl").exists()
    if not args.skip_train or not model_exists:
        test_entrenamiento(samples=args.samples)
    else:
        print("\nEtapa 1 omitida (--skip-train y modelo ya existe)")

    bboxes = test_detector(image_path)
    test_segmentador(image_path, bboxes)
    test_ocr(image_path)

    print("\n" + "=" * 60)
    print(f"Diagnóstico completo. Resultados en: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
