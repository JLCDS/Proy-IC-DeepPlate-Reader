"""
Train a HOG + SVM character classifier from synthetic data.

Data is generated on-the-fly using PIL (system fonts + augmentation),
so no external dataset is required. Produces two files:
  - ocr_svm.pkl   : trained SVC with probability=True
  - ocr_le.pkl    : fitted LabelEncoder (maps class index -> char)
"""
from __future__ import annotations
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import hog
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from ..utils import get_logger

logger = get_logger("ocr.trainer")

CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
TARGET_SIZE = (32, 32)

# Windows font locations; fallback to PIL default if none found
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/courbd.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def _load_fonts(sizes: list[int] = (24, 28, 32)) -> list:
    fonts: list = []
    for path in _FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        for size in sizes:
            try:
                fonts.append(ImageFont.truetype(path, size))
            except Exception:
                pass
    if not fonts:
        logger.warning("No TrueType fonts found; using PIL default (quality will be lower)")
        fonts = [ImageFont.load_default()]
    logger.info(f"Loaded {len(fonts)} font variants")
    return fonts


def _render_char(char: str, font, img_size: int = 40) -> np.ndarray:
    img = Image.new("L", (img_size, img_size), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img_size - tw) // 2 - bbox[0]
    y = (img_size - th) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    return np.array(img)


def _augment(base: np.ndarray, n: int) -> list[np.ndarray]:
    """Generate n augmented variants of a character image."""
    results = [base]
    for _ in range(n):
        img = base.copy().astype(np.float32)
        # Gaussian noise
        img += np.random.normal(0, random.uniform(5, 25), img.shape)
        img = np.clip(img, 0, 255)
        # Optional blur
        if random.random() < 0.5:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        # Small rotation
        angle = random.uniform(-12, 12)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
        # Brightness jitter
        img = np.clip(img * random.uniform(0.70, 1.30), 0, 255)
        # Low-resolution simulation (downscale then upscale = JPEG/camera blur)
        if random.random() < 0.5:
            factor = random.uniform(0.25, 0.6)
            small_w = max(4, int(w * factor))
            small_h = max(4, int(h * factor))
            small = cv2.resize(img.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        # JPEG compression artifact simulation
        if random.random() < 0.4:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 70)]
            _, enc = cv2.imencode('.jpg', img.astype(np.uint8), encode_param)
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        results.append(img.astype(np.uint8))
    return results


def _extract_hog(img: np.ndarray) -> np.ndarray:
    from .reader import _prepare_char
    return hog(
        _prepare_char(img),
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    )


def generate_dataset(samples_per_class: int = 300) -> tuple[np.ndarray, list[str]]:
    """Render synthetic character images and extract HOG features."""
    fonts = _load_fonts()
    X: list[np.ndarray] = []
    y: list[str] = []

    for char in CHARS:
        raw_imgs: list[np.ndarray] = []
        for font in fonts:
            base = _render_char(char, font)
            per_font = max(1, samples_per_class // len(fonts))
            raw_imgs.extend(_augment(base, per_font))

        for img in raw_imgs[:samples_per_class]:
            X.append(_extract_hog(img))
            y.append(char)

    logger.info(f"Dataset: {len(X)} samples, {len(CHARS)} classes")
    return np.array(X, dtype=np.float32), y


def train_svm(output_dir: str | Path, samples_per_class: int = 300) -> None:
    """Generate synthetic data, train SVM, and save model + label encoder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y_raw = generate_dataset(samples_per_class)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training SVM on {len(X_train)} samples...")
    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    logger.info(f"Test set report:\n{report}")
    acc = (y_pred == y_test).mean()
    logger.info(f"Test accuracy: {acc:.2%}")

    model_path = output_dir / "ocr_svm.pkl"
    le_path = output_dir / "ocr_le.pkl"
    joblib.dump(clf, model_path)
    joblib.dump(le, le_path)
    logger.info(f"Saved model ->{model_path}")
    logger.info(f"Saved label encoder ->{le_path}")
