"""Train the HOG + SVM character classifier from synthetic data."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepplate.ocr.trainer import train_svm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OCR model (HOG + SVM)")
    p.add_argument(
        "--output-dir", default="models/ocr",
        help="Directory to save ocr_svm.pkl and ocr_le.pkl (default: models/ocr)",
    )
    p.add_argument(
        "--samples", type=int, default=300,
        help="Synthetic samples per character class (default: 300)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_svm(output_dir=args.output_dir, samples_per_class=args.samples)
    print(f"\nDone. Models saved to: {args.output_dir}/")
    print("  ocr_svm.pkl  — SVM classifier")
    print("  ocr_le.pkl   — LabelEncoder (index → character)")


if __name__ == "__main__":
    main()
