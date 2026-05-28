"""Fine-tune / train a YOLOv8 model for plate detection."""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train plate detection model")
    p.add_argument("--data", required=True, help="Path to data.yaml (YOLO format)")
    p.add_argument("--weights", default="yolov8n.pt", help="Base weights to start from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    p.add_argument("--name", default="plate_detector", help="Run name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="models/detection",
        name=args.name,
        exist_ok=True,
    )
    print(f"Training complete. Weights saved to models/detection/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
