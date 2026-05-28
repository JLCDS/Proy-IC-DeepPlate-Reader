"""CLI entry point for inference on images, video files, or camera."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepplate.pipeline import LPRPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepPlate-Reader inference")
    p.add_argument("--config", default="configs/pipeline.yaml")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a single image or a folder of images")
    src.add_argument("--video", help="Path to a video file")
    src.add_argument("--camera", type=int, help="Camera device index (e.g. 0)")
    p.add_argument("--output", default=None, help="Output video path (video mode only)")
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    pipeline = LPRPipeline(config)

    if args.image:
        path = Path(args.image)
        images = list(path.glob("*.jpg")) + list(path.glob("*.png")) if path.is_dir() else [path]
        for img in images:
            results = pipeline.run_on_image(img)
            for r in results:
                print(f"{img.name}  ->  {r.plate_text}  [{r.ocr_confidence:.2f}]")

    elif args.video:
        pipeline.run_on_video(
            source=args.video,
            show=not args.no_preview,
            output_path=args.output,
        )

    elif args.camera is not None:
        pipeline.run_on_video(
            source=args.camera,
            show=not args.no_preview,
        )


if __name__ == "__main__":
    main()
