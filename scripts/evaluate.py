"""Evaluate OCR accuracy on a labeled dataset."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepplate.pipeline import LPRPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DeepPlate-Reader on a labeled dataset")
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--labels", required=True,
                   help="CSV file with columns: image_path,plate_text")
    p.add_argument("--output", default="outputs/eval_results.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    pipeline = LPRPipeline(config)

    rows = list(csv.DictReader(open(args.labels)))
    correct = 0
    out_rows = []

    for row in rows:
        results = pipeline.run_on_image(row["image_path"])
        predicted = results[0].plate_text if results else ""
        gt = row["plate_text"].upper().replace(" ", "")
        pred_clean = predicted.upper().replace(" ", "")
        match = gt == pred_clean
        correct += int(match)
        out_rows.append({"image": row["image_path"], "gt": gt,
                          "pred": pred_clean, "match": match})

    accuracy = correct / len(rows) if rows else 0.0
    print(f"Accuracy: {accuracy:.2%}  ({correct}/{len(rows)})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt", "pred", "match"])
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
