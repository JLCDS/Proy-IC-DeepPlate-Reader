"""
Convert Roboflow RetinaNet CSV dataset to YOLO format.

RetinaNet CSV format (no header):
    filename, x1, y1, x2, y2, class_name

YOLO label format (per image .txt file, normalized):
    class_id  cx  cy  w  h

Usage:
    python scripts/convert_to_yolo.py
    python scripts/convert_to_yolo.py --src data/dataset/MyDataset --dst data/raw/plates
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Roboflow uses "valid" but YOLO convention is "val"
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RetinaNet CSV → YOLO format converter")
    p.add_argument(
        "--src",
        default=r"data\dataset\Proyecto Placas.v1-primera-version.retinanet",
        help="Root folder containing train/valid/test subdirs with _annotations.csv",
    )
    p.add_argument(
        "--dst",
        default=r"data\raw\plates",
        help="Output folder for YOLO-structured dataset",
    )
    return p.parse_args()


def build_class_map(src_root: Path) -> dict[str, int]:
    """Scan all CSVs to collect unique class names and assign integer IDs."""
    classes: set[str] = set()
    for csv_file in src_root.rglob("_annotations.csv"):
        for row in csv.reader(open(csv_file, newline="")):
            if len(row) == 6 and row[5].strip():
                classes.add(row[5].strip())
    return {name: idx for idx, name in enumerate(sorted(classes))}


def convert_split(
    src_split: Path,
    dst_images: Path,
    dst_labels: Path,
    class_map: dict[str, int],
) -> tuple[int, int]:
    """Convert one split. Returns (images_written, labels_written)."""
    csv_path = src_split / "_annotations.csv"
    if not csv_path.exists():
        return 0, 0

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    # Group rows by filename so multi-box images work correctly
    rows_by_image: dict[str, list[tuple[int, int, int, int, str]]] = defaultdict(list)
    for row in csv.reader(open(csv_path, newline="")):
        row = [c.strip() for c in row]
        if len(row) != 6 or not row[0]:
            continue
        fname, x1, y1, x2, y2, cls = row
        if not cls:
            continue
        rows_by_image[fname].append((int(x1), int(y1), int(x2), int(y2), cls))

    images_written = labels_written = 0
    for fname, boxes in tqdm(rows_by_image.items(), desc=f"  {src_split.name}", leave=False):
        src_img = src_split / fname
        if not src_img.exists():
            print(f"  [WARN] image not found: {src_img}")
            continue

        # Get image dimensions (fast header-only read)
        with Image.open(src_img) as im:
            img_w, img_h = im.size

        # Write label file
        label_lines: list[str] = []
        for x1, y1, x2, y2, cls in boxes:
            if cls not in class_map:
                continue
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            # Clamp to [0, 1] to guard against annotation errors
            cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
            bw, bh = max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))
            label_lines.append(f"{class_map[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not label_lines:
            continue

        (dst_labels / (src_img.stem + ".txt")).write_text("\n".join(label_lines))
        shutil.copy2(src_img, dst_images / fname)
        images_written += 1
        labels_written += 1

    return images_written, labels_written


def write_data_yaml(dst_root: Path, class_map: dict[str, int]) -> None:
    names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]
    lines = [
        f"path: {dst_root.as_posix()}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        "",
        f"nc: {len(names)}",
        "names:",
    ] + [f"  - {n}" for n in names]
    (dst_root / "data.yaml").write_text("\n".join(lines))
    print(f"  data.yaml written → {dst_root / 'data.yaml'}")


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise SystemExit(f"Source not found: {src_root}")

    print("Scanning class names …")
    class_map = build_class_map(src_root)
    print(f"  Classes: {class_map}")

    total_imgs = total_labels = 0
    for src_name, yolo_name in SPLIT_MAP.items():
        src_split = src_root / src_name
        if not src_split.exists():
            continue
        print(f"Converting '{src_name}' → '{yolo_name}' …")
        imgs, lbls = convert_split(
            src_split=src_split,
            dst_images=dst_root / "images" / yolo_name,
            dst_labels=dst_root / "labels" / yolo_name,
            class_map=class_map,
        )
        print(f"  {imgs} images  |  {lbls} label files")
        total_imgs += imgs
        total_labels += lbls

    write_data_yaml(dst_root, class_map)
    print(f"\nDone. Total: {total_imgs} images, {total_labels} label files → {dst_root}")
    print(f"Use configs/data.yaml or point --data to: {(dst_root / 'data.yaml').as_posix()}")


if __name__ == "__main__":
    main()
