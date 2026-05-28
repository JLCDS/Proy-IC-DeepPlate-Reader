# DeepPlate-Reader

Deep Learning-based License Plate Recognition (LPR) system optimized for degraded images: blur, low resolution, poor lighting, and partial occlusion.

## Pipeline Overview

```
Input (image / video / camera)
        │
        ▼
┌─────────────────┐
│  Plate Detector  │  ← YOLOv8 fine-tuned on plate datasets
│  (YOLOv8)       │
└────────┬────────┘
         │ bounding boxes
         ▼
┌─────────────────┐
│  Image Enhancer  │  ← Upscaling · Denoising · CLAHE · Sharpening
└────────┬────────┘
         │ clean crop
         ▼
┌─────────────────┐
│   OCR Engine    │  ← EasyOCR / PaddleOCR + regex post-processing
└────────┬────────┘
         │ plate text + confidence
         ▼
     Output / API
```

## Project Structure

```
Proy-IC-DeepPlate-Reader/
├── configs/                 # YAML config for pipeline and training
│   ├── pipeline.yaml
│   └── data.yaml
├── data/
│   ├── raw/                 # Raw datasets (gitignored)
│   ├── processed/           # Processed annotations
│   └── samples/             # Sample images for quick tests
├── models/                  # Saved weights (gitignored)
├── notebooks/               # Exploration and analysis
├── scripts/
│   ├── infer.py             # Run inference on image/video/camera
│   ├── train.py             # Fine-tune detection model
│   └── evaluate.py          # Measure OCR accuracy
├── src/deepplate/
│   ├── detection/           # PlateDetector (YOLOv8 wrapper)
│   ├── enhancement/         # ImageEnhancer (classical CV)
│   ├── ocr/                 # PlateOCR (EasyOCR wrapper + postprocess)
│   ├── pipeline/            # LPRPipeline (orchestrates all stages)
│   └── utils/               # Logger, visualization helpers
└── tests/
    └── unit/                # Unit tests for each module
```

## Dataset

Source: [Proyecto Placas v1 — Roboflow](https://universe.roboflow.com/juan-felipe-orozco-cortes/proyecto-placas) (CC BY 4.0)

| Split | Images |
|-------|--------|
| train | 792    |
| val   | 75     |
| test  | 38     |

Raw format: **RetinaNet CSV** (`_annotations.csv` per split, no header).  
After conversion: **YOLO** format in `data/raw/plates/`.

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install project
pip install -e ".[dev]"
```

## Usage

### Inference on images
```bash
python scripts/infer.py --image data/samples/  --config configs/pipeline.yaml
```

### Live camera
```bash
python scripts/infer.py --camera 0
```

### Video file
```bash
python scripts/infer.py --video path/to/video.mp4 --output outputs/result.mp4
```

### Prepare dataset (run once)
```bash
# Converts RetinaNet CSV → YOLO format into data/raw/plates/
python scripts/convert_to_yolo.py
```

### Train detection model
```bash
python scripts/train.py --data data/raw/plates/data.yaml --epochs 50 --device 0
```

### Evaluate accuracy
```bash
python scripts/evaluate.py --labels data/processed/labels.csv
```

### Run tests
```bash
pytest
```

## Configuration

Edit `configs/pipeline.yaml` to adjust detection confidence, enhancement parameters, OCR engine, and input source.

## Tech Stack

| Component | Library |
|-----------|---------|
| Plate Detection | `ultralytics` YOLOv8 |
| Image Enhancement | `opencv-python` (CLAHE, denoising, sharpening) |
| OCR | `easyocr` |
| Config | `pydantic-settings` + YAML |
| Logging | `loguru` |
| Testing | `pytest` + `pytest-cov` |
