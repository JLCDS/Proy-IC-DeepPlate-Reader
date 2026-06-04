# DeepPlate-Reader

Sistema de Reconocimiento Automático de Placas Vehiculares Colombianas (LPR) construido enteramente desde cero usando visión por computador clásica y aprendizaje automático, sin modelos preentrenados.

**Curso:** Inteligencia Computacional — Semestre 2026-I  
**Formato de placa objetivo:** `ABC123` (3 letras + 3 dígitos, Colombia)

---

## Pipeline

```
Imagen de entrada
      │
      ▼
┌─────────────────────────────────────┐
│  Detección de placa                 │  Filtro bilateral → Canny → Contornos → NMS
└──────────────────┬──────────────────┘
                   │ bbox (x1,y1,x2,y2)
                   ▼
┌─────────────────────────────────────┐
│  Mejora de imagen                   │  Upscale x2 · CLAHE · Denoise · Sharpening
└──────────────────┬──────────────────┘
                   │ crop mejorado
                   ▼
┌─────────────────────────────────────┐
│  Segmentacion de caracteres         │  Otsu binarize → Componentes conectados
└──────────────────┬──────────────────┘
                   │ lista de crops por caracter
                   ▼
┌─────────────────────────────────────┐
│  Clasificacion HOG + SVM            │  324 features · RBF kernel · 36 clases
└──────────────────┬──────────────────┘
                   │ texto + confianza
                   ▼
              Resultado
```

---

## Estructura del proyecto

```
Proy-IC-DeepPlate-Reader/
├── app.py                     # Interfaz web Streamlit (punto de entrada)
├── configs/
│   └── pipeline.yaml          # Parametros de deteccion, mejora y OCR
├── data/
│   └── dataset/               # Dataset anotado (RetinaNet CSV)
├── models/                    # Modelos entrenados (generados localmente, no en git)
│   └── ocr/
│       ├── ocr_svm.pkl        # <- generado por scripts/train.py
│       └── ocr_le.pkl         # <- generado por scripts/train.py
├── notebooks/                 # Analisis exploratorio del dataset
├── outputs/                   # Resultados generados (no en git)
├── scripts/
│   ├── train.py               # Entrena el SVM desde cero (datos sinteticos)
│   ├── infer.py               # Inferencia por linea de comandos
│   ├── evaluate.py            # Metricas sobre un CSV etiquetado
│   ├── generar_figuras.py     # Genera figuras para el informe
│   └── diagnostico.py        # Diagnostico paso a paso del pipeline
├── src/deepplate/
│   ├── detection/             # PlateDetector (Canny + contornos)
│   ├── enhancement/           # ImageEnhancer (CLAHE + denoise)
│   ├── segmentation/          # CharacterSegmenter (Otsu + CC)
│   ├── ocr/                   # PlateOCR (HOG + SVM) + trainer
│   ├── pipeline/              # LPRPipeline (orquesta todas las etapas)
│   └── utils/                 # Logger, visualizacion
└── tests/
    └── unit/
```

---

## Ejecucion rapida (Windows)

Doble clic en **`launch.bat`** — instala todo, entrena el modelo y abre la app automaticamente.

> **Requisito:** tener [Python 3.10+](https://www.python.org/downloads/) instalado con la opcion **"Add Python to PATH"** marcada.

---

## Instalacion manual

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd Proy-IC-DeepPlate-Reader

# 2. Instalar dependencias
pip install -r requirements.txt
```

> **Requiere Python >= 3.10**

---

## Uso

### 1. Entrenar el clasificador (obligatorio la primera vez)

```bash
python scripts/train.py --samples 500
```

Genera `models/ocr/ocr_svm.pkl` y `models/ocr/ocr_le.pkl`.  
Duracion: ~30 segundos en CPU. Accuracy esperada: ~97%.

### 2. Lanzar la interfaz web

```bash
streamlit run app.py
```

Abre `http://localhost:8501`. La app tiene 4 secciones:

| Seccion | Descripcion |
|---------|-------------|
| Demo en vivo | Sube una imagen y ve el resultado del pipeline |
| Visualizaciones de datos | Scatter de dimensiones del dataset + grid de placas segmentadas |
| Metricas del clasificador | Matriz de confusion + F1-score por clase |
| Acerca del sistema | Descripcion del pipeline |

### 3. Inferencia por linea de comandos

```bash
# Imagen unica
python scripts/infer.py --image ruta/imagen.jpg

# Carpeta de imagenes
python scripts/infer.py --image data/dataset/.../test/

# Camara en vivo
python scripts/infer.py --camera 0

# Video
python scripts/infer.py --video ruta/video.mp4 --output outputs/resultado.mp4
```

### 4. Diagnostico del pipeline (paso a paso)

```bash
python scripts/diagnostico.py                         # imagen por defecto
python scripts/diagnostico.py --image mi_placa.jpg   # imagen propia
python scripts/diagnostico.py --skip-train           # si ya entrenaste
```

Guarda imagenes intermedias en `outputs/diagnostico/` para inspeccionar cada etapa.

### 5. Generar figuras del informe

```bash
python scripts/generar_figuras.py
# Guarda en outputs/figuras_informe/:
#   figura1_pipeline.png
#   figura2_confusion_matrix.png
```

---

## Dataset

**Fuente:** Proyecto Placas v1 — Roboflow (CC BY 4.0)  
**Formato:** RetinaNet CSV (`_annotations.csv` por split)

| Split      | Imagenes | Anotaciones |
|------------|----------|-------------|
| train      | 40       | 40          |
| validation | 8        | 8           |
| test       | 21       | 21          |

Imagenes tomadas con camara de telefono (Redmi Note 8) en condiciones reales de calle en Colombia.

---

## Tecnologias

| Componente              | Libreria              |
|-------------------------|-----------------------|
| Deteccion de placa      | `opencv-python`       |
| Segmentacion caracteres | `opencv-python`       |
| Extraccion HOG          | `scikit-image`        |
| Clasificador SVM        | `scikit-learn`        |
| Persistencia del modelo | `joblib`              |
| Generacion datos train  | `Pillow` (PIL)        |
| Interfaz web            | `streamlit`           |
| Graficas interactivas   | `plotly`, `matplotlib`|

---

## Configuracion

Edita `configs/pipeline.yaml` para ajustar los parametros:

```yaml
detection:
  min_aspect: 1.1        # ratio minimo ancho/alto de la placa
  canny_low: 50
  canny_high: 200

enhancement:
  upscale_factor: 2
  contrast_clip_limit: 2.0

segmentation:
  min_char_height_ratio: 0.25

ocr:
  model_path: models/ocr/ocr_svm.pkl
  min_confidence: 0.25
```
