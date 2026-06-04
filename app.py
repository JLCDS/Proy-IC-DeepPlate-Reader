from __future__ import annotations
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepPlate-Reader",
    page_icon="🚘",
    layout="wide",
)

MODEL_PATH = ROOT / "models/ocr/ocr_svm.pkl"
LE_PATH    = ROOT / "models/ocr/ocr_le.pkl"


# ─── Carga de modelos (se cachea en toda la sesión) ───────────────────────────
@st.cache_resource(show_spinner="Cargando modelos...")
def load_pipeline():
    from deepplate.detection import PlateDetector
    from deepplate.enhancement import ImageEnhancer
    from deepplate.segmentation import CharacterSegmenter
    from deepplate.ocr import PlateOCR

    return {
        "detector": PlateDetector(),
        "enhancer": ImageEnhancer(),
        "segmenter": CharacterSegmenter(),
        "ocr": PlateOCR(MODEL_PATH, LE_PATH, min_chars=3, min_confidence=0.25),
    }


def bytes_to_bgr(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _vehicle_type(text: str) -> tuple[str, str]:
    import re
    t = text.upper().replace(" ", "")
    if re.match(r"[A-Z]{3}\d{2}[A-Z]$", t):
        return "Motocicleta", "Moto"
    if re.match(r"[A-Z]{3}\d{3}$", t):
        return "Vehiculo particular / servicio publico", "Carro"
    return "Formato no reconocido", "?"


# ─── Página 1: Demo en vivo ───────────────────────────────────────────────────
def page_demo(pipe: dict) -> None:
    st.title("Reconocimiento de Placas Vehiculares")
    st.markdown("Sube una imagen de un vehículo o placa para detectar y leer el número.")

    uploaded = st.file_uploader(
        "Imagen (JPG / PNG)", type=["jpg", "jpeg", "png"], key="upload_demo"
    )

    # ── Selector de imágenes de ejemplo ──────────────────────────────────────
    BASE = ROOT / "data/dataset/Proyecto Placas.v1-primera-version.retinanet"
    CURATED = [
        (BASE / "train/51_jpg.rf.3de2b93ad828f7e9798af166217e4666.jpg",
         "BJX 656 — Mitsubishi, placa amarilla frontal"),
        (BASE / "train/19_jpg.rf.1b9c46f026e67be468299b29f4963d2b.jpg",
         "MOX 296 — Renault, placa blanca frontal"),
        (BASE / "train/15_jpg.rf.27df7ca5d7b5b7b10ce43ded3fd64523.jpg",
         "MNN 465 — Chevrolet, placa blanca trasera"),
        (BASE / "train/114_jpg.rf.862fa0eb11110aa344d0becfafd509c8.jpg",
         "BXR 004 — Renault Logan, placa blanca trasera"),
        (BASE / "train/IMG_20210811_054925_jpg.rf.09f58d73c01dc4b4dffa454fd3e12292.jpg",
         "JPW 623 — Renault Kwid, placa amarilla trasera"),
        (BASE / "test/82_jpg.rf.463549de38faf7e0c31558c8d493d7a3.jpg",
         "EOS 709 — Taxi Hyundai  [lectura difícil]"),
    ]
    # Filtrar las que existen en disco
    options = [(p, lbl) for p, lbl in CURATED if p.exists()]

    with st.expander("Usar imagen de ejemplo del dataset", expanded=False):
        labels = [lbl for _, lbl in options]
        sel_label = st.radio("Elige una imagen:", labels, index=0, key="sample_radio")
        sel_path = options[labels.index(sel_label)][0]
        use_sample = st.button("Cargar imagen seleccionada")

    img = None
    if use_sample:
        img = cv2.imread(str(sel_path))
        st.info(f"Imagen cargada: {sel_path.name}")
    elif uploaded:
        img = bytes_to_bgr(uploaded.read())

    if img is None:
        st.info("Sube una imagen o usa el ejemplo para comenzar.")
        return

    # ── Ejecutar pipeline ──
    with st.spinner("Procesando..."):
        bboxes = pipe["detector"].detect(img)
        detections = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            crop = img[y1:y2, x1:x2]
            enhanced = pipe["enhancer"].enhance(crop)
            chars = pipe["segmenter"].segment(enhanced)
            text, conf = pipe["ocr"].read(chars)
            detections.append({
                "bbox": bbox,
                "enhanced": enhanced,
                "chars": chars,
                "text": text,
                "conf": conf,
            })

    # ── Imagen anotada ──
    vis = img.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = (0, 200, 0) if d["text"] else (0, 140, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = d["text"] if d["text"] else "?"
        cv2.putText(vis, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    col_orig, col_res = st.columns(2)
    with col_orig:
        st.subheader("Imagen original")
        st.image(bgr_to_rgb(img), use_container_width=True)
    with col_res:
        st.subheader(f"Detecciones ({len(bboxes)} región(es) encontrada(s))")
        st.image(bgr_to_rgb(vis), use_container_width=True)

    if not detections:
        st.warning("No se detectaron regiones de placa. Intenta con otra imagen.")
        return

    st.divider()
    st.subheader("Detalle por placa detectada")

    for i, d in enumerate(detections):
        label = d["text"] if d["text"] else "No reconocida"
        vtype, vtag = _vehicle_type(label) if d["text"] else ("Desconocido", "?")
        with st.expander(
            f"Placa #{i + 1} — {label}  |  [{vtag}] {vtype}  (confianza: {d['conf']:.0%})",
            expanded=True,
        ):
            c1, c2, c3 = st.columns([1, 1, 2])

            with c1:
                st.caption("Recorte mejorado")
                st.image(bgr_to_rgb(d["enhanced"]), use_container_width=True)
                if d["text"]:
                    st.metric("Tipo de vehiculo", vtype)
                    st.metric("Formato", "ABC12D (moto)" if vtag == "Moto" else "ABC123 (carro/bus)")

            with c2:
                st.caption(f"Caracteres segmentados ({len(d['chars'])})")
                if d["chars"]:
                    row = np.hstack([
                        cv2.resize(c, (48, 64)) for c in d["chars"]
                    ])
                    st.image(row, clamp=True, use_container_width=True)
                else:
                    st.write("—")

            # ── VISUALIZACIÓN 1: Confianza por carácter ──────────────────────
            with c3:
                st.caption("Confianza por carácter (viz. 1)")
                if d["chars"]:
                    ocr = pipe["ocr"]
                    feats = np.array([ocr._extract_hog(c) for c in d["chars"]])
                    probs = ocr.clf.predict_proba(feats)
                    pred_ids = probs.argmax(axis=1)
                    pred_chars = ocr.le.inverse_transform(pred_ids)
                    pred_confs = probs.max(axis=1)

                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    bars = ax.bar(range(len(pred_chars)), pred_confs,
                                  color=["#2ecc71" if c > 0.5 else "#e74c3c" for c in pred_confs])
                    ax.set_xticks(range(len(pred_chars)))
                    ax.set_xticklabels(pred_chars, fontsize=14, fontweight="bold")
                    ax.set_ylim(0, 1)
                    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
                    ax.set_ylabel("Confianza SVM")
                    ax.set_title("Predicción por posición")
                    ax.set_xlabel("Carácter predicho")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    n_conf = int(sum(c > 0.5 for c in pred_confs))
                    st.caption(
                        f"**Analisis:** {n_conf} de {len(pred_confs)} caracteres superan "
                        f"el umbral de confianza (0.5). Las barras verdes indican predicciones "
                        f"confiables; las rojas sugieren ambiguedad visual entre caracteres "
                        f"similares (ej. I/1, O/0, B/8)."
                    )


# ─── Página 2: Visualizaciones de datos ──────────────────────────────────────
def page_visualizaciones(pipe: dict) -> None:
    st.title("Visualizaciones del Dataset y Segmentación")

    ann_dir = ROOT / "data/dataset/Proyecto Placas.v1-primera-version.retinanet"
    test_ann = ann_dir / "test/_annotations.csv"

    # ── VISUALIZACIÓN 2: Distribución de tamaños de placa ────────────────────
    st.subheader("Viz. 2 — Distribución de tamaños de placa en el dataset")

    import csv
    splits = {"train": [], "test": [], "valid": []}
    for split in splits:
        csv_path = ann_dir / split / "_annotations.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for row in csv.reader(f):
                if len(row) == 6:
                    try:
                        w = int(row[3]) - int(row[1])
                        h = int(row[4]) - int(row[2])
                        splits[split].append((w, h))
                    except ValueError:
                        pass

    all_pairs = [(w, h, s) for s, pairs in splits.items() for w, h in pairs]
    if all_pairs:
        import plotly.graph_objects as go
        fig = go.Figure()
        colors = {"train": "#3498db", "test": "#e74c3c", "valid": "#2ecc71"}
        for split, pairs in splits.items():
            if pairs:
                ws, hs = zip(*pairs)
                fig.add_trace(go.Scatter(
                    x=ws, y=hs, mode="markers",
                    name=split.capitalize(),
                    marker=dict(color=colors[split], size=10, opacity=0.7),
                    text=[f"Aspecto: {w/h:.2f}" for w, h in pairs],
                ))
        fig.update_layout(
            xaxis_title="Ancho de placa (px)",
            yaxis_title="Alto de placa (px)",
            title="Dimensiones de bounding-box de placa por split",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        total = len(all_pairs)
        avg_w = np.mean([p[0] for p in all_pairs])
        avg_h = np.mean([p[1] for p in all_pairs])
        avg_asp = np.mean([p[0] / p[1] for p in all_pairs])
        st.markdown(
            f"**{total} anotaciones** | Ancho promedio: **{avg_w:.1f}px** | "
            f"Alto promedio: **{avg_h:.1f}px** | Relacion de aspecto media: **{avg_asp:.2f}**"
        )
        st.info(
            "**Analisis Viz. 2:** Las placas del dataset tienen un ancho entre 65 y 125 px "
            "y una relacion de aspecto promedio de {:.2f}, inferior al ratio teorico de la "
            "placa colombiana estandar (330x130 mm = 2.54:1). Esto se debe a que los "
            "bounding boxes incluyen margen alrededor de la placa y a la variacion angular "
            "de las tomas. Este analisis fue clave para calibrar el filtro de deteccion "
            "(min_aspect = 1.1) y evitar que el detector rechazara placas reales.".format(avg_asp)
        )
    else:
        st.warning("No se encontraron anotaciones.")

    st.divider()

    st.divider()

    # ── VISUALIZACIÓN 3: Grid de placas del dataset con segmentación ─────────
    st.subheader("Viz. 3 — Placas del dataset con caracteres segmentados")

    if not test_ann.exists():
        st.warning("No se encontraron anotaciones de test.")
        return

    test_img_dir = ann_dir / "test"
    with open(test_ann) as f:
        test_rows = [r for r in csv.reader(f) if len(r) == 6]

    n_show = st.slider("Cuántas placas mostrar", 3, min(12, len(test_rows)), 6)

    cols = st.columns(3)
    for idx, (fname, xmin, ymin, xmax, ymax, _) in enumerate(test_rows[:n_show]):
        img_path = test_img_dir / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        x1, y1, x2, y2 = int(xmin), int(ymin), int(xmax), int(ymax)
        crop = img[y1:y2, x1:x2]
        enhanced = pipe["enhancer"].enhance(crop)
        chars = pipe["segmenter"].segment(enhanced)
        text, conf = pipe["ocr"].read(chars)

        col = cols[idx % 3]
        vt, vk = _vehicle_type(text) if text else ("?", "?")
        with col:
            st.image(bgr_to_rgb(enhanced),
                     caption=f'OCR: "{text}" | {vk} | chars={len(chars)}',
                     use_container_width=True)
            if chars:
                strip = np.hstack([cv2.resize(c, (32, 48)) for c in chars])
                st.image(strip, clamp=True, use_container_width=True)

    st.info(
        "**Analisis Viz. 3:** Cada tarjeta muestra el recorte de placa mejorado (CLAHE + "
        "denoise) y la tira de caracteres segmentados por el modulo Otsu + componentes "
        "conectados. El texto OCR y el tipo de vehiculo se infieren del formato: "
        "ABC123 = carro/servicio publico, ABC12D = motocicleta. Las placas con fondo "
        "amarillo pertenecen a vehiculos de servicio publico o motocicletas."
    )


# ─── Página 3: Métricas del clasificador SVM ─────────────────────────────────
def page_metricas() -> None:
    st.title("Métricas del Clasificador HOG + SVM")
    st.markdown(
        "El clasificador se entrenó **desde cero** con datos sintéticos "
        "(caracteres renderizados con fuentes del sistema + aumentación)."
    )

    if not MODEL_PATH.exists():
        st.error("Modelo no encontrado. Ejecuta: `python scripts/train.py`")
        return

    import joblib
    from skimage.feature import hog
    from sklearn.metrics import confusion_matrix, classification_report
    from deepplate.ocr.trainer import generate_dataset, CHARS
    from deepplate.ocr.reader import _prepare_char

    with st.spinner("Generando conjunto de prueba sintético (puede tardar ~20s)..."):
        X_test, y_raw = generate_dataset(samples_per_class=40)
        clf = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)
        y_test = le.transform(y_raw)
        y_pred = clf.predict(X_test)

    accuracy = (y_pred == y_test).mean()
    st.metric("Accuracy en test sintético", f"{accuracy:.2%}", delta=None)

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    # ── VISUALIZACIÓN 4: Accuracy por clase ──────────────────────────────────
    st.subheader("Viz. 4 — Precisión (F1) por clase de carácter")
    classes = list(le.classes_)
    f1_scores = [report[c]["f1-score"] for c in classes]

    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=classes,
        y=f1_scores,
        marker_color=["#2ecc71" if v >= 0.9 else "#f39c12" if v >= 0.7 else "#e74c3c"
                      for v in f1_scores],
        text=[f"{v:.2f}" for v in f1_scores],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Carácter",
        yaxis_title="F1-Score",
        yaxis_range=[0, 1.1],
        title="F1-Score por clase (verde >= 0.9, naranja >= 0.7, rojo < 0.7)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    low = [c for c, v in zip(classes, f1_scores) if v < 0.7]
    st.info(
        f"**Analisis Viz. 4:** El clasificador alcanza F1 >= 0.9 en la mayoria de clases. "
        f"Las clases con menor rendimiento son: **{', '.join(low) if low else 'ninguna'}**. "
        f"Estas confusiones son esperadas por similitud visual: I/1, O/0, B/8, S/5, Z/2. "
        f"El codigo de colores (verde/naranja/rojo) permite identificar rapidamente "
        f"que caracteres necesitarian mas datos de entrenamiento."
    )

    st.divider()

    # ── Matriz de confusión ────────────────────────────────────────────────────
    st.subheader("Viz. 5 — Matriz de confusion normalizada (36 clases A-Z + 0-9)")
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig2, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión normalizada")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.info(
        "**Analisis Viz. 5:** La diagonal principal concentra los valores mas altos "
        "(cercanos a 1.0), lo que indica que el clasificador predice correctamente la "
        "gran mayoria de caracteres. Las confusiones mas frecuentes aparecen en pares "
        "visualmente similares: I/1 (trazo vertical), O/0 (circulo cerrado), B/8 "
        "(dos curvas superpuestas) y S/5 (curva en S). Estos errores son inherentes "
        "al reconocimiento optico de caracteres y se reducen con mayor variedad en "
        "los datos de entrenamiento."
    )


# ─── Página 4: Acerca del sistema ─────────────────────────────────────────────
def page_acerca() -> None:
    st.title("Acerca de DeepPlate-Reader")
    st.markdown("""
## Pipeline de reconocimiento de placas vehiculares

El sistema implementa un pipeline completo de **Reconocimiento de Placas Vehiculares (LPR)**
usando únicamente técnicas clásicas de visión por computador y aprendizaje automático.

### Etapas del pipeline

| Etapa | Técnica | Librería |
|-------|---------|----------|
| Detección de placa | Filtro bilateral → Canny → Contornos → NMS | OpenCV |
| Mejora de imagen | CLAHE + Desnoise + Sharpening | OpenCV |
| Segmentación de caracteres | Otsu binarize → Componentes conectados | OpenCV |
| Reconocimiento OCR | HOG features + SVM (RBF kernel) | scikit-learn |

### Entrenamiento del clasificador
- **Entrenado desde cero**: el SVM se construye y entrena completamente en este proyecto
- **Datos sintéticos**: ~14 400 imágenes generadas con PIL (fuentes del sistema + aumentación)
- **Aumentación**: ruido gaussiano, rotación ±12°, blur, simulación de baja resolución, artefactos JPEG
- **36 clases**: A-Z (26 letras) + 0-9 (10 dígitos)
- **Accuracy en test sintético**: ~96-99%

### Formato de placas objetivo
Colombia — placas de vehículos: **ABC123** (3 letras + 3 dígitos)
    """)


# ─── Navegación ───────────────────────────────────────────────────────────────
def main() -> None:
    models_ready = MODEL_PATH.exists() and LE_PATH.exists()

    with st.sidebar:
        st.title("DeepPlate-Reader")
        st.caption("Inteligencia Computacional — 2026-I")
        st.divider()
        page = st.radio(
            "Navegacion",
            ["Demo en vivo", "Visualizaciones de datos", "Metricas del clasificador", "Acerca del sistema"],
            index=0,
        )
        st.divider()
        if models_ready:
            st.success("Modelo SVM cargado")
        else:
            st.error("Modelo no encontrado")
            st.code("python scripts/train.py")

    if not models_ready:
        st.error("Primero entrena el modelo: `python scripts/train.py`")
        st.stop()

    pipe = load_pipeline()

    if page == "Demo en vivo":
        page_demo(pipe)
    elif page == "Visualizaciones de datos":
        page_visualizaciones(pipe)
    elif page == "Metricas del clasificador":
        page_metricas()
    else:
        page_acerca()


if __name__ == "__main__":
    main()
