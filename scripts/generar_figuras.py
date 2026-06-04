import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path(__file__).parent.parent / "outputs" / "figuras_informe"
OUT.mkdir(parents=True, exist_ok=True)


# ─── FIGURA 1: Diagrama de bloques del pipeline ───────────────────────────────

def figura1_pipeline():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    bloques = [
        (0.2,  "Imagen\nde entrada",               "#4A90D9"),
        (2.6,  "Deteccion\nde placa\nCanny + NMS",  "#E67E22"),
        (5.0,  "Mejora\nde imagen\nCLAHE + Denoise","#27AE60"),
        (7.4,  "Segmentacion\ncaracteres\nOtsu + CC","#8E44AD"),
        (9.8,  "HOG features\n(324 dims)\n8x8 celdas","#D35400"),
        (12.2, "SVM\nRBF kernel\n36 clases",         "#C0392B"),
        (14.6, "Texto\nde placa\nABC123",             "#2C3E50"),
    ]

    bw, bh, by = 2.1, 2.6, 1.2

    for x, label, color in bloques:
        rect = mpatches.FancyBboxPatch(
            (x, by), bw, bh,
            boxstyle="round,pad=0.12",
            linewidth=2, edgecolor="white", facecolor=color, alpha=0.93,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            x + bw / 2, by + bh / 2, label,
            ha="center", va="center", fontsize=9.5,
            color="white", fontweight="bold", linespacing=1.55, zorder=4,
        )

    xs_end   = [x + bw for x, _, _ in bloques[:-1]]
    xs_start = [x      for x, _, _ in bloques[1:]]
    etiquetas = ["BGR\n416 px", "bbox\n(x1,y1,\nx2,y2)", "crop\nRGB",
                 "chars[ ]", "feat\n324d", "clase\n+ conf"]

    for x0, x1, lbl in zip(xs_end, xs_start, etiquetas):
        mid = (x0 + x1) / 2
        ax.annotate(
            "", xy=(x1 + 0.02, by + bh / 2), xytext=(x0 - 0.02, by + bh / 2),
            arrowprops=dict(arrowstyle="->", color="#555", lw=2.4,
                            mutation_scale=20), zorder=2,
        )
        ax.text(mid, by + bh + 0.3, lbl, ha="center", va="bottom",
                fontsize=7.5, color="#444", style="italic")

    ax.set_title(
        "Fig. 1.  Pipeline completo de DeepPlate-Reader",
        fontsize=13, fontweight="bold", pad=16, color="#1A1A2E",
    )
    plt.tight_layout(pad=0.4)
    path = OUT / "figura1_pipeline.png"
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="#F8F9FA")
    plt.close()
    print(f"Figura 1 guardada -> {path}")


# ─── FIGURA 2: Matriz de confusion normalizada ────────────────────────────────

def figura2_confusion():
    import joblib
    from sklearn.metrics import confusion_matrix
    from deepplate.ocr.trainer import generate_dataset, CHARS

    MODEL  = Path(__file__).parent.parent / "models/ocr/ocr_svm.pkl"
    LE_P   = Path(__file__).parent.parent / "models/ocr/ocr_le.pkl"

    if not MODEL.exists():
        print("Modelo no encontrado. Ejecuta train.py primero.")
        return

    clf = joblib.load(MODEL)
    le  = joblib.load(LE_P)

    print("Generando conjunto de prueba sintetico (puede tardar ~20s)...")
    X_test, y_raw = generate_dataset(samples_per_class=50)
    y_test = le.transform(y_raw)
    y_pred = clf.predict(X_test)

    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    classes = list(le.classes_)
    acc     = (y_pred == y_test).mean()
    print(f"Accuracy en test: {acc:.2%}")

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8.5, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Clase predicha", fontsize=11, labelpad=8)
    ax.set_ylabel("Clase real",     fontsize=11, labelpad=8)
    ax.set_title(
        f"Fig. 2.  Matriz de confusion normalizada — SVM HOG  "
        f"(Accuracy = {acc:.2%})",
        fontsize=12, fontweight="bold", pad=14,
    )

    # Anotaciones numéricas solo en celdas relevantes (>= 0.05)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if v > 0.6 else "#333")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proporcion de predicciones", fontsize=10)

    plt.tight_layout()
    path = OUT / "figura2_confusion_matrix.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figura 2 guardada -> {path}")


if __name__ == "__main__":
    figura1_pipeline()
    figura2_confusion()
    print(f"\nAmbas figuras en: {OUT}")
