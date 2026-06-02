"""Disegno degli overlay di rilevamento / compliance e dei grafici di EDA / risultati."""

from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Overlay di rilevamento e compliance (OpenCV, BGR)
# --------------------------------------------------------------------------- #
def draw_detections(image: np.ndarray, detections: list[dict], class_names: list[str]) -> np.ndarray:
    """Disegna i bounding box con nome della classe + confidenza."""
    import cv2

    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
        cls = d["cls"]
        name = class_names[cls] if cls < len(class_names) else str(cls)
        label = f"{name} {d.get('conf', 0):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 2)
        _put_label(out, label, x1, y1, (255, 180, 0))
    return out


def draw_compliance(image: np.ndarray, assessments) -> np.ndarray:
    """Disegna i box delle persone colorati per stato di sicurezza con una label per lavoratore."""
    import cv2

    out = image.copy()
    for a in assessments:
        x1, y1, x2, y2 = (int(v) for v in a.person_box)
        cv2.rectangle(out, (x1, y1), (x2, y2), a.color, 2)
        present = [k for k, v in a.ppe_present.items() if v]
        label = a.status.value + (f" [{','.join(present)}]" if present else "")
        _put_label(out, label, x1, y1, a.color)
    return out


def _put_label(image: np.ndarray, text: str, x: int, y: int, color) -> None:
    import cv2

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y0 = max(th + 4, y)
    cv2.rectangle(image, (x, y0 - th - 4), (x + tw + 2, y0), color, -1)
    cv2.putText(image, text, (x + 1, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Grafici di EDA / risultati (matplotlib)
# --------------------------------------------------------------------------- #
def plot_class_distribution(per_class_counts: dict[str, int], out_path: str | Path, title="Distribuzione delle classi"):
    import matplotlib.pyplot as plt

    names = list(per_class_counts.keys())
    counts = list(per_class_counts.values())
    order = np.argsort(counts)[::-1]
    names = [names[i] for i in order]
    counts = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, counts, color="#3b76af")
    ax.set_title(title)
    ax.set_ylabel("numero di oggetti")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    _save(fig, out_path)


def plot_bbox_size_distribution(bbox_wh: list[tuple[float, float]], out_path: str | Path):
    import matplotlib.pyplot as plt

    wh = np.asarray(bbox_wh) if bbox_wh else np.zeros((1, 2))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(wh[:, 0], wh[:, 1], s=4, alpha=0.2, color="#c0504d")
    ax.set_xlabel("larghezza normalizzata")
    ax.set_ylabel("altezza normalizzata")
    ax.set_title("Distribuzione delle dimensioni dei bounding box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, out_path)


def plot_confusion_matrix(cm, class_labels: list[str], out_path: str | Path, title="Confusion matrix"):
    # Nota: "Confusion matrix" è mantenuto come termine tecnico standard.
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_labels)), class_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(class_labels)), class_labels)
    ax.set_xlabel("predetto")
    ax.set_ylabel("reale")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    _save(fig, out_path)


def _save(fig, out_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
