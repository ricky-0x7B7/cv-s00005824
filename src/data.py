"""I/O del dataset, auditing ed estrazione dei crop per il dataset SH17.

Formato delle label YOLO (una riga per oggetto, normalizzato in [0, 1])::

    <class_id> <x_center> <y_center> <width> <height>
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class BBox:
    """Un singolo bounding box YOLO normalizzato."""

    cls: int
    xc: float
    yc: float
    w: float
    h: float

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Converte in angoli assoluti in pixel (x1, y1, x2, y2)."""
        x1 = (self.xc - self.w / 2) * img_w
        y1 = (self.yc - self.h / 2) * img_h
        x2 = (self.xc + self.w / 2) * img_w
        y2 = (self.yc + self.h / 2) * img_h
        return (
            max(0, int(round(x1))),
            max(0, int(round(y1))),
            min(img_w, int(round(x2))),
            min(img_h, int(round(y2))),
        )


def read_yolo_label(path: str | Path) -> list[BBox]:
    """Effettua il parsing di un file di label YOLO ``.txt``. File mancante -> lista vuota."""
    path = Path(path)
    if not path.exists():
        return []
    boxes: list[BBox] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        boxes.append(BBox(int(float(cls)), float(xc), float(yc), float(w), float(h)))
    return boxes


def label_path_for_image(image_path: Path) -> Path:
    """Mappa ``.../images/foo.jpg`` su ``.../labels/foo.txt`` (convenzione YOLO)."""
    parts = list(image_path.parts)
    # Sostituisce l'ultimo segmento 'images' con 'labels'.
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def list_images(images_dir: str | Path) -> list[Path]:
    """Elenca ricorsivamente i file immagine all'interno di una directory."""
    images_dir = Path(images_dir)
    return sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )


def iter_image_label_pairs(images_dir: str | Path):
    """Restituisce coppie ``(image_path, [BBox, ...])``."""
    for img in list_images(images_dir):
        yield img, read_yolo_label(label_path_for_image(img))


def audit_dataset(images_dir: str | Path, class_names: list[str]) -> dict:
    """Calcola le statistiche del dataset per l'analisi esplorativa dei dati.

    Restituisce un dict con i totali, il conteggio degli oggetti per classe, gli
    oggetti per immagine, la distribuzione delle dimensioni dei bounding box e
    l'elenco delle immagini illeggibili.
    """
    import cv2

    images = list_images(images_dir)
    class_counts: Counter[int] = Counter()
    objects_per_image: list[int] = []
    bbox_areas: list[float] = []          # normalizzato (w*h)
    bbox_wh: list[tuple[float, float]] = []
    corrupt: list[str] = []
    n_with_labels = 0

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            corrupt.append(str(img_path))
            continue
        boxes = read_yolo_label(label_path_for_image(img_path))
        objects_per_image.append(len(boxes))
        if boxes:
            n_with_labels += 1
        for b in boxes:
            class_counts[b.cls] += 1
            bbox_areas.append(b.w * b.h)
            bbox_wh.append((b.w, b.h))

    total_objects = sum(class_counts.values())
    per_class = {
        class_names[c] if c < len(class_names) else str(c): n
        for c, n in sorted(class_counts.items())
    }
    areas = np.array(bbox_areas) if bbox_areas else np.array([0.0])

    return {
        "n_images": len(images),
        "n_images_with_labels": n_with_labels,
        "n_corrupt": len(corrupt),
        "corrupt_files": corrupt,
        "n_objects": total_objects,
        "n_classes_present": len(class_counts),
        "objects_per_image_mean": float(np.mean(objects_per_image or [0])),
        "objects_per_image_max": int(np.max(objects_per_image or [0])),
        "per_class_counts": per_class,
        "bbox_area_norm_mean": float(areas.mean()),
        "bbox_area_norm_median": float(np.median(areas)),
        # Frazione di oggetti "piccoli" (< 1% dell'area dell'immagine) — guida
        # l'analisi degli errori sugli oggetti piccoli.
        "small_object_fraction": float((areas < 0.01).mean()),
        "_bbox_wh": bbox_wh,            # dati grezzi per il plotting
        "_areas": areas,
    }


def extract_crops_for_binary(
    images_dir: str | Path,
    positive_class: int,
    negative_class: int,
    crop_size: tuple[int, int] = (128, 128),
    helmet_iou_exclude: float = 0.2,
    max_per_class: int = 4000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Costruisce un dataset binario bilanciato di crop per la baseline HOG+SVM.

    I positivi sono crop dei box di ``positive_class`` (es. casco). I negativi sono
    crop dei box di ``negative_class`` (es. testa) che non si sovrappongono a un box
    positivo oltre ``helmet_iou_exclude`` (una testa scoperta). I crop vengono
    ridimensionati a ``crop_size`` e restituiti come array uint8 ``(N, H, W, 3)`` più
    le label.
    """
    import cv2

    from src.postprocess import iou_xyxy

    rng = np.random.default_rng(seed)
    pos_crops: list[np.ndarray] = []
    neg_crops: list[np.ndarray] = []

    for img_path, boxes in iter_image_label_pairs(images_dir):
        if not boxes:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        pos_xyxy = [b.to_xyxy(w, h) for b in boxes if b.cls == positive_class]
        neg_boxes = [b for b in boxes if b.cls == negative_class]

        for box in pos_xyxy:
            crop = _safe_crop(image, box, crop_size)
            if crop is not None:
                pos_crops.append(crop)

        for b in neg_boxes:
            nb = b.to_xyxy(w, h)
            # Salta le teste che si sovrappongono a un casco (testa ambigua / coperta).
            if any(iou_xyxy(nb, pb) > helmet_iou_exclude for pb in pos_xyxy):
                continue
            crop = _safe_crop(image, nb, crop_size)
            if crop is not None:
                neg_crops.append(crop)

    pos_crops = _subsample(pos_crops, max_per_class, rng)
    neg_crops = _subsample(neg_crops, max_per_class, rng)

    X = np.array(pos_crops + neg_crops, dtype=np.uint8)
    y = np.array([1] * len(pos_crops) + [0] * len(neg_crops), dtype=np.int64)
    return X, y


def _safe_crop(image: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]):
    import cv2

    x1, y1, x2, y2 = box
    if x2 - x1 < 8 or y2 - y1 < 8:        # scarta i box degeneri / minuscoli
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def _subsample(items: list, k: int, rng: np.random.Generator) -> list:
    if len(items) <= k:
        return items
    idx = rng.choice(len(items), size=k, replace=False)
    return [items[i] for i in idx]
