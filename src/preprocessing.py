"""Preprocessing e data augmentation.

Due ruoli:
1. Documentare/visualizzare la strategia di augmentation (notebook 02). YOLO
   applica la propria augmentation in fase di addestramento (configurata in
   configs/train_yolo.yaml); la pipeline di Albumentations qui rispecchia quella
   strategia a scopo esplicativo e per la baseline classica.
2. Fornire il ridimensionamento letterbox deterministico usato dai percorsi di
   inferenza manuale.

Nota di dominio: le rotazioni sono mantenute piccole e i flip verticali sono
disabilitati — i lavoratori e le attrezzature in un cantiere sono in posizione
verticale.
"""

from __future__ import annotations

import numpy as np


def build_augmentation_pipeline(image_size: int = 640, train: bool = True):
    """Restituisce una trasformazione Albumentations adatta alle immagini di cantiere.

    Ricade su un ridimensionamento quasi privo di effetti se Albumentations non è
    disponibile, in modo che il resto del codice continui a funzionare.
    """
    try:
        import albumentations as A
    except ImportError:
        return None

    if not train:
        return A.Compose([A.LongestMaxSize(max_size=image_size)])

    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.1), rotate=(-7, 7), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def letterbox(image: np.ndarray, new_size: int = 640, color=(114, 114, 114)) -> np.ndarray:
    """Ridimensiona mantenendo le proporzioni e applica padding a un quadrato ``new_size`` (stile YOLO)."""
    import cv2

    h, w = image.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    top, left = (new_size - nh) // 2, (new_size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def normalize(image: np.ndarray) -> np.ndarray:
    """Scala un'immagine uint8 in float32 nell'intervallo [0, 1]."""
    return image.astype(np.float32) / 255.0
