"""Estrazione di feature artigianali (HOG) per la baseline classica.

HOG (Histogram of Oriented Gradients) cattura la struttura locale di bordi e
gradienti ed è un descrittore classico per i task di riconoscimento basati sulla
forma, come rilevare la presenza di un casco su un crop di testa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HOGConfig:
    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (8, 8)
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = True
    grayscale: bool = True


def extract_hog(image: np.ndarray, config: HOGConfig | None = None) -> np.ndarray:
    """Estrae un vettore di feature HOG da una singola immagine (H, W, 3) o (H, W)."""
    from skimage.color import rgb2gray
    from skimage.feature import hog

    cfg = config or HOGConfig()
    if cfg.grayscale and image.ndim == 3:
        # cv2 carica in BGR; la pesatura di rgb2gray va bene qui come proxy della luminanza.
        image = rgb2gray(image)
        channel_axis = None
    else:
        channel_axis = None if image.ndim == 2 else -1

    return hog(
        image,
        orientations=cfg.orientations,
        pixels_per_cell=cfg.pixels_per_cell,
        cells_per_block=cfg.cells_per_block,
        block_norm=cfg.block_norm,
        transform_sqrt=cfg.transform_sqrt,
        channel_axis=channel_axis,
        feature_vector=True,
    )


def extract_hog_batch(images: np.ndarray, config: HOGConfig | None = None) -> np.ndarray:
    """Estrae le feature HOG per un batch ``(N, H, W, 3)`` -> ``(N, D)``."""
    cfg = config or HOGConfig()
    feats = [extract_hog(img, cfg) for img in images]
    return np.asarray(feats, dtype=np.float32)
