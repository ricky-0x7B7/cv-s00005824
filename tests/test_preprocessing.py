"""Test per le funzioni di preprocessing e l'estrazione delle feature HOG."""

import numpy as np

from src.features import HOGConfig, extract_hog, extract_hog_batch
from src.preprocessing import letterbox, normalize


def test_letterbox_is_square():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = letterbox(img, new_size=640)
    assert out.shape == (640, 640, 3)


def test_normalize_range():
    img = (np.ones((4, 4, 3)) * 255).astype(np.uint8)
    out = normalize(img)
    assert out.dtype == np.float32
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_hog_feature_is_1d_and_deterministic():
    img = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    f1 = extract_hog(img, HOGConfig())
    f2 = extract_hog(img, HOGConfig())
    assert f1.ndim == 1
    assert np.allclose(f1, f2)


def test_hog_batch_shape():
    imgs = np.random.default_rng(0).integers(0, 255, (5, 128, 128, 3), dtype=np.uint8)
    feats = extract_hog_batch(imgs, HOGConfig())
    assert feats.shape[0] == 5
    assert feats.ndim == 2
