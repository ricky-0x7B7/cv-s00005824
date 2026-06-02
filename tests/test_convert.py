"""Test per la conversione / suddivisione del dataset e il parsing delle etichette su dati sintetici."""

import numpy as np
import pytest

from src.convert import prepare_sh17, write_data_yaml
from src.data import BBox, read_yolo_label


def _make_fake_sh17(root, n_images=10):
    import cv2

    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(n_images):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i}.jpg"), img)
        (lbl_dir / f"img_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n10 0.3 0.3 0.1 0.1\n")


def test_read_yolo_label(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.2\n10 0.1 0.1 0.05 0.05\n")
    boxes = read_yolo_label(p)
    assert len(boxes) == 2
    assert boxes[0].cls == 0
    assert isinstance(boxes[1], BBox)


def test_bbox_to_xyxy():
    b = BBox(0, 0.5, 0.5, 0.5, 0.5)
    assert b.to_xyxy(100, 100) == (25, 25, 75, 75)


def test_read_missing_label_returns_empty(tmp_path):
    assert read_yolo_label(tmp_path / "missing.txt") == []


def test_prepare_sh17_random_split(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    _make_fake_sh17(raw, n_images=10)

    summary = prepare_sh17(raw, out, val_frac=0.2, test_frac=0.2, seed=42, copy=True)
    total = sum(summary["counts"].values())
    assert total == 10
    assert (out / "images" / "train").exists()
    assert (out / "labels" / "train").exists()
    assert (out / "data.yaml").exists()


def test_prepare_sh17_missing_raw_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_sh17(tmp_path / "nope", tmp_path / "out")


def test_data_yaml_has_17_classes(tmp_path):
    out = tmp_path / "processed"
    out.mkdir()
    import yaml

    path = write_data_yaml(out)
    cfg = yaml.safe_load(path.read_text())
    assert len(cfg["names"]) == 17
    assert cfg["names"][10] == "helmet"
