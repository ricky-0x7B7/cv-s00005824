"""Utility di inferenza: esegue YOLO su un'immagine e normalizza l'output.

I rilevamenti sono rappresentati come semplici dict, così il resto della pipeline
(postprocess, compliance, visualization) resta disaccoppiato da Ultralytics::

    {"cls": int, "conf": float, "xyxy": (x1, y1, x2, y2)}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import select_device


def load_model(weights: str | Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


def results_to_detections(result) -> list[dict]:
    """Converte un singolo Result di Ultralytics in una lista di dict di rilevamento."""
    detections: list[dict] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    for box, conf, cls in zip(xyxy, confs, clss):
        detections.append(
            {"cls": int(cls), "conf": float(conf), "xyxy": tuple(float(v) for v in box)}
        )
    return detections


def predict_image(
    model,
    image: "np.ndarray | str | Path",
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    device: str = "auto",
) -> list[dict]:
    """Esegue il rilevamento su un'immagine (path o array BGR) -> dict di rilevamento."""
    resolved_device = select_device(device)
    results = model.predict(
        source=image, conf=conf, iou=iou, imgsz=imgsz, device=resolved_device, verbose=False
    )
    return results_to_detections(results[0])
