"""Post-processing del rilevamento: IoU, NMS e filtraggio dei box.

Opera su box in pixel assoluti nel formato ``(x1, y1, x2, y2)``. Ultralytics
applica già la NMS internamente; queste utility servono per un filtraggio
aggiuntivo e per i percorsi classici/manuali in cui controlliamo noi il
post-processing.
"""

from __future__ import annotations

import numpy as np


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-Union (IoU) di due box in (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_inside(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> bool:
    """True se il centro di ``inner`` cade all'interno di ``outer``."""
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
    """Non-maximum suppression (NMS) greedy. Restituisce gli indici dei box mantenuti.

    Parameters
    ----------
    boxes:
        Array ``(N, 4)`` di (x1, y1, x2, y2).
    scores:
        Array ``(N,)`` di punteggi di confidenza.
    """
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([iou_xyxy(tuple(boxes[i]), tuple(boxes[j])) for j in rest])
        order = rest[ious <= iou_threshold]
    return keep


def filter_detections(
    detections: list[dict],
    conf_threshold: float = 0.25,
    min_box_area: float = 0.0,
) -> list[dict]:
    """Applica una soglia sulla confidenza e scarta i box minuscoli.

    Ogni rilevamento è un dict con almeno le chiavi ``conf`` e ``xyxy``.
    ``min_box_area`` è in pixel² (0 disabilita il filtro sull'area).
    """
    out = []
    for d in detections:
        if d["conf"] < conf_threshold:
            continue
        x1, y1, x2, y2 = d["xyxy"]
        if (x2 - x1) * (y2 - y1) < min_box_area:
            continue
        out.append(d)
    return out
