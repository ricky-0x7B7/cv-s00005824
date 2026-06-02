"""Valutazione YOLO: esegue la validazione, esporta le metriche (JSON/CSV) e raccoglie i grafici.

Ultralytics calcola precision/recall/mAP e genera la confusion matrix e le curve
PR/F1 durante ``model.val()``; questo modulo lo incapsula, serializza i numeri
principali e copia le figure generate in ``reports/``.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

from src.config import select_device


def evaluate_yolo(
    weights: str | Path,
    data: str,
    device: str = "auto",
    imgsz: int = 640,
    split: str = "val",
    metrics_dir: str | Path = "reports/metrics",
    figures_dir: str | Path = "reports/figures",
) -> dict:
    """Valuta i pesi addestrati e salva metriche + figure.

    Restituisce il dict con le metriche principali.
    """
    from ultralytics import YOLO

    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    resolved_device = select_device(device)
    results = model.val(data=data, imgsz=imgsz, split=split, device=resolved_device)

    box = results.box
    summary = {
        "weights": str(weights),
        "data": data,
        "split": split,
        "device": resolved_device,
        "precision_mean": float(box.mp),
        "recall_mean": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class_mAP50": {
            model.names[int(c)]: float(ap)
            for c, ap in zip(box.ap_class_index, box.ap50)
        },
    }
    summary["inference_time_ms"] = _measure_inference_time(model, results, resolved_device)

    # Serializza le metriche.
    (metrics_dir / "yolo_metrics.json").write_text(json.dumps(summary, indent=2))
    _write_csv(metrics_dir / "yolo_results.csv", summary)

    # Copia in reports/ le figure generate (confusion matrix, curve PR / F1).
    _collect_figures(Path(results.save_dir), figures_dir)
    return summary


def _measure_inference_time(model, results, device: str, n_warmup: int = 3) -> float:
    """Tempo medio di inferenza (ms) best-effort riportato dal dict speed di Ultralytics."""
    try:
        speed = results.speed  # {'preprocess', 'inference', 'postprocess'} in ms
        return float(speed.get("inference", 0.0))
    except Exception:  # pragma: no cover - dipende dalla versione di ultralytics
        return -1.0


def _write_csv(path: Path, summary: dict) -> None:
    flat = {k: v for k, v in summary.items() if not isinstance(v, dict)}
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in flat.items():
            writer.writerow([k, v])
        for cls, ap in summary.get("per_class_mAP50", {}).items():
            writer.writerow([f"mAP50/{cls}", ap])


def _collect_figures(run_dir: Path, figures_dir: Path) -> None:
    # Ultralytics 8.4 antepone "Box" alle curve di rilevamento (BoxPR_curve.png, ...);
    # le versioni più vecchie usavano PR_curve.png. Mappiamo entrambi così le figure
    # vengono sempre raccolte.
    wanted = {
        "confusion_matrix.png": "confusion_matrix.png",
        "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",
        "BoxPR_curve.png": "pr_curve.png",
        "PR_curve.png": "pr_curve.png",
        "BoxF1_curve.png": "f1_curve.png",
        "F1_curve.png": "f1_curve.png",
        "BoxP_curve.png": "p_curve.png",
        "BoxR_curve.png": "r_curve.png",
        "results.png": "yolo_training_curves.png",
    }
    if not run_dir.exists():
        return
    for src_name, dst_name in wanted.items():
        src = run_dir / src_name
        if src.exists():
            shutil.copy2(src, figures_dir / dst_name)
