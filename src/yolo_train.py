"""Wrapper leggero attorno all'addestramento YOLO di Ultralytics.

Mantiene gli import di Ultralytics lazy, così il resto della libreria (audit,
baseline, compliance) resta utilizzabile senza che la dipendenza pesante sia
installata.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.config import PROJECT_ROOT, select_device


def load_train_config(path: str | Path) -> dict:
    """Carica configs/train_yolo.yaml in un dict."""
    return yaml.safe_load(Path(path).read_text())


def train_yolo(
    data: str,
    model: str = "yolov8n.pt",
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    project: str = "runs/ppe_detection",
    name: str = "yolo_sh17",
    **overrides,
):
    """Esegue il fine-tuning di un modello YOLO preaddestrato sul dataset dei DPI.

    Argomenti aggiuntivi di Ultralytics (augmentation, patience, lr0, ...) possono
    essere passati tramite ``**overrides`` — tipicamente il contenuto di
    configs/train_yolo.yaml. Restituisce l'oggetto results di Ultralytics.
    """
    from ultralytics import YOLO

    resolved_device = select_device(device)
    # Usa una directory di progetto assoluta così Ultralytics scrive in
    # <repo>/runs/... invece di annidarsi sotto la sua impostazione globale runs_dir.
    project_path = project if Path(project).is_absolute() else str(PROJECT_ROOT / project)
    yolo = YOLO(model)
    return yolo.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        project=project_path,
        name=name,
        **overrides,
    )
