"""Esegue il fine-tuning di YOLO sul dataset DPI SH17.

Esempi
------
Addestramento completo (device automatico)::

    python scripts/train_yolo.py --data configs/sh17.yaml --model yolov8n.pt \\
        --epochs 80 --imgsz 640 --batch 16 --device auto

Smoke test su CPU (verifica che la pipeline funzioni)::

    python scripts/train_yolo.py --data configs/sh17.yaml --model yolov8n.pt \\
        --epochs 1 --imgsz 320 --batch 2 --device cpu

I flag della CLI hanno la precedenza sui valori in configs/train_yolo.yaml.
"""

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from src.config import set_reproducibility
from src.yolo_train import load_train_config, train_yolo

# Chiavi di augmentation / ottimizzatore inoltrate dallo YAML a Ultralytics.
PASSTHROUGH_KEYS = (
    "patience", "optimizer", "lr0", "lrf", "cos_lr", "amp",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
    "fliplr", "flipud", "mosaic", "close_mosaic", "mixup", "seed",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning di YOLO")
    parser.add_argument("--data", default="data/processed/sh17/data.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default="configs/train_yolo.yaml")
    parser.add_argument("--name", default=None)
    parser.add_argument("--fraction", type=float, default=None,
                        help="Frazione del set di addestramento da usare (es. 0.02 per uno smoke test veloce)")
    parser.add_argument("--resume", default=None,
                        help="Percorso a un checkpoint last.pt per riprendere un'esecuzione interrotta")
    args = parser.parse_args()

    # Percorso di ripresa: Ultralytics ripristina tutte le impostazioni dal checkpoint.
    if args.resume:
        from ultralytics import YOLO

        from src.config import select_device
        results = YOLO(args.resume).train(resume=True, device=select_device(args.device or "auto"))
        print("Addestramento ripreso completato. Pesi migliori:", Path(results.save_dir) / "weights" / "best.pt")
        return

    cfg = load_train_config(args.config) if Path(args.config).exists() else {}
    set_reproducibility(cfg.get("seed", 42))

    # La CLI ha la precedenza sullo YAML; lo YAML ha la precedenza sui valori predefiniti nel codice.
    overrides = {k: cfg[k] for k in PASSTHROUGH_KEYS if k in cfg}
    if args.fraction is not None:
        overrides["fraction"] = args.fraction
    results = train_yolo(
        data=args.data or cfg.get("data", "configs/sh17.yaml"),
        model=args.model or cfg.get("model", "yolov8n.pt"),
        epochs=args.epochs if args.epochs is not None else cfg.get("epochs", 80),
        imgsz=args.imgsz if args.imgsz is not None else cfg.get("imgsz", 640),
        batch=args.batch if args.batch is not None else cfg.get("batch", 16),
        device=args.device or cfg.get("device", "auto"),
        project=cfg.get("project", "runs/ppe_detection"),
        name=args.name or cfg.get("name", "yolo_sh17"),
        **overrides,
    )
    print("Addestramento completato. Pesi migliori:", Path(results.save_dir) / "weights" / "best.pt")


if __name__ == "__main__":
    main()
