"""Valuta i pesi YOLO addestrati ed esporta metriche + figure.

Uso::

    python scripts/evaluate_yolo.py --weights models/best.pt \\
        --data configs/sh17.yaml --device auto --split test
"""

import _bootstrap  # noqa: F401
import argparse
import json

from src.evaluate import evaluate_yolo


def main() -> None:
    parser = argparse.ArgumentParser(description="Valuta il rilevatore YOLO")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--data", default="data/processed/sh17/data.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    summary = evaluate_yolo(
        weights=args.weights, data=args.data, device=args.device,
        imgsz=args.imgsz, split=args.split,
    )
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))
    print("Metrics -> reports/metrics/yolo_metrics.json")
    print("Figures -> reports/figures/")


if __name__ == "__main__":
    main()
