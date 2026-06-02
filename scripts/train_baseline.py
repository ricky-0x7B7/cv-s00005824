"""Addestra il modello di riferimento classico HOG + SVM (helmet vs no_helmet).

Uso::

    python scripts/train_baseline.py --raw data/processed/sh17/images/train \\
        --config configs/baseline_svm.yaml --output models/hog_svm_baseline.pkl
"""

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from src.classical_model import SVMConfig, evaluate_classifier, save_model, train_svm
from src.config import CLASS_TO_ID, set_reproducibility
from src.data import extract_crops_for_binary
from src.features import HOGConfig, extract_hog_batch
from src.visualization import plot_confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Modello di riferimento HOG + SVM")
    parser.add_argument("--raw", default="data/processed/sh17/images/train",
                        help="Directory delle immagini con le etichette YOLO affiancate (in labels/)")
    parser.add_argument("--config", default="configs/baseline_svm.yaml")
    parser.add_argument("--output", default="models/hog_svm_baseline.pkl")
    parser.add_argument("--metrics", default="reports/metrics/baseline_metrics.json")
    parser.add_argument("--figure", default="reports/figures/baseline_confusion_matrix.png")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_reproducibility(cfg.get("seed", 42))

    print("Estrazione dei ritagli...")
    X_img, y = extract_crops_for_binary(
        images_dir=args.raw,
        positive_class=CLASS_TO_ID[cfg["positive_class"]],
        negative_class=CLASS_TO_ID[cfg["negative_class"]],
        crop_size=tuple(cfg["crop_size"]),
        helmet_iou_exclude=cfg["helmet_iou_exclude"],
        max_per_class=cfg["max_per_class"],
        seed=cfg.get("seed", 42),
    )
    print(f"  ritagli: {len(y)}  (positivi={int(y.sum())}, negativi={int((y == 0).sum())})")
    if len(y) == 0:
        raise SystemExit("Nessun ritaglio estratto — controlla il percorso del dataset e le etichette.")

    print("Estrazione delle feature HOG...")
    hog_cfg = HOGConfig(
        orientations=cfg["hog"]["orientations"],
        pixels_per_cell=tuple(cfg["hog"]["pixels_per_cell"]),
        cells_per_block=tuple(cfg["hog"]["cells_per_block"]),
        block_norm=cfg["hog"]["block_norm"],
        transform_sqrt=cfg["hog"]["transform_sqrt"],
        grayscale=cfg["hog"]["channel_axis"] is None,
    )
    X = extract_hog_batch(X_img, hog_cfg)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg["val_fraction"], stratify=y, random_state=cfg.get("seed", 42)
    )

    print("Addestramento dell'SVM...")
    svm_cfg = SVMConfig(
        kernel=cfg["svm"]["kernel"], C=cfg["svm"]["C"],
        gamma=cfg["svm"]["gamma"], class_weight=cfg["svm"]["class_weight"],
    )
    model = train_svm(X_tr, y_tr, svm_cfg)

    metrics = evaluate_classifier(model, X_val, y_val)
    print("Metriche di validazione:")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k:9s}: {metrics[k]:.4f}")

    save_model(model, args.output)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(metrics["confusion_matrix"], ["no_helmet", "helmet"], args.figure,
                          title="Matrice di confusione HOG+SVM")
    print(f"Modello salvato -> {args.output}\nMetriche salvate -> {args.metrics}\nFigura salvata -> {args.figure}")


if __name__ == "__main__":
    main()
