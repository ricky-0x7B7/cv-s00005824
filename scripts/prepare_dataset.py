"""Costruisce la struttura train/val/test YOLO e data.yaml a partire dal dataset SH17 grezzo.

Uso::

    python scripts/prepare_dataset.py --raw data/raw/sh17 --out data/processed/sh17
"""

import _bootstrap  # noqa: F401
import argparse

from src.config import set_reproducibility
from src.convert import prepare_sh17


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara SH17 nel formato YOLO")
    parser.add_argument("--raw", default="data/raw/sh17", help="Radice di SH17 grezzo")
    parser.add_argument("--out", default="data/processed/sh17", help="Radice di output elaborata")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copia i file invece di creare collegamenti simbolici")
    args = parser.parse_args()

    set_reproducibility(args.seed)
    summary = prepare_sh17(
        raw_root=args.raw,
        processed_root=args.out,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        copy=args.copy,
    )
    print("Origine dello split:", summary["split_source"])
    for split, n in summary["counts"].items():
        print(f"  {split:5s}: {n} immagini")
    print("Scritto:", summary["data_yaml"])


if __name__ == "__main__":
    main()
