"""Prepara il dataset SH17 in una struttura YOLO train/val/test.

SH17 fornisce le immagini (scaricate tramite lo script ufficiale Pexels) più le
label ``.txt`` in formato YOLO, e gli elenchi opzionali ``train_files.txt`` /
``val_files.txt`` / ``test_files.txt``. Questo modulo organizza tutto nella
struttura canonica di Ultralytics::

    data/processed/sh17/
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    └── data.yaml

Per impostazione predefinita le immagini vengono collegate via symlink (nessuna
duplicazione dei dati); passa ``copy=True`` per copiarle invece (es. su
filesystem senza supporto ai symlink).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from src.config import SH17_CLASSES
from src.data import IMAGE_EXTS, label_path_for_image, list_images

SPLIT_FILES = {"train": "train_files.txt", "val": "val_files.txt", "test": "test_files.txt"}


def _resolve_image_path(line: str, raw_root: Path) -> Path:
    """Risolve una voce di un file di split in un path immagine effettivo.

    I file di split di SH17 possono contenere path assoluti (dopo l'esecuzione
    dello script ufficiale update_train_test_files.py), path relativi alla radice
    del dataset, o nomi di file nudi. Proviamo tutti e tre, preferendo un file
    esistente.
    """
    p = Path(line)
    candidates = [p, raw_root / line, raw_root / "images" / p.name]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # best effort; le voci inesistenti vengono saltate dopo


def _read_split_lists(raw_root: Path) -> dict[str, list[Path]] | None:
    """Legge i file di split .txt di SH17 se presenti, altrimenti None.

    SH17 fornisce ``train_files.txt`` + ``test_files.txt`` (senza un val separato).
    Quando esistono solo questi due, usiamo ``test_files.txt`` come split di
    validazione, in linea con il benchmark ufficiale (1.620 immagini di validazione).
    """
    found = {}
    for split, fname in SPLIT_FILES.items():
        fpath = raw_root / fname
        if fpath.exists():
            lines = [ln.strip() for ln in fpath.read_text().splitlines() if ln.strip()]
            found[split] = [_resolve_image_path(ln, raw_root) for ln in lines]

    if "train" in found and "val" not in found and "test" in found:
        found["val"] = found.pop("test")  # convenzione SH17: elenco test == set di val
    if "train" in found and "val" in found:
        return found
    return None


def _random_split(images: list[Path], val_frac: float, test_frac: float, seed: int) -> dict[str, list[Path]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    n_test = int(len(images) * test_frac)
    n_val = int(len(images) * val_frac)
    test_idx = set(idx[:n_test].tolist())
    val_idx = set(idx[n_test : n_test + n_val].tolist())
    split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for i, img in enumerate(images):
        if i in test_idx:
            split["test"].append(img)
        elif i in val_idx:
            split["val"].append(img)
        else:
            split["train"].append(img)
    return split


def _place(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def prepare_sh17(
    raw_root: str | Path,
    processed_root: str | Path,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
    copy: bool = False,
) -> dict:
    """Costruisce il dataset YOLO elaborato e scrive il relativo ``data.yaml``.

    Restituisce un dict di riepilogo con il conteggio delle immagini per split.
    """
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    if not raw_root.exists():
        raise FileNotFoundError(
            f"Dataset grezzo non trovato in {raw_root}. Scarica prima SH17 "
            f"(vedi data/README.md)."
        )

    split = _read_split_lists(raw_root)
    source = "elenchi ufficiali train/val/test"
    if split is None:
        images = list_images(raw_root)
        if not images:
            raise FileNotFoundError(f"Nessuna immagine ({IMAGE_EXTS}) trovata in {raw_root}.")
        split = _random_split(images, val_frac, test_frac, seed)
        source = f"split casuale (val={val_frac}, test={test_frac}, seed={seed})"

    counts: dict[str, int] = {}
    for split_name, imgs in split.items():
        n = 0
        for img in imgs:
            if not img.exists():
                continue
            lbl = label_path_for_image(img)
            _place(img, processed_root / "images" / split_name / img.name, copy)
            if lbl.exists():
                _place(lbl, processed_root / "labels" / split_name / lbl.name, copy)
            n += 1
        counts[split_name] = n

    names = _class_names_from_source(raw_root)
    data_yaml = write_data_yaml(processed_root, names=names)
    return {"counts": counts, "split_source": source, "data_yaml": str(data_yaml)}


def _class_names_from_source(raw_root: Path) -> dict[int, str] | None:
    """Legge i nomi delle classi dal sh17.yaml del dataset se fornito, altrimenti None.

    I file di label .txt sono indicizzati rispetto al sh17.yaml del dataset, quindi
    affidarsi a esso (quando presente) evita qualsiasi disallineamento rispetto al
    nostro elenco hardcoded.
    """
    src_yaml = raw_root / "sh17.yaml"
    if not src_yaml.exists():
        return None
    try:
        cfg = yaml.safe_load(src_yaml.read_text())
        names = cfg.get("names")
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        if isinstance(names, list):
            return dict(enumerate(names))
    except Exception:
        return None
    return None


def write_data_yaml(processed_root: str | Path, names: dict[int, str] | None = None) -> Path:
    """Scrive un data.yaml Ultralytics autocontenuto dentro la radice elaborata."""
    processed_root = Path(processed_root)
    if names is None:
        names = {i: name for i, name in enumerate(SH17_CLASSES)}
    cfg = {
        "path": str(processed_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    out = processed_root / "data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out
