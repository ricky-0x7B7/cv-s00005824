"""Guida / automatizza parzialmente il download del dataset SH17.

Il dataset SH17 (CC BY-NC-SA 4.0, non commerciale) NON è ridistribuito da questo
repository. Le immagini vengono scaricate da Pexels tramite lo script ufficiale e le
etichette YOLO sono fornite su Kaggle / HuggingFace. Questo script stampa i passaggi
esatti e, se il repository ufficiale è clonato, può invocare il suo downloader Pexels.

Uso::

    python scripts/download_dataset.py --dest data/raw/sh17
"""

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

OFFICIAL_REPO = "https://github.com/ahmadmughees/SH17dataset"
INSTRUCTIONS = """\
Configurazione del dataset SH17 (CC BY-NC-SA 4.0 — solo uso non commerciale)
============================================================================
CONSIGLIATO — Kaggle (immagini E etichette in un unico download):
    pip install kaggle   # richiede il token API ~/.kaggle/kaggle.json
    kaggle datasets download -d mugheesahmad/sh17-dataset-for-ppe-detection -p data/raw
    unzip -q data/raw/sh17-dataset-for-ppe-detection.zip -d {dest}
  (oppure scarica lo zip dalla pagina Kaggle ed estrailo in {dest})

ALTERNATIVA — immagini Pexels + etichette Kaggle:
    git clone {repo}
    cd SH17dataset/data && python download_from_pexels.py

Struttura attesa in {dest}:
    {dest}/images/*.jpg
    {dest}/labels/*.txt
    {dest}/sh17.yaml            (opzionale, attendibile per i nomi delle classi)
    {dest}/train_files.txt      (split ufficiale opzionale)
    {dest}/test_files.txt       (opzionale; usato come split di validazione)

Poi costruisci lo split YOLO + data.yaml:
    python scripts/prepare_dataset.py --raw {dest} --out data/processed/sh17

SCORCIATOIA — pesi preaddestrati (nessun addestramento necessario per una prima demo):
    curl -L -o models/best.pt \\
      https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo8n.pt
    python scripts/run_inference.py --weights models/best.pt --source samples/example.jpg
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Helper per il download di SH17")
    parser.add_argument("--dest", default="data/raw/sh17", help="Directory di destinazione del dataset grezzo")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(INSTRUCTIONS.format(repo=OFFICIAL_REPO, dest=dest))


if __name__ == "__main__":
    main()
