# Setup del dataset

Il dataset grezzo **non** è incluso in questo repository (dimensione + licenza
non commerciale SH17 **CC BY-NC-SA 4.0**).

## SH17 (dataset principale)
- Sorgente: <https://github.com/ahmadmughees/SH17dataset> · Paper: arXiv:2407.04590
- 8,099 immagini · 75,994 istanze · 17 classi · in media 9.38 istanze/immagine
- Licenza: **CC BY-NC-SA 4.0** (non commerciale, attribuzione, share-alike)

### Opzione A — Kaggle (consigliata: immagini **e** label in un unico download)
Dataset: `mugheesahmad/sh17-dataset-for-ppe-detection`

```bash
# with the Kaggle CLI (needs ~/.kaggle/kaggle.json API token)
pip install kaggle
kaggle datasets download -d mugheesahmad/sh17-dataset-for-ppe-detection -p data/raw
unzip -q data/raw/sh17-dataset-for-ppe-detection.zip -d data/raw/sh17
```
…oppure scarica lo zip dalla pagina Kaggle ed estrailo in `data/raw/sh17`.

### Opzione B — script Pexels (solo immagini) + label Kaggle
```bash
git clone https://github.com/ahmadmughees/SH17dataset
cd SH17dataset/data && python download_from_pexels.py   # images
# then fetch the YOLO labels from the Kaggle dataset above
```

### Struttura attesa sotto `data/raw/sh17`
```text
data/raw/sh17/
├── images/        # *.jpg / *.jpeg
├── labels/        # *.txt (YOLO format)
├── sh17.yaml      # (if shipped) authoritative class names — trusted automatically
└── train_files.txt / test_files.txt   # optional official split
```

### Costruzione dello split pronto per YOLO + data.yaml
```bash
python scripts/prepare_dataset.py --raw data/raw/sh17 --out data/processed/sh17
```
- Se sono presenti `train_files.txt`/`test_files.txt` vengono utilizzati (la
  lista `test` di SH17 viene trattata come split di validazione, in linea con il benchmark).
- Altrimenti viene creato uno split casuale riproducibile (val=15%, test=10%, seed=42).
- Se esiste `data/raw/sh17/sh17.yaml`, i suoi nomi di classe vengono usati alla lettera.

## Pesi preaddestrati (scorciatoia — esegui l'intera pipeline senza addestrare)
Gli autori di SH17 pubblicano pesi addestrati. Usali per fare subito una demo di
inferenza / compliance / valutazione:
```bash
curl -L -o models/best.pt \
  https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo8n.pt
python scripts/run_inference.py --weights models/best.pt --source samples/example.jpg
```
(Per l'esame dovresti comunque eseguire i tuoi esperimenti di fine-tuning — questi
pesi sono una baseline / sanity check utile.)

## Classi (17, indicizzate da 0 — ordine autoritativo da sh17.yaml)
`person, ear, ear-mufs, face, face-guard, face-mask, foot, tool, glasses,
gloves, helmet, hands, head, medical-suit, shoes, safety-suit, safety-vest`

> ⚠️ La lista di classi leggibile dal README di SH17 è in un ordine *diverso*
> rispetto agli indici reali delle label. I file `.txt` delle label seguono
> `sh17.yaml` (sopra) — è quello che usa questo progetto.

> `data/raw/` e `data/processed/` sono ignorati da git.
