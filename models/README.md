# Modelli addestrati

I pesi dei modelli **non** sono committati (il `.gitignore` esclude `*.pt`/`*.pkl`).
Distribuiscili tramite una GitHub Release (`v1.0`) o l'Hugging Face Hub.

| File | Prodotto da | Descrizione |
|---|---|---|
| `best.pt` | `scripts/train_yolo.py` | Detector YOLO fine-tuned (copia da `runs/ppe_detection/<name>/weights/best.pt`) |
| `hog_svm_baseline.pkl` | `scripts/train_baseline.py` | Baseline classica HOG + SVM helmet-vs-no_helmet |

## Scaricare i pesi YOLO (`best.pt`)

I pesi sono pubblicati come **asset della Release `v1.0`**:
<https://github.com/ricky-0x7B7/cv-s00005824/releases/tag/v1.0>

Scegli uno dei tre metodi e posiziona il file in `models/best.pt`:

```bash
# 1) GitHub CLI (consigliato)
gh release download v1.0 -R ricky-0x7B7/cv-s00005824 -p best.pt -D models/

# 2) curl (link diretto, nessun login)
curl -L -o models/best.pt \
  https://github.com/ricky-0x7B7/cv-s00005824/releases/download/v1.0/best.pt

# 3) Browser: apri la pagina della Release qui sopra e scarica best.pt da "Assets"
```

Poi esegui l'inferenza:

```bash
python scripts/run_inference.py --weights models/best.pt --source samples/esempio_cantiere.jpg --device auto
```

> La baseline `hog_svm_baseline.pkl` (~130 MB) non è pubblicata: rigenerala con
> `make baseline` (o `python scripts/train_baseline.py`).

> Se decidi di committare direttamente `best.pt`, rimuovi la regola `*.pt` da
> `.gitignore` oppure usa Git LFS.
