# Rilevamento DPI in cantiere e compliance di sicurezza

Una pipeline completa di Computer Vision per il rilevamento automatico dei lavoratori e dei loro Dispositivi di Protezione Individuale (DPI) nei cantieri, con classificazione finale dello stato di sicurezza di ciascun lavoratore come **SAFE**, **PARTIALLY SAFE** o **UNSAFE**.

Progetto sviluppato per l'esame di **Introduction to Computer Vision** del corso BsC in Computer Engineering & AI di EPICODE Institute of Technology, utilizzando il dataset pubblico **SH17**.

> Il progetto combina una baseline classica basata su **HOG + SVM**, un detector profondo **YOLOv8 fine-tuned** e un motore di **compliance basato su regole**, con l'obiettivo di valutare la fattibilità del monitoraggio automatico dei DPI in scenari di cantiere reali.

---

# 1. Contesto e motivazione

La mancata adozione dei Dispositivi di Protezione Individuale rappresenta una delle principali cause di rischio nei cantieri e negli ambienti industriali.

Il monitoraggio della compliance viene normalmente svolto da operatori umani attraverso controlli visivi periodici. Sebbene efficace, questo approccio presenta alcuni limiti:

* richiede tempo e risorse;
* è soggetto a errori e distrazioni;
* può risultare difficile in ambienti affollati o dinamici;
* non produce una documentazione strutturata delle osservazioni.

Negli ultimi anni i progressi nel Deep Learning hanno reso possibile il rilevamento automatico di persone e DPI direttamente da immagini e video.

Questo progetto esplora tale possibilità attraverso una pipeline completa di Computer Vision che:

1. rileva lavoratori e DPI;
2. valuta la presenza dei DPI essenziali;
3. produce un giudizio interpretabile di compliance.

Il sistema non è progettato come strumento di enforcement automatico delle norme di sicurezza, ma come **supporto decisionale** per responsabili della sicurezza e operatori umani.

---

# 2. Obiettivi del progetto

Gli obiettivi principali sono:

* analizzare un dataset reale dedicato alla sicurezza sul lavoro;
* confrontare un approccio classico e uno basato su Deep Learning;
* addestrare un detector multi-classe per il rilevamento di DPI;
* costruire un motore di compliance interpretabile;
* valutare quantitativamente e qualitativamente le prestazioni;
* analizzare i limiti del sistema e le principali modalità di errore.

---

# 3. Dataset

Il progetto utilizza il dataset pubblico **SH17**.

## Caratteristiche principali

| Metrica          | Valore |
| ---------------- | -----: |
| Immagini totali  |  8.099 |
| Train            |  6.479 |
| Validation       |  1.620 |
| Oggetti annotati | 75.994 |
| Classi           |     17 |

Le classi comprendono sia DPI sia parti del corpo:

```text
person
helmet
safety-vest
gloves
glasses
shoes
medical-suit
safety-suit
face-mask
face-guard
ear-mufs
tool

head
face
hands
ear
foot
```

## Sfide del dataset

L'analisi esplorativa ha evidenziato due criticità principali.

### Forte sbilanciamento delle classi

Le classi più frequenti sono:

| Classe | Istanze |
| ------ | ------: |
| hands  |  12.638 |
| person |  11.068 |
| head   |   9.558 |

Le classi più rilevanti per la compliance sono invece rare:

| Classe      | Istanze |
| ----------- | ------: |
| helmet      |     773 |
| safety-vest |     433 |

Questo squilibrio influenza direttamente le prestazioni del detector e rappresenta uno dei principali fattori che spiegano gli errori osservati durante la valutazione.

### Presenza di molti oggetti piccoli

SH17 contiene un elevato numero di oggetti che occupano meno dell'1% dell'area dell'immagine, rendendo difficile il rilevamento di:

* caschi lontani;
* gilet poco visibili;
* occhiali;
* protezioni auricolari;
* lavoratori in secondo piano.

---

# 4. Architettura della soluzione

L'intera pipeline segue il seguente flusso:

```text
Acquisition
    ↓
EDA e dataset audit
    ↓
Preprocessing e augmentation
    ↓
Baseline HOG + SVM
    ↓
YOLOv8 fine-tuned
    ↓
NMS e confidence filtering
    ↓
Compliance engine
    ↓
Valutazione
    ↓
Failure analysis
```

## Componenti principali

| Componente        | Scopo                                      |
| ----------------- | ------------------------------------------ |
| Dataset Audit     | Analisi della distribuzione dei dati       |
| HOG + SVM         | Baseline classica                          |
| YOLOv8n           | Detector principale                        |
| Compliance Engine | Valutazione SAFE / PARTIALLY SAFE / UNSAFE |
| Evaluation Suite  | Metriche quantitative                      |
| Failure Analysis  | Analisi sistematica degli errori           |

---

# 5. Baseline classica

Per soddisfare i requisiti del corso e fornire un termine di confronto è stata sviluppata una baseline basata su feature tradizionali.

Pipeline:

```text
Crop
→ Resize 128×128
→ Grayscale
→ HOG
→ StandardScaler
→ SVM (RBF)
```

Task affrontato:

```text
helmet vs no_helmet
```

La baseline opera esclusivamente su crop già ritagliati e non è in grado di effettuare localizzazione.

## Risultati baseline

| Metrica   | Valore |
| --------- | -----: |
| Accuracy  |  0.917 |
| Precision |  0.913 |
| Recall    |  0.542 |
| F1-score  |  0.680 |

Questa soluzione ottiene buoni risultati di classificazione ma non può affrontare il problema completo di rilevamento multi-classe richiesto dal caso d'uso.

---

# 6. Detector YOLOv8

Il modello principale è un **YOLOv8n** preaddestrato e successivamente fine-tuned sul dataset SH17.

## Configurazione

| Parametro  | Valore       |
| ---------- | ------------ |
| Modello    | YOLOv8n      |
| Epochs     | 80           |
| Image Size | 640          |
| Batch Size | 16           |
| Parametri  | 3.0 M        |
| GFLOPs     | 8.1          |
| Hardware   | Apple M4 Max |
| Backend    | PyTorch MPS  |

## Motivazione della scelta

YOLOv8 è stato selezionato perché offre:

* rilevamento multi-classe end-to-end;
* velocità compatibili con scenari real-time;
* semplicità di deployment;
* ottimo compromesso tra accuratezza e costo computazionale.

L'addestramento completo ha richiesto circa **18,6 ore** su Apple Silicon.

---

# 7. Motore di compliance

Dopo il rilevamento, ogni DPI viene associato a un lavoratore tramite regole geometriche basate su:

* posizione relativa;
* centro del bounding box;
* overlap tra oggetti.

La classificazione finale segue le regole:

| DPI rilevati     | Stato          |
| ---------------- | -------------- |
| Casco + gilet    | SAFE           |
| Solo uno dei due | PARTIALLY SAFE |
| Nessuno dei due  | UNSAFE         |

L'obiettivo è fornire un output interpretabile e facilmente verificabile da un operatore umano.

---

# 8. Risultati

## YOLOv8n

Validazione su:

```text
1.620 immagini
15.358 istanze
```

| Metrica         |      Valore |
| --------------- | ----------: |
| Precision       |       0.628 |
| Recall          |       0.522 |
| mAP@50          |       0.543 |
| mAP@50-95       |       0.337 |
| Tempo inferenza | ~2.4 ms/img |

### Confronto con benchmark SH17

| Modello           | mAP@50 | mAP@50-95 |
| ----------------- | -----: | --------: |
| YOLOv8n ufficiale |  0.580 |     0.366 |
| Nostro modello    |  0.543 |     0.337 |

Il modello raggiunge prestazioni vicine al benchmark pubblicato, con uno scarto di circa 3–4 punti di mAP.

---

# 9. Analisi critica dei risultati

Le prestazioni risultano elevate sulle classi più frequenti:

| Classe | mAP@50 |
| ------ | -----: |
| head   |  0.894 |
| face   |  0.891 |
| person |  0.887 |
| hands  |  0.826 |

Le prestazioni diminuiscono invece sulle classi rare:

| Classe      | mAP@50 |
| ----------- | -----: |
| safety-vest |  0.431 |
| face-guard  |  0.378 |
| safety-suit |  0.288 |
| ear-mufs    |  0.194 |
| foot        |  0.102 |

L'analisi suggerisce che il principale limite del sistema non sia l'architettura utilizzata, ma la scarsità di esempi disponibili per alcune classi DPI.

---

# 10. Compliance

Analisi eseguita su:

```text
200 immagini
333 lavoratori rilevati
```

| Stato          | Conteggio |
| -------------- | --------: |
| UNSAFE         |       309 |
| PARTIALLY SAFE |        14 |
| SAFE           |        10 |

Questo risultato non va interpretato come reale distribuzione delle violazioni.

L'analisi degli errori mostra che gran parte dei casi classificati come UNSAFE deriva da mancate rilevazioni di DPI rari, in particolare dei gilet ad alta visibilità.

Per questo motivo il sistema deve essere considerato un supporto decisionale e non uno strumento automatico di applicazione delle norme.

---

# 11. Principali contributi del progetto

* Audit completo del dataset SH17.
* Baseline classica HOG + SVM.
* Fine-tuning di YOLOv8n per 17 classi.
* Motore di compliance basato su regole interpretabili.
* Pipeline di valutazione quantitativa.
* Failure analysis automatizzata.
* Demo di inferenza e visualizzazione.

---

# 12. Setup

## Conda / Miniforge

```bash
conda env create -f environment.yml
conda activate s00005824
python -m ipykernel install --user --name s00005824
jupyter lab
```

## Docker

```bash
docker build -t s00005824:cpu -f Dockerfile.cpu .
docker run --rm -it -p 8888:8888 -v "$PWD":/app s00005824:cpu
```

---

# 13. Utilizzo

## Baseline

```bash
make baseline
```

## Training

```bash
make train
```

## Evaluation

```bash
make eval
```

## Inference

```bash
python scripts/run_inference.py \
    --weights models/best.pt \
    --source samples/esempio_cantiere.jpg
```

## Demo web

```bash
streamlit run app/streamlit_app.py
```

---

# 14. Limiti e sviluppi futuri

Possibili direzioni future:

* oversampling delle classi DPI rare;
* class weighting dedicato;
* inferenza ad alta risoluzione;
* inferenza a tile per piccoli oggetti;
* tracking multi-frame;
* confronto con YOLOv8s;
* studio sistematico delle augmentation.

---

# 15. Considerazioni etiche

L'utilizzo di sistemi automatici di monitoraggio della sicurezza richiede particolare attenzione a:

* privacy dei lavoratori;
* conservazione dei dati;
* rischio di sorveglianza impropria;
* falsi positivi e falsi negativi;
* responsabilità delle decisioni.

Il sistema è stato progettato come strumento di supporto al giudizio umano e non come sostituto delle procedure di sicurezza.

---

# 16. Riferimenti

* SH17 Dataset — https://github.com/ahmadmughees/SH17dataset
* Ultralytics YOLO — https://docs.ultralytics.com
* Dalal & Triggs, *Histograms of Oriented Gradients for Human Detection*, CVPR 2005
