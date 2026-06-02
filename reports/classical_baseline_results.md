# Baseline classica — HOG + SVM

## 1. Obiettivo della baseline

Prima di sviluppare il detector basato su Deep Learning è stata realizzata una baseline classica utilizzando feature handcrafted e un classificatore tradizionale.

L'obiettivo non era costruire un sistema competitivo rispetto a YOLO, ma:

* soddisfare il requisito di utilizzare tecniche classiche di Computer Vision;
* fornire un punto di riferimento quantitativo;
* confrontare approcci tradizionali e approcci deep learning;
* evidenziare le limitazioni delle feature progettate manualmente in un contesto reale.

La baseline affronta un problema significativamente più semplice rispetto al detector finale:

> classificare la presenza o assenza di un casco su immagini già ritagliate.

Non esegue quindi localizzazione, rilevamento multi-classe o associazione DPI-persona.

---

# 2. Definizione del task

È stato costruito un dataset binario derivato dalle annotazioni SH17.

## Classe positiva

Bounding box appartenenti alla classe:

```text
helmet
```

Numero di crop:

```text
773
```

## Classe negativa

Bounding box appartenenti alla classe:

```text
head
```

che non presentano sovrapposizione con alcun casco annotato.

Numero massimo utilizzato:

```text
4.000
```

## Dataset finale

| Metrica     |             Valore |
| ----------- | -----------------: |
| Crop totali |              4.773 |
| Train       |              3.818 |
| Validation  |                955 |
| Split       | Stratificato 80/20 |

Questo setup permette di valutare la capacità del modello di distinguere una testa con casco da una testa senza casco in condizioni realistiche.

---

# 3. Motivazione della scelta HOG

Per rappresentare le immagini è stato utilizzato **Histogram of Oriented Gradients (HOG)**, una delle tecniche più diffuse nella Computer Vision tradizionale.

HOG descrive la distribuzione locale dei gradienti e dei bordi presenti nell'immagine, trasformando ogni crop in un vettore numerico utilizzabile da algoritmi di machine learning classici.

La tecnica è stata scelta perché:

* semplice da implementare;
* interpretabile;
* storicamente efficace per il rilevamento di persone e oggetti;
* rappresentativa dell'approccio "feature engineering + classificatore".

---

# 4. Pipeline

L'intero processo segue il seguente flusso:

```text
Crop
→ Resize 128×128
→ Grayscale
→ HOG
→ StandardScaler
→ SVM
```

## Configurazione HOG

| Parametro    | Valore |
| ------------ | ------ |
| Orientazioni | 9      |
| Cell size    | 8×8    |
| Block size   | 2×2    |

## Configurazione SVM

| Parametro    | Valore   |
| ------------ | -------- |
| Kernel       | RBF      |
| C            | 10       |
| Class Weight | balanced |

L'utilizzo di `class_weight=balanced` è stato necessario per compensare il forte sbilanciamento tra esempi positivi e negativi.

---

# 5. Risultati

Valutazione sul set di validazione composto da 955 campioni.

| Metrica   | Valore |
| --------- | -----: |
| Accuracy  |  0.917 |
| Precision |  0.913 |
| Recall    |  0.542 |
| F1-score  |  0.680 |

## Matrice di confusione

|                 | Predetto no_helmet | Predetto helmet |
| --------------- | -----------------: | --------------: |
| Reale no_helmet |                792 |               8 |
| Reale helmet    |                 71 |              84 |

---

# 6. Interpretazione dei risultati

A prima vista l'accuracy del 91.7% potrebbe suggerire prestazioni elevate.

Tuttavia un'analisi più approfondita mostra un comportamento fortemente conservativo.

Il modello genera soltanto:

```text
8 falsi positivi
```

ma manca:

```text
71 caschi reali
```

Questo comportamento produce:

* precision elevata (0.913);
* recall limitata (0.542).

In altre parole, quando la SVM predice la presenza di un casco tende ad avere ragione, ma fallisce nel riconoscere una parte significativa dei caschi effettivamente presenti.

---

# 7. Principali limitazioni

L'analisi qualitativa suggerisce che gli errori siano principalmente dovuti a:

## Variabilità visiva dei caschi

I caschi presentano:

* colori differenti;
* forme differenti;
* orientamenti differenti;
* livelli di occlusione differenti.

Le feature HOG non riescono a rappresentare efficacemente tutta questa variabilità.

## Sensibilità al punto di vista

Piccole variazioni di prospettiva modificano significativamente i gradienti utilizzati da HOG.

## Assenza di informazione semantica

Le feature sono costruite manualmente e non possono apprendere automaticamente rappresentazioni più astratte dell'oggetto.

## Problema semplificato

La baseline lavora su crop già estratti.

Non deve:

* trovare le persone;
* localizzare i DPI;
* distinguere 17 classi;
* gestire scene affollate.

Si tratta quindi di un problema molto più semplice rispetto a quello affrontato dal detector finale.

---

# 8. Confronto concettuale con YOLO

La differenza principale tra la baseline e il detector finale non è solamente nelle metriche ottenute, ma nel tipo di problema affrontato.

| HOG + SVM              | YOLOv8                               |
| ---------------------- | ------------------------------------ |
| Classificazione        | Rilevamento                          |
| Binaria                | 17 classi                            |
| Crop pre-ritagliati    | Immagine completa                    |
| Nessuna localizzazione | Bounding box complete                |
| Nessuna compliance     | Integrazione nella pipeline completa |

Mentre la baseline risponde alla domanda:

> "Questo crop contiene un casco?"

YOLO risponde alla domanda:

> "Dove si trovano persone e DPI all'interno della scena?"

---

# 9. Conclusioni

La baseline HOG + SVM costituisce un utile riferimento per comprendere il contributo delle tecniche tradizionali di Computer Vision.

Pur ottenendo una buona accuracy (0.917) e un'elevata precision (0.913), il modello mostra una recall limitata e non è in grado di affrontare il problema completo di rilevamento dei DPI.

Il confronto evidenzia come le feature progettate manualmente risultino adeguate per semplici task di classificazione binaria, ma diventino rapidamente insufficienti quando il problema richiede localizzazione, rilevamento multi-classe e comprensione di scene complesse.

Questa osservazione costituisce una delle principali motivazioni che giustificano l'adozione di un detector profondo basato su YOLOv8 nel resto del progetto.