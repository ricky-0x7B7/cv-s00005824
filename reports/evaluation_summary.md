# Sintesi della valutazione

## Executive Summary

Questo documento riassume i risultati principali ottenuti durante lo sviluppo del sistema di rilevamento dei Dispositivi di Protezione Individuale (DPI) basato su Computer Vision.

Il progetto combina tre componenti principali:

1. una baseline classica basata su HOG + SVM;
2. un detector profondo YOLOv8n fine-tuned sul dataset SH17;
3. un motore di compliance basato su regole per classificare i lavoratori come SAFE, PARTIALLY SAFE o UNSAFE.

L'obiettivo della valutazione è verificare non solo le prestazioni dei singoli modelli, ma anche la capacità dell'intera pipeline di supportare il monitoraggio della sicurezza nei cantieri.

Nel complesso, il detector YOLOv8n raggiunge prestazioni vicine al benchmark ufficiale SH17 e si dimostra affidabile nel rilevamento di persone e caschi. Le principali limitazioni emergono invece sulle classi DPI rare, in particolare i gilet ad alta visibilità, che influenzano direttamente il comportamento del motore di compliance.

---

# 1. Risultati principali

## YOLOv8n — Object Detection

Valutazione sullo split ufficiale SH17:

* 1.620 immagini
* 15.358 istanze annotate
* 17 classi

### Metriche

| Metrica            |      Valore |
| ------------------ | ----------: |
| Precision          |       0.628 |
| Recall             |       0.522 |
| mAP@50             |       0.543 |
| mAP@50-95          |       0.337 |
| Tempo di inferenza | ~2.4 ms/img |

### Interpretazione

Il detector raggiunge prestazioni vicine al benchmark ufficiale pubblicato per YOLOv8n sul dataset SH17:

| Modello        | mAP@50 | mAP@50-95 |
| -------------- | -----: | --------: |
| Nostro modello |  0.543 |     0.337 |
| Benchmark SH17 |  0.580 |     0.366 |

Lo scarto di circa 3–4 punti di mAP suggerisce che il modello sia riuscito a catturare gran parte della capacità predittiva riportata dagli autori del dataset pur essendo stato addestrato interamente in locale.

---

## Baseline classica — HOG + SVM

Valutazione sul task binario:

```text id="v8s4kq"
helmet vs no_helmet
```

su 955 crop di validazione.

### Metriche

| Metrica   | Valore |
| --------- | -----: |
| Accuracy  |  0.917 |
| Precision |  0.913 |
| Recall    |  0.542 |
| F1-score  |  0.680 |

### Interpretazione

La baseline mostra una precision elevata ma una recall limitata.

Questo significa che il classificatore tende a essere conservativo:

* commette pochi falsi positivi;
* perde una quota significativa dei caschi reali.

Pur ottenendo risultati soddisfacenti sul task semplificato di classificazione binaria, la baseline non è in grado di affrontare il problema completo di rilevamento multi-classe e localizzazione richiesto dall'applicazione finale.

---

# 2. Valutazione del motore di compliance

Per valutare il comportamento della pipeline completa è stato eseguito un test su:

```text id="v43m9e"
200 immagini
333 lavoratori rilevati
```

### Distribuzione degli stati

| Stato          | Conteggio | Percentuale |
| -------------- | --------: | ----------: |
| UNSAFE         |       309 |       92.8% |
| PARTIALLY SAFE |        14 |        4.2% |
| SAFE           |        10 |        3.0% |

A prima vista questi risultati potrebbero suggerire una situazione di diffusa non conformità.

Tuttavia un'analisi più approfondita mostra che questa interpretazione sarebbe fuorviante.

---

# 3. Il risultato più importante

Il dato più significativo dell'intera valutazione non è il valore di mAP ottenuto dal detector, ma il comportamento osservato nel motore di compliance.

L'elevata percentuale di lavoratori classificati come UNSAFE deriva principalmente da rilevamenti mancati e non da reali violazioni delle norme di sicurezza.

In particolare, la classe:

```text id="8gw52s"
safety-vest
```

presenta:

* una delle distribuzioni più sbilanciate del dataset;
* una mAP inferiore rispetto alle classi principali;
* un impatto diretto sulla logica di compliance.

Poiché la regola SAFE richiede il rilevamento contemporaneo di casco e gilet, la mancata individuazione del gilet porta automaticamente alla classificazione come PARTIALLY SAFE o UNSAFE.

Questo fenomeno rappresenta la principale limitazione dell'intero sistema.

---

# 4. Analisi critica

## Punti di forza

### Rilevamento delle persone

Le classi:

* person
* head
* face
* hands

raggiungono prestazioni elevate e dimostrano che il modello comprende correttamente la struttura generale della scena.

### Rilevamento del casco

La classe helmet raggiunge:

```text id="95xfs7"
mAP@50 = 0.661
```

un risultato soddisfacente per il caso d'uso previsto.

### Prestazioni computazionali

L'inferenza di circa:

```text id="gdn5uj"
2.4 ms per immagine
```

rende possibile l'utilizzo del sistema in scenari real-time.

---

## Principali limitazioni

### Classi rare

Le categorie con poche istanze di training mostrano prestazioni significativamente inferiori.

### Oggetti piccoli

Molti DPI occupano una porzione minima dell'immagine e risultano difficili da rilevare.

### Compliance conservativa

La logica di classificazione tende a penalizzare qualsiasi DPI non rilevato, producendo un numero elevato di falsi UNSAFE.

---

# 5. Messaggi chiave

L'intera valutazione può essere sintetizzata in quattro osservazioni principali.

## 1. YOLO supera nettamente la baseline classica

La classificazione tradizionale è utile come riferimento, ma non è sufficiente per affrontare un problema reale di rilevamento DPI.

## 2. Il dataset influenza fortemente le prestazioni

Lo sbilanciamento delle classi emerge come il principale fattore che limita il detector.

## 3. Il sistema è affidabile per persone e caschi

Le classi più importanti e meglio rappresentate raggiungono prestazioni solide.

## 4. La compliance va interpretata con cautela

I risultati SAFE/UNSAFE non rappresentano direttamente la conformità reale dei lavoratori, ma riflettono anche le limitazioni del detector.

---

# 6. Conclusioni

La valutazione mostra che il sistema sviluppato è in grado di rilevare lavoratori e DPI con prestazioni vicine ai benchmark pubblicati per SH17, mantenendo tempi di inferenza compatibili con applicazioni operative.

L'analisi evidenzia tuttavia che la qualità della compliance dipende fortemente dalla capacità di rilevare i DPI meno rappresentati nel dataset.

Di conseguenza, il sistema deve essere interpretato come uno strumento di supporto decisionale e non come un meccanismo automatico di applicazione delle norme di sicurezza.

Le future evoluzioni dovrebbero concentrarsi principalmente sul miglioramento del rilevamento delle classi rare, in particolare dei gilet ad alta visibilità, che rappresentano oggi la principale fonte di errore della pipeline.