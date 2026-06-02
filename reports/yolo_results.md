# Risultati del rilevamento YOLO

## 1. Obiettivo dell'esperimento

L'obiettivo principale del progetto è verificare la capacità di un detector moderno di identificare lavoratori e Dispositivi di Protezione Individuale (DPI) in immagini di cantiere reali.

Per questo scopo è stato addestrato un modello **YOLOv8n** sul dataset SH17 e successivamente valutato sullo split ufficiale di validazione.

L'esperimento rappresenta il nucleo principale del progetto e costituisce il riferimento per tutte le analisi successive, incluse la valutazione della compliance e l'analisi degli errori.

---

# 2. Configurazione sperimentale

## Modello

| Parametro    | Valore  |
| ------------ | ------- |
| Architettura | YOLOv8n |
| Parametri    | 3.0 M   |
| GFLOPs       | 8.1     |

## Training

| Parametro      | Valore       |
| -------------- | ------------ |
| Epochs         | 80           |
| Image Size     | 640          |
| Batch Size     | 16           |
| Ottimizzatore  | AdamW (auto) |
| Early Stopping | Patience 20  |
| Seed           | 42           |

## Hardware

| Componente        | Valore        |
| ----------------- | ------------- |
| Sistema           | Apple Silicon |
| GPU               | Apple M4 Max  |
| Backend           | PyTorch MPS   |
| Tempo di training | ~18.6 ore     |

Il modello è stato addestrato interamente in locale utilizzando il backend Metal Performance Shaders (MPS) di PyTorch.

---

# 3. Motivazione della scelta di YOLOv8

YOLOv8 è stato scelto come detector principale per diversi motivi.

## Rilevamento multi-classe

Il progetto richiede il riconoscimento simultaneo di:

* lavoratori;
* caschi;
* gilet;
* guanti;
* occhiali;
* altri DPI.

YOLO consente di affrontare tutte le classi in un unico modello.

## Prestazioni real-time

L'inferenza è sufficientemente veloce da rendere possibile l'utilizzo in applicazioni operative e sistemi di monitoraggio.

## Semplicità di deployment

YOLOv8 fornisce una pipeline integrata che include:

* object detection;
* non-maximum suppression;
* esportazione del modello;
* visualizzazione dei risultati.

## Trade-off accuratezza/velocità

La variante nano è stata selezionata per ottenere un compromesso favorevole tra costo computazionale e qualità delle predizioni.

---

# 4. Dataset di valutazione

La valutazione è stata eseguita sullo split ufficiale di validazione SH17.

| Metrica          | Valore |
| ---------------- | -----: |
| Immagini         |  1.620 |
| Istanze annotate | 15.358 |
| Classi           |     17 |

Le metriche riportate rappresentano quindi una valutazione completa dell'intero problema di rilevamento.

---

# 5. Risultati complessivi

## Metriche globali

| Metrica            |      Valore |
| ------------------ | ----------: |
| Precision          |       0.628 |
| Recall             |       0.522 |
| mAP@50             |   **0.543** |
| mAP@50-95          |   **0.337** |
| Tempo di inferenza | ~2.4 ms/img |

Questi risultati mostrano che il modello è in grado di identificare correttamente una larga parte degli oggetti presenti nelle immagini mantenendo al contempo tempi di inferenza compatibili con applicazioni real-time.

---

# 6. Confronto con il benchmark ufficiale SH17

Per contestualizzare le prestazioni ottenute è utile confrontarle con il benchmark pubblicato dagli autori del dataset.

| Modello           | Precision | Recall | mAP@50 | mAP@50-95 |
| ----------------- | --------: | -----: | -----: | --------: |
| Nostro modello    |     0.628 |  0.522 |  0.543 |     0.337 |
| YOLOv8n ufficiale |     0.675 |  0.536 |  0.580 |     0.366 |

## Interpretazione

Il modello addestrato nel progetto si colloca a circa:

* 3.7 punti di mAP@50;
* 2.9 punti di mAP@50-95;

dal benchmark ufficiale.

Considerando che:

* il training è stato eseguito interamente in locale;
* è stato utilizzato hardware consumer;
* è stato condotto un singolo esperimento completo;

il risultato può essere considerato vicino alle prestazioni riportate dagli autori del dataset.

---

# 7. Analisi per classe

Le prestazioni non sono uniformi tra le diverse categorie.

## Classi con migliori prestazioni

| Classe | mAP@50 |
| ------ | -----: |
| head   |  0.894 |
| face   |  0.891 |
| person |  0.887 |
| hands  |  0.826 |
| ear    |  0.726 |
| helmet |  0.661 |

Queste classi condividono due caratteristiche:

* elevata presenza nel dataset;
* dimensioni relativamente grandi all'interno delle immagini.

Il modello apprende quindi rappresentazioni robuste e generalizzabili.

---

## Classi con prestazioni più basse

| Classe      | mAP@50 |
| ----------- | -----: |
| safety-vest |  0.431 |
| face-guard  |  0.378 |
| safety-suit |  0.288 |
| tool        |  0.279 |
| ear-mufs    |  0.194 |
| foot        |  0.102 |

Queste categorie risultano significativamente più difficili da rilevare.

Le cause principali sono:

* scarsità di esempi nel training set;
* dimensioni ridotte degli oggetti;
* elevata variabilità visiva;
* occlusioni frequenti.

---

# 8. Analisi delle curve di training

L'andamento delle curve mostra un comportamento stabile durante tutto l'addestramento.

Osservazioni principali:

* diminuzione regolare delle loss;
* assenza di instabilità evidenti;
* miglioramenti progressivamente più contenuti nelle ultime epoche;
* early stopping non attivato.

Il modello continua quindi a migliorare fino alla conclusione dell'addestramento, pur con guadagni decrescenti nelle fasi finali.

---

# 9. Cosa abbiamo imparato

L'esperimento evidenzia alcuni aspetti particolarmente interessanti.

## Il problema non è il rilevamento delle persone

Le classi:

* person;
* head;
* face;
* hands;

raggiungono mAP molto elevate.

Questo suggerisce che l'architettura è pienamente in grado di comprendere la struttura generale della scena.

## Il vero collo di bottiglia sono i DPI rari

Le classi più problematiche coincidono quasi perfettamente con le classi meno rappresentate nel dataset.

Questo indica che una parte significativa dell'errore è attribuibile ai dati disponibili piuttosto che all'architettura utilizzata.

## Il casco è rilevato in modo soddisfacente

La classe helmet raggiunge:

```text id="a5nvrr"
mAP@50 = 0.661
```

un risultato significativamente superiore rispetto a molte altre classi DPI e sufficiente per supportare il motore di compliance.

## Il gilet rappresenta la principale criticità

La classe:

```text id="jsyyv8"
safety-vest
```

ottiene:

```text id="prlhpv"
mAP@50 = 0.431
```

e costituisce la principale fonte di errori nel sistema di compliance.

---

# 10. Confronto con la baseline classica

La baseline HOG + SVM affronta un problema molto più semplice:

* classificazione binaria;
* crop già estratti;
* nessuna localizzazione.

YOLO affronta invece il problema completo:

* rilevamento multi-classe;
* localizzazione;
* associazione DPI-persona;
* integrazione nella pipeline di compliance.

Il confronto evidenzia il salto di complessità tra i due approcci e giustifica l'utilizzo del detector profondo come componente principale del sistema.

---

# 11. Limiti osservati

L'analisi dei risultati mette in evidenza alcuni limiti.

## Classi rare

Le categorie con meno esempi rimangono difficili da apprendere.

## Oggetti piccoli

Molti DPI occupano una porzione minima dell'immagine.

## Scene affollate

Occlusioni e sovrapposizioni tra lavoratori introducono errori di rilevamento.

## Assenza di tracking

Ogni immagine viene elaborata indipendentemente, senza sfruttare informazione temporale.

---

# 12. Conclusioni

Il detector YOLOv8n rappresenta il componente più efficace dell'intera pipeline.

Con un valore di:

* mAP@50 pari a 0.543;
* mAP@50-95 pari a 0.337;

il modello raggiunge prestazioni vicine al benchmark ufficiale SH17 pur essendo stato addestrato interamente in locale.

L'analisi mostra che il sistema è particolarmente affidabile nel rilevamento di persone e caschi, mentre incontra maggiori difficoltà sulle classi DPI rare e sugli oggetti molto piccoli.

I risultati confermano la validità dell'approccio scelto e costituiscono la base per il successivo motore di compliance e per l'analisi dettagliata degli errori.