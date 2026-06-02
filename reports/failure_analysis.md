# Analisi degli errori

## 1. Obiettivo

Le metriche aggregate descrivono le prestazioni complessive del sistema, ma non spiegano le cause degli errori osservati durante il rilevamento e la valutazione della compliance.

Per comprendere meglio il comportamento del modello è stata condotta una failure analysis qualitativa e quantitativa su un sottoinsieme di immagini del set di validazione.

L'obiettivo è identificare:

* le principali modalità di errore;
* le cause tecniche degli errori;
* l'impatto di tali errori sulla compliance;
* possibili strategie di miglioramento.

---

# 2. Metodologia

L'analisi è stata eseguita utilizzando:

```text
scripts/make_examples.py
```

su:

```text
200 immagini di validazione
```

Per ciascuna immagine sono stati confrontati:

* ground truth;
* rilevamenti del modello;
* output del motore di compliance.

I casi più significativi sono stati esportati automaticamente in:

```text
outputs/failure_cases/
```

e annotati visivamente per facilitare l'ispezione manuale.

---

# 3. Errori rilevati automaticamente

L'analisi ha identificato 20 casi rappresentativi.

| Tipo              | Conteggio | Significato                                   |
| ----------------- | --------: | --------------------------------------------- |
| person_overcount  |        11 | più persone rilevate rispetto al ground truth |
| person_undercount |         7 | persone presenti ma non rilevate              |
| helmet_missed     |         2 | casco annotato ma non rilevato                |

Questi risultati suggeriscono che la maggior parte degli errori riguarda il rilevamento e il conteggio delle persone, mentre i caschi risultano relativamente ben appresi dal modello.

---

# 4. Errore dominante: falsi UNSAFE

L'osservazione più importante emersa dall'analisi riguarda il comportamento del motore di compliance.

Su:

```text
333 lavoratori rilevati
```

la distribuzione osservata è:

| Stato          | Conteggio |
| -------------- | --------: |
| UNSAFE         |       309 |
| PARTIALLY SAFE |        14 |
| SAFE           |        10 |

A prima vista questo risultato potrebbe suggerire una diffusa assenza di DPI.

Tuttavia l'analisi qualitativa mostra che la maggior parte di questi casi non rappresenta una reale violazione delle norme di sicurezza.

La causa principale è invece la mancata rilevazione di DPI effettivamente presenti, in particolare dei gilet ad alta visibilità.

Di conseguenza:

> il problema principale non è la classificazione della compliance, ma il mancato rilevamento di alcuni DPI essenziali.

Questo risultato rappresenta la conclusione più importante dell'intera analisi degli errori.

---

# 5. Casco non rilevato

## Descrizione

In alcuni casi il detector non riesce a identificare un casco presente nel ground truth.

## Cause osservate

* dimensioni molto ridotte;
* forte occlusione;
* prospettive sfavorevoli;
* presenza parziale del casco nell'immagine.

## Evidenze

Esempi:

```text
helmet_missed__*
```

## Impatto

L'errore genera:

* falsi negativi;
* classificazioni PARTIALLY SAFE o UNSAFE non corrette.

## Possibili mitigazioni

* aumento della risoluzione di input;
* oversampling delle istanze helmet;
* tecniche di copy-paste augmentation;
* maggiore quantità di dati annotati.

---

# 6. Gilet mancato → falso UNSAFE

## Descrizione

Questo rappresenta il problema più frequente e più impattante dell'intera pipeline.

Il lavoratore viene correttamente rilevato ma il gilet non viene individuato.

## Cause osservate

### Classe rara

Nel dataset SH17:

| Classe      | Istanze |
| ----------- | ------: |
| safety-vest |     433 |

pari a circa:

```text
0.7% delle annotazioni
```

### Prestazioni inferiori

La classe safety-vest ottiene:

```text
mAP@50 = 0.431
```

inferiore rispetto a molte altre classi principali.

## Impatto

Poiché la logica di compliance richiede la presenza contemporanea di:

* casco;
* gilet;

la mancata rilevazione del gilet comporta automaticamente un peggioramento della valutazione finale.

Questo singolo errore spiega gran parte dell'elevata percentuale di lavoratori classificati come UNSAFE.

## Possibili mitigazioni

* raccolta di ulteriori esempi di gilet;
* class weighting dedicato;
* oversampling;
* soglie di confidenza specifiche per classe.

---

# 7. Lavoratori mancati

## Descrizione

In alcune immagini il detector non individua tutti i lavoratori presenti.

## Cause osservate

* persone distanti;
* occlusioni;
* scene particolarmente dense;
* lavoratori parzialmente visibili.

## Evidenze

Esempi:

```text
person_undercount__*
```

## Impatto

L'errore riduce la recall e può impedire completamente la valutazione della compliance di alcuni soggetti presenti nella scena.

## Possibili mitigazioni

* inferenza a tile;
* immagini a maggiore risoluzione;
* tracking multi-frame.

---

# 8. Persone duplicate

## Descrizione

In alcuni casi il modello produce più bounding box per lo stesso lavoratore.

## Evidenze

Esempi:

```text
person_overcount__*
```

## Cause osservate

* sovrapposizione tra persone;
* NMS non sufficientemente aggressiva;
* possibili discrepanze nelle annotazioni del dataset.

## Impatto

L'errore altera il conteggio dei lavoratori e può generare valutazioni duplicate della compliance.

## Possibili mitigazioni

* tuning della confidence;
* tuning della soglia IoU della NMS;
* post-processing aggiuntivo.

---

# 9. Collasso delle classi rare

## Descrizione

Le classi meno rappresentate mostrano prestazioni significativamente inferiori.

## Esempi

| Classe      | mAP@50 |
| ----------- | -----: |
| foot        |  0.102 |
| ear-mufs    |  0.194 |
| safety-suit |  0.288 |

## Interpretazione

Queste classi dispongono di pochi esempi di training e spesso corrispondono a oggetti molto piccoli.

L'errore non appare riconducibile a problemi specifici dell'architettura ma alla limitata disponibilità di dati.

## Possibili mitigazioni

* aumento delle annotazioni;
* fusione di classi semanticamente simili;
* esclusione delle classi marginali dalla logica di compliance.

---

# 10. Associazione DPI → persona errata

## Descrizione

Il motore di compliance associa ogni DPI a un lavoratore utilizzando regole geometriche.

Quando due persone si sovrappongono, questa procedura può fallire.

## Cause osservate

* lavoratori molto vicini;
* sovrapposizione dei bounding box;
* occlusioni reciproche.

## Impatto

Un DPI può essere assegnato alla persona sbagliata, alterando il risultato finale della compliance.

## Possibili mitigazioni

* tracking;
* pose estimation;
* associazione basata su keypoint.

---

# 11. Cosa abbiamo imparato

L'analisi degli errori suggerisce alcune conclusioni importanti.

## Il rilevamento delle persone non è il problema principale

Le classi person, head e face mostrano prestazioni elevate e stabili.

## Il casco è appreso correttamente

Gli errori sui caschi esistono ma rimangono relativamente contenuti.

## I DPI rari rappresentano il vero collo di bottiglia

Le prestazioni peggiori coincidono sistematicamente con le classi meno rappresentate nel dataset.

## La compliance eredita gli errori del detector

Il motore di compliance è semplice e interpretabile, ma la qualità del suo output dipende direttamente dalla qualità del rilevamento.

---

# 12. Conclusioni

La failure analysis mostra che la maggior parte degli errori osservati non deriva da instabilità del modello o da problemi dell'architettura YOLOv8.

Le criticità principali sono invece riconducibili a caratteristiche strutturali del dataset:

* forte sbilanciamento delle classi;
* scarsità di esempi DPI;
* presenza di oggetti molto piccoli;
* occlusioni e scene affollate.

In particolare, il mancato rilevamento dei gilet ad alta visibilità emerge come la causa dominante degli errori di compliance.

Questo risultato suggerisce che i futuri miglioramenti dovrebbero concentrarsi prioritariamente sulla qualità e sulla distribuzione dei dati piuttosto che sulla sostituzione dell'architettura di rilevamento.