# Analisi del dataset — SH17

## 1. Introduzione

La qualità e la distribuzione dei dati rappresentano uno degli elementi più importanti in qualsiasi progetto di Computer Vision. Prima di procedere all'addestramento dei modelli è quindi stata eseguita un'analisi esplorativa del dataset SH17, con l'obiettivo di comprenderne struttura, distribuzione delle classi, criticità e possibili implicazioni sulle prestazioni finali.

L'analisi del dataset ha inoltre guidato diverse decisioni progettuali, tra cui la scelta dell'architettura di rilevamento, delle strategie di preprocessing e l'interpretazione dei risultati ottenuti in fase di valutazione.

---

# 2. Origine del dataset

Il progetto utilizza il dataset pubblico **SH17**, progettato per il rilevamento di lavoratori e Dispositivi di Protezione Individuale (DPI) in ambienti industriali e cantieri.

## Sorgente

* Repository: https://github.com/ahmadmughees/SH17dataset
* Paper: arXiv:2407.04590
* Licenza: CC BY-NC-SA 4.0

Le immagini derivano principalmente dalla piattaforma Pexels e sono state annotate manualmente per identificare sia i DPI sia alcune parti del corpo utili al rilevamento.

---

# 3. Obiettivo del dataset

A differenza dei dataset generalisti come COCO, SH17 è stato costruito specificamente per il monitoraggio della sicurezza sul lavoro.

Le annotazioni comprendono:

* persone;
* caschi di sicurezza;
* gilet ad alta visibilità;
* guanti;
* occhiali protettivi;
* protezioni auricolari;
* mascherine;
* tute protettive;
* parti anatomiche come testa, mani, volto e orecchie.

Questa struttura rende SH17 particolarmente adatto allo sviluppo di sistemi di compliance automatica basati sulla presenza o assenza di DPI.

---

# 4. Statistiche generali

## Dimensioni del dataset

| Metrica                    | Valore |
| -------------------------- | -----: |
| Immagini totali            |  8.099 |
| Train                      |  6.479 |
| Validation                 |  1.620 |
| Oggetti annotati totali    | 75.994 |
| Oggetti annotati nel train | 60.636 |
| Istanze nella validation   | 15.358 |
| Numero di classi           |     17 |
| Oggetti medi per immagine  |   9.38 |

Lo split utilizzato corrisponde a quello ufficialmente fornito dagli autori del dataset.

---

# 5. Distribuzione delle classi

La distribuzione delle classi è fortemente sbilanciata.

## Classi più frequenti

| Classe | Istanze |    % |
| ------ | ------: | ---: |
| hands  |  12.638 | 20.8 |
| person |  11.068 | 18.3 |
| head   |   9.558 | 15.8 |
| face   |   7.095 | 11.7 |
| ear    |   6.118 | 10.1 |

Queste cinque categorie rappresentano da sole circa il 77% delle annotazioni presenti nel dataset.

## Classi DPI principali

| Classe      | Istanze |   % |
| ----------- | ------: | --: |
| gloves      |   2.261 | 3.7 |
| glasses     |   1.547 | 2.6 |
| helmet      |     773 | 1.3 |
| foot        |     610 | 1.0 |
| face-mask   |     519 | 0.9 |
| safety-vest |     433 | 0.7 |

## Classi molto rare

Le categorie:

* ear-mufs
* safety-suit
* medical-suit
* face-guard

presentano meno di 300 esempi ciascuna.

---

# 6. Analisi dello sbilanciamento

Lo sbilanciamento delle classi rappresenta la caratteristica più importante dell'intero dataset.

In particolare, i DPI che costituiscono il nucleo della valutazione di sicurezza risultano estremamente meno rappresentati rispetto alle parti del corpo.

Ad esempio:

| Classe      | Istanze |
| ----------- | ------: |
| person      |  11.068 |
| head        |   9.558 |
| helmet      |     773 |
| safety-vest |     433 |

Questo significa che durante l'addestramento il modello osserva circa:

* 14 volte più teste che caschi;
* 25 volte più persone che gilet ad alta visibilità.

Di conseguenza il detector tende naturalmente a ottimizzarsi sulle classi più frequenti, mentre le classi rare risultano più difficili da apprendere.

Questa proprietà spiega gran parte delle differenze osservate successivamente nelle metriche per classe.

---

# 7. Analisi delle dimensioni degli oggetti

Un'altra caratteristica rilevante del dataset è la presenza di numerosi oggetti molto piccoli.

Secondo la documentazione SH17, oltre 39.000 annotazioni occupano meno dell'1% dell'area dell'immagine.

Gli oggetti maggiormente interessati da questo fenomeno sono:

* protezioni auricolari;
* occhiali;
* piedi;
* DPI osservati a distanza;
* lavoratori in secondo piano.

Il rilevamento di oggetti piccoli è notoriamente una delle sfide principali per i detector one-stage e contribuisce direttamente alla riduzione della recall sulle classi meno rappresentate.

---

# 8. Implicazioni sul progetto

L'analisi del dataset ha influenzato numerose scelte progettuali.

## Scelta di YOLOv8

La presenza di molti oggetti piccoli e numerose classi ha portato alla scelta di un detector moderno preaddestrato, in grado di sfruttare conoscenza trasferita da dataset più ampi.

## Mantenimento delle 17 classi

Nonostante il forte sbilanciamento, tutte le classi sono state mantenute per preservare la completezza del problema e consentire una valutazione realistica delle prestazioni.

## Interpretazione delle metriche

I risultati del modello non possono essere interpretati soltanto attraverso le metriche aggregate.

Una parte significativa degli errori è infatti riconducibile alle proprietà del dataset piuttosto che a limiti dell'architettura utilizzata.

---

# 9. Limiti del dataset

L'analisi evidenzia alcuni limiti strutturali.

## Sbilanciamento delle classi

I DPI più importanti per la compliance risultano tra le categorie meno rappresentate.

## Oggetti molto piccoli

La presenza di annotazioni estremamente piccole rende difficile il rilevamento accurato.

## Complessità delle scene

Molte immagini contengono:

* occlusioni;
* lavoratori sovrapposti;
* DPI parzialmente visibili;
* elevata densità di oggetti.

Tali condizioni aumentano la difficoltà del problema rispetto ai classici benchmark di object detection.

---

# 10. Conclusioni

L'analisi del dataset mostra che SH17 costituisce un benchmark realistico e particolarmente impegnativo per il rilevamento di DPI.

Le due caratteristiche dominanti del dataset sono:

1. il forte sbilanciamento delle classi;
2. l'elevata presenza di oggetti piccoli.

Entrambe influenzano direttamente il comportamento dei modelli addestrati e spiegano gran parte delle difficoltà osservate nelle classi DPI più rare.

Per questo motivo i risultati ottenuti devono essere interpretati alla luce delle caratteristiche intrinseche del dataset e non esclusivamente come misura della qualità dell'architettura utilizzata.