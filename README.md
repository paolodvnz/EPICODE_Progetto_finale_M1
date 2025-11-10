# Progetto Analisi di Vendite in una Catena di Negozi 

Progetto Python per l'analisi di dati di vendita utilizzando le principali librerie di data science.

## Descrizione

Questo progetto dimostra l'utilizzo di Python per la generazione, manipolazione, analisi e visualizzazione di dati relativi a vendite di prodotti di una catena di negozi. Il codice è strutturato in 7 parti progressive che coprono le operazioni più comuni nell'analisi dati.

## Struttura del Progetto

### PARTE 0 - CSV File Manager
Implementazione di due classi per la gestione di file CSV:
- `dict_csv`: Gestione CSV tramite liste di dizionari
- `pandas_csv`: Gestione CSV tramite DataFrame pandas

Entrambe supportano operazioni di salvataggio, caricamento e aggiunta di righe, con funzionalità di backup.

### PARTE 1 - Dataset di Base
Generazione di un dataset sintetico di 100 vendite con:
- Date consecutive (100 giorni a partire dal 1 novembre 2025)
- 6 negozi (Milano, Torino, Roma, Firenze, Napoli, Bari)
- 9 prodotti divisi in 3 categorie:
  - **Informatica**: Smartphone, Laptop, Tablet (€200-€2000)
  - **Accessori**: Webcam, Mouse, Tastiera (€50-€500)
  - **Elettrodomestici**: TV, Lavastoviglie, Frigorifero (€300-€3000)
- Quantità casuali (1-10 unità per vendita)
- Prezzi verosimili per categoria

### PARTE 2 - Importazione con Pandas
Caricamento del dataset CSV in un DataFrame pandas e visualizzazione delle informazioni di base:
- Prime righe del dataset
- Dimensioni (righe e colonne)
- Tipi di dati e informazioni sul DataFrame

### PARTE 3 - Elaborazione con Pandas
Analisi dei dati con operazioni di aggregazione:
- Calcolo dell'incasso per ogni vendita
- Incasso cumulativo nel tempo
- Statistiche per negozio (incasso medio e totale)
- Top 3 prodotti per quantità venduta
- Analisi combinata negozio-prodotto

### PARTE 4 - Uso di NumPy
Operazioni numeriche con array NumPy:
- Conversione da pandas a numpy
- Calcolo di statistiche
- Operazioni vettoriali su array 2D
- Verifica dell'equivalenza dei calcoli con pandas

### PARTE 5 - Visualizzazione con Matplotlib
Creazione di grafici per la visualizzazione dei risultati:
- **Grafico 1**: Grafici a barre per incasso totale e medio per negozio
- **Grafico 2**: Grafico a torta per la distribuzione percentuale degli incassi per negozio e per prodotto
- **Grafico 3**: Grafici temporali per incasso giornaliero, media modile settimanale e cumulativo

### PARTE 6 - Analisi Avanzata
Analisi per categoria di prodotto:
- Mappatura prodotti alle rispettive categorie
- Aggregazione per categoria (incasso totale e quantità media)
- Salvataggio del dataset analizzato completo

### PARTE 7 - Estensioni
Analisi aggiuntive e visualizzazioni avanzate:
- Normalizzazione percentuale dei dati
- Grafico combinato (barre + linee) per confronto incasso e quantità
- Identificazione dei top 5 prodotti più redditizi

## File Generati

Durante l'esecuzione, il programma genera i seguenti file CSV:
- `vendite.csv`: Dataset base delle vendite
- `vendite_analizzate.csv`: Dataset completo con tutte le colonne calcolate (incasso, categoria, ecc.)

## Utilizzo

Eseguire semplicemente lo script Python:

```bash
python analisi_vendite.py
```

Il programma eseguirà tutte le 7 parti in sequenza, mostrando:
- Informazioni di log per ogni operazione
- Statistiche calcolate
- Grafici interattivi (matplotlib)

## Note

- Il dataset è generato casualmente, quindi i risultati varieranno ad ogni esecuzione
- I prezzi sono generati in modo verosimile per categoria di prodotto
- Le date sono consecutive per facilitare l'analisi temporale 
  
