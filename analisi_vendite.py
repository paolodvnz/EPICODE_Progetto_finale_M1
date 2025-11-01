import numpy as np
import csv
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path

# =======================================================================================================
# PARTE 0 - FILE MANAGER
# =======================================================================================================

class dict_csv():
    """
    Classe per gestire file CSV utilizzando liste di dizionari.
    
    Permette di salvare, caricare e aggiungere righe a file CSV,
    con supporto per file di backup automatici.
    """
    
    def __init__(self, file: str, path_save: str = ".", path_load: str = "."):
        """
        Inizializza il gestore CSV.
        
        Args:
            file: Nome del file senza estensione .csv
            path_save: Percorso dove salvare i file (default: cartella corrente)
            path_load: Percorso da dove caricare i file (default: cartella corrente)
        """
        self.path_save = Path(path_save)
        self.path_load = Path(path_load)
        
        self.file = self.path_save / f"{file}.csv"
        self._file_backup = self.path_save / f"{file}_backup.csv"
        self.file_load = self.path_load / f"{file}.csv"
        self._file_backup_load = self.path_load / f"{file}_backup.csv"
    
    def save(self, dict_list, backup=False):
        """
        Salva una lista di dizionari in un file CSV.
        
        Args:
            dict_list: Lista di dizionari da salvare. 
                      Le chiavi del primo dizionario definiscono le colonne.
            backup: Se True, salva nel file di backup invece che nel file principale.
        """
        keys = list(dict_list[0].keys())
        file = self._file_backup if backup else self.file
        
        # Crea la directory se non esiste
        file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(dict_list)
    
    def load(self, backup=False) -> list:
        """
        Carica un file CSV in una lista di dizionari.
        
        Args:
            backup: Se True, carica dal file di backup invece che dal file principale.
            
        Returns:
            Lista di dizionari, uno per ogni riga del CSV.
            
        Raises:
            FileNotFoundError: Se il file specificato non esiste.
        """
        list_dict = []
        file = self._file_backup_load if backup else self.file_load
        
        if not file.exists():
            raise FileNotFoundError(f"Il file '{file}' non esiste")
        
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                list_dict.append(row)
        return list_dict
    
    def append_rows(self, rows):
        """
        Aggiunge nuove righe a un file CSV esistente.
        
        Args:
            rows: Lista di dizionari da aggiungere al file.
                 Le chiavi devono corrispondere alle colonne esistenti.
                 
        Raises:
            FileNotFoundError: Se il file principale non esiste.
        """
        if not self.file.exists():
            raise FileNotFoundError(f"Il file '{self.file}' non esiste. Usa save() per creare il file prima.")
        
        keys = list(rows[0].keys())
        with open(self.file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writerows(rows)


class pandas_csv():
    """
    Classe per gestire file CSV utilizzando DataFrame pandas.
    
    Permette di salvare, caricare e aggiungere righe a file CSV
    sfruttando le funzionalità di pandas, con supporto per backup.
    """
    
    def __init__(self, file: str, path_save: str = ".", path_load: str = "."):
        """
        Inizializza il gestore CSV pandas.
        
        Args:
            file: Nome del file senza estensione .csv
            path_save: Percorso dove salvare i file (default: cartella corrente)
            path_load: Percorso da dove caricare i file (default: cartella corrente)
        """
        self.path_save = Path(path_save)
        self.path_load = Path(path_load)
        
        self.file = self.path_save / f"{file}.csv"
        self._file_backup = self.path_save / f"{file}_backup.csv"
        self.file_load = self.path_load / f"{file}.csv"
        self._file_backup_load = self.path_load / f"{file}_backup.csv"
    
    def save(self, df: pd.DataFrame, backup=False):
        """
        Salva un DataFrame pandas in un file CSV.
        
        Args:
            df: DataFrame da salvare.
            backup: Se True, salva nel file di backup.
        """
        file = self._file_backup if backup else self.file
        
        # Crea la directory se non esiste
        file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(file, index=False, encoding="utf-8")
    
    def load(self, backup=False) -> pd.DataFrame:
        """
        Carica un file CSV in un DataFrame pandas.
        
        Args:
            backup: Se True, carica dal file di backup.
            
        Returns:
            DataFrame contenente i dati del CSV.
            
        Raises:
            FileNotFoundError: Se il file specificato non esiste.
        """
        file = self._file_backup_load if backup else self.file_load
        
        if not file.exists():
            raise FileNotFoundError(f"Il file '{file}' non esiste")
        
        return pd.read_csv(file, encoding="utf-8")
    
    def append_rows(self, df: pd.DataFrame):
        """
        Aggiunge nuove righe a un file CSV esistente.
        
        Args:
            df: DataFrame contenente le righe da aggiungere.
            
        Raises:
            FileNotFoundError: Se il file principale non esiste.
        """
        if not self.file.exists():
            raise FileNotFoundError(f"Il file '{self.file}' non esiste. Usa save() per creare il file prima.")
        
        df.to_csv(self.file, mode='a', header=False, index=False, encoding="utf-8")

# =======================================================================================================
# PARTE 1 - DATASET DI BASE
# =======================================================================================================

print("="*80)
print("PARTE 1 - DATASET DI BASE")
print("="*80)

# Numero di righe (vendite) che verranno generate nel dataset
dimensione_dataset = 100
print(f"\nDimensione dataset: {dimensione_dataset} vendite")

# Creazione di un array di date consecutive partendo dal 1 novembre 2025,
# ogni vendita avrà una data diversa nell'arco di 100 giorni
date_dataset = np.arange('2025-11-01', dimensione_dataset, dtype='datetime64[D]')
print(f"\nPeriodo temporale: da {date_dataset[0]} a {date_dataset[-1]}")


# Lista dei negozi presenti nel dataset
negozi = ['Milano', 'Torino', 'Roma', 'Firenze', 'Napoli', 'Bari']
print(f"\nNegozi disponibili ({len(negozi)}):\n {', '.join(negozi)}")

# Lista dei prodotti venduti, suddivisi per categoria:
# - Informatica: Smartphone, Laptop, Tablet
# - Accessori: Webcam, Mouse, Tastiera
# - Elettrodomestici: TV, Lavastoviglie, Frigorifero
prodotti = [
    'Smartphone', 'Laptop', 'Tablet', 
    'Webcam', 'Mouse', 'Tastiera', 
    'TV', 'Lavastoviglie', 'Frigorifero'
]
print(f"\nProdotti disponibili ({len(prodotti)}):")
print(f"   - Informatica: Smartphone, Laptop, Tablet")
print(f"   - Accessori: Webcam, Mouse, Tastiera")
print(f"   - Elettrodomestici: TV, Lavastoviglie, Frigorifero")


# Genera casualmente le quantità vendute per ogni vendita (da 1 a 10 unità)
quantita_dataset = np.random.randint(1, 11, dimensione_dataset)
print(f"\nQuantità vendute generate:\n   - Min = {quantita_dataset.min()}\n   - Max = {quantita_dataset.max()}")

def estrai_prezzo(prodotto):
    """
    Estrae un prezzo casuale ma verosimile per il prodotto specificato.
    
    Fasce di prezzo:
    - Informatica (Smartphone, Laptop, Tablet): €200 - €2000
    - Accessori (Webcam, Mouse, Tastiera): €50 - €500
    - Elettrodomestici (TV, Lavastoviglie, Frigorifero): €30 - €300
    
    Args:
        prodotto (str): Nome del prodotto
    
    Returns:
        float: Prezzo casuale nella fascia appropriata
    """
    if prodotto in ['Smartphone', 'Laptop', 'Tablet']:
        prezzo_min = 200
        prezzo_max = 2000
    elif prodotto in ['Webcam', 'Mouse', 'Tastiera']:
        prezzo_min = 50
        prezzo_max = 500
    elif prodotto in ['TV', 'Lavastoviglie', 'Frigorifero']:
        prezzo_min = 500
        prezzo_max = 3000
    else:
        raise ValueError('Prodotto non presente!')
    
    # Formula per generare un valore casuale nell'intervallo [prezzo_min, prezzo_max]
    return (prezzo_max - prezzo_min) * np.random.random() + prezzo_min

print(f"\nFasce di prezzo configurate:")
print(f"   - Informatica: €200 - €2000")
print(f"   - Accessori: €20 - €200")
print(f"   - Elettrodomestici: €500 - €3000")

print(f"\n{'>'*10} Generazione dati casuali in corso...")

# Genera casualmente un negozio per ogni vendita
negozi_dataset = np.random.choice(negozi, dimensione_dataset).tolist()

# Genera casualmente un prodotto per ogni vendita
prodotti_dataset = np.random.choice(prodotti, dimensione_dataset).tolist()

# Genera un prezzo appropriato per ogni prodotto selezionato
# Il prezzo viene arrotondato a 2 decimali (centesimi)
prezzi_dataset = [round(estrai_prezzo(prodotto), 2) for prodotto in prodotti_dataset]

print(f"Dati generati con successo!")

print(f"\n{'>'*10} Creazione struttura dati...")

# Crea una lista di dizionari, dove ogni dizionario rappresenta una transazione
# Ogni transazione contiene: data, negozio, prodotto, quantità e prezzo unitario
dataset_dict = [
    {
        'data': data, 
        'negozio': negozio, 
        'prodotto': prodotto, 
        'quantita': quantita, 
        'prezzo_unitario': prezzo
    } 
    for data, negozio, prodotto, quantita, prezzo in zip(
        date_dataset, negozi_dataset, prodotti_dataset, quantita_dataset, prezzi_dataset
    )
]

print(f"Struttura dati creata: {len(dataset_dict)} record")

# Mostra un esempio delle prime 3 vendite
print(f"\nEsempio delle prime 3 vendite:")
for i, record in enumerate(dataset_dict[:3], 1):
    print(f"\nVendita {i}:")
    print(f"   - Data: {record['data']}")
    print(f"   - Negozio: {record['negozio']}")
    print(f"   - Prodotto: {record['prodotto']}")
    print(f"   - Quantità: {record['quantita']}")
    print(f"   - Prezzo unitario: €{record['prezzo_unitario']:.2f}")

print(f"\n{'>'*10} Salvataggio dataset su file...")

# Inizializza il file manager per gestire il CSV
vendite_csv = dict_csv('vendite')

# Salva i dati nella cartella corrente come file 'vendite.csv'
vendite_csv.save(dataset_dict)

print(f"Dataset salvato con successo nel file 'vendite.csv' nella cartella corrente")

print("\nFINE PARTE 1")
print("="*80)

# =======================================================================================================
# PARTE 2 - IMPORTAZIONE CON PANDAS
# =======================================================================================================

print("\n" + "="*80)
print("PARTE 2 - IMPORTAZIONE CON PANDAS")
print("="*80)

print(f"\n{'>'*10} Caricamento del dataset vendite.csv...")

# Inizializza il file manager per leggere il file CSV
vendite_pandas = pandas_csv('vendite')

# Carica il file CSV come DataFrame di pandas
df = vendite_pandas.load()

print(f"Dataset caricato con successo!")

print("\nPrime 5 vendite del dataset:")
print("-"*80)
print(f"{df.head(5)}")
print("-"*80)

print("\nDimensioni del dataset:")
print(f"Numero di righe (vendite): {df.shape[0]}")
print(f"Numero di colonne (features): {df.shape[1]}")

print("\nInformazioni dettagliate sul dataset:")
print("-"*80)
print(df.info())
print("-"*80)

print("\nFINE PARTE 2")
print("="*80)

# =======================================================================================================
# PARTE 3 - ELABORAZIONE CON PANDAS
# =======================================================================================================

print("\n" + "="*80)
print("PARTE 3 - ELEBORAZIONE CON PANDAS")
print("="*80)

# Crea una copia del dataframe originale per lavorarci senza modificare l'originale
# Questo è importante per mantenere i dati grezzi intatti
df_c = df.copy()
print("\nCopia del dataset creata per l'elaborazione")

# Aggiunge una nuova colonna 'incasso' calcolata moltiplicando quantità per prezzo unitario
df_c['incasso'] = (df_c['quantita'] * df_c['prezzo_unitario']).round(2)
print("\nColonna 'incasso' aggiunta al dataset")

# Aggiunge una colonna 'incasso_cumulativo' che somma progressivamente gli incassi
df_c['incasso_cumulativo'] = df_c['incasso'].cumsum().round(2)
print("\nColonna 'incasso_cumulativo' aggiunta al dataset")

# Aggiunge colonna 'incasso_rolling7' che tiene traccia della media mobile settimanale 
df_c['incasso_rolling'] = df_c['incasso'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
).round(2)

print("\nColonna 'incasso_rolling' aggiunta al dataset")

print("\n" + "-"*80)
print("ANALISI DEGLI INCASSI PER NEGOZIO")

# Raggruppa i dati per negozio e calcola:
# - incasso_medio
# - incasso_totale
df_negozi = df_c.groupby('negozio').agg(
    incasso_medio = ('incasso', 'mean'),
    incasso_totale = ('incasso', 'sum')
).round(2)

print("\nStatistiche incassi per negozio:\n")
print(df_negozi)

print("\n" + "-"*80)
print("ANALISI DEI PRODOTTI PIU' VENDUTI")

# Raggruppa i dati per prodotto e calcola:
# - quantita_totale
# - incasso_totale
# Ordina i risultati per quantità venduta 
df_prodotti = df_c.groupby('prodotto').agg(
    quantita_totale = ('quantita', 'sum'),
    incasso_totale = ('incasso', 'sum')
).sort_values(by='quantita_totale', ascending=False)

print("\nTop 3 prodotti per quantita' venduta:\n")
print(df_prodotti.head(3))

print("\n" + "-"*80)
print("ANALISI INCASSI PER NEGOZIO E PRODOTTO")

# Raggruppa i dati per combinazione di negozio e prodotto
# Calcola incasso medio e totale per ogni combinazione
df_prod_neg = df_c.groupby(['negozio', 'prodotto']).agg(
    incasso_medio = ('incasso', 'mean'),
    incasso_totale = ('incasso', 'sum')
).round(2)

print("\nIncassi per negozio e prodotto (prime 3 righe per ogni negozio):")

# Itera su tutti i negozi unici presenti nel dataframe
for group in df_prod_neg.index.get_level_values('negozio').unique():
    print(f"\nNegozio: {group}")
    print("-"*45)
    # Stampa le prime 3 combinazioni prodotto-incasso per questo negozio
    print(df_prod_neg.loc[group].head(3))
    print("-"*45)

print("\nFINE PARTE 3")
print("="*80)

# =======================================================================================================
# PARTE 4 - USO DI NUMPY
# =======================================================================================================

print("\n" + "="*80)
print("PARTE 4 - USO DI NUMPY")
print("="*80)

# Converte la colonna 'quantita' del dataframe in un array numpy
quantita_np = df_c['quantita'].to_numpy()
print(f"\nColonna 'quantita' convertita in array numpy:")
print(f"   - Tipo di dato: {type(quantita_np)}")
print(f"   - Dimensione array: {quantita_np.shape}")

print("\n" + "-"*80)
print("STATISTICHE SULLA QUANTITA' VENDUTA\n")

# Calcola le statistiche descrittive usando le funzioni numpy
media_quantita = quantita_np.mean()
std_quantita = quantita_np.std()
min_quantita = quantita_np.min()
max_quantita = quantita_np.max()

# Crea una maschera booleana per filtrare le quantità superiori alla media
mask = quantita_np > media_quantita

# Calcolo la percentuale di prenotazioni con importo sopra la media
quantita_sopra_media = quantita_np[mask].size
percentuale = quantita_sopra_media / quantita_np.size

# Stampa i risultati delle statistiche
print(f"   - Media quantita': {media_quantita:.2f}")
print(f"   - Deviazione standard: {std_quantita:.2f}")
print(f"   - Minimo: {min_quantita}")
print(f"   - Massimo: {max_quantita}")
print(f"   - Vendite con quantità sopra la media: {quantita_sopra_media} ({percentuale:.1%})")

print("\n" + "-"*80)
print("OPERAZIONI CON ARRAY 2D")

# Estrae due colonne dal dataframe e le converte in un array numpy 2D,
# ogni riga contiene [quantita, prezzo_unitario]
quantita_prezzo_np = df_c[['quantita','prezzo_unitario']].to_numpy()
print(f"\nArray 2D creato con dimensioni: {quantita_prezzo_np.shape}")
print(f"\nPrime 3 righe dell'array [quantita, prezzo_unitario]:\n")
print(quantita_prezzo_np[:3])

# Calcola l'incasso moltiplicando elemento per elemento:
# - quantita_prezzo_np[:,0] seleziona tutte le quantità (prima colonna)
# - quantita_prezzo_np[:,1] seleziona tutti i prezzi unitari (seconda colonna)
incasso_np = quantita_prezzo_np[:,0] * quantita_prezzo_np[:,1]
print(f"\nIncasso calcolato con operazione vettoriale numpy")
print(f"\nPrimi 5 incassi calcolati: {incasso_np[:5]}")

print("\n" + "-"*80)
print("VERIFICA EQUIVALENZA CON PANDAS")

# Confronta il risultato ottenuto con numpy con quello calcolato da pandas
# .all() restituisce True solo se tutti gli elementi sono uguali
risultato_confronto = (incasso_np == df_c['incasso'].to_numpy()).all()
print(f"\nGli incassi calcolati con numpy corrispondono a quelli di pandas? {risultato_confronto}")

print("\nFINE PARTE 4")
print("="*80)

# =======================================================================================================
# PARTE 5 - VISUALIZZAZIONE CON MATPLOTLIB
# =======================================================================================================

print("\n" + "="*80)
print("PARTE 5 - VISUALIZZAZIONE CON MATPLOTLIB")
print("="*80)

# Reset degli indici per trasformare gli indici in colonne normali
df_negozi = df_negozi.reset_index()
df_prodotti = df_prodotti.reset_index()
df_prod_neg = df_prod_neg.reset_index()
print("\nIndici dei dataframe resettati per la visualizzazione")

print("\n" + "-"*80)
print("GRAFICO 1: Incassi per negozio")

# Crea una figura con 2 subplot affiancati
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Incasso totale per negozio
bars1 = ax[0].bar(df_negozi['negozio'], df_negozi['incasso_totale'], 
          color='steelblue', edgecolor='black', linewidth=1.2, alpha=0.8)
ax[0].set_title('Incasso Totale per Negozio', fontsize=14, fontweight='bold')
ax[0].set_xlabel('Negozio', fontsize=11, fontweight='bold')
ax[0].set_ylabel('Incasso Totale (€)', fontsize=11, fontweight='bold')
ax[0].tick_params(axis='x', rotation=45, labelsize=10)
ax[0].tick_params(axis='y', labelsize=10)
ax[0].grid(alpha=0.3, linestyle='--', axis='y')

# Aggiunge i valori sopra ogni barra del subplot 1
for bar in bars1:
    height = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2., height,
               f'€{height:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

# Subplot 2: Incasso medio per negozio
bars2 = ax[1].bar(df_negozi['negozio'], df_negozi['incasso_medio'], 
          color='coral', edgecolor='black', linewidth=1.2, alpha=0.8)
ax[1].set_title('Incasso Medio per Negozio', fontsize=14, fontweight='bold')
ax[1].set_xlabel('Negozio', fontsize=11, fontweight='bold')
ax[1].set_ylabel('Incasso Medio (€)', fontsize=11, fontweight='bold')
ax[1].tick_params(axis='x', rotation=45, labelsize=10)
ax[1].tick_params(axis='y', labelsize=10)
ax[1].grid(alpha=0.3, linestyle='--', axis='y')

# Aggiunge i valori sopra ogni barra del subplot 2
for bar in bars2:
    height = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2., height,
               f'€{height:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "-"*80)
print("GRAFICO 2: Distribuzione incassi per negozio e per prodotto")

# Definisce palette di colori diverse per i due grafici a torta
colori_negozi = plt.cm.Set3(np.linspace(0, 1, len(df_negozi)))
colori_prodotti = plt.cm.Pastel1(np.linspace(0, 1, len(df_prodotti)))

# Crea una figura con 2 grafici a torta affiancati
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: Distribuzione incassi per negozio
ax[0].pie(
    df_negozi['incasso_totale'], 
    labels=df_negozi['negozio'], 
    autopct='%1.1f%%',
    colors=colori_negozi,
    shadow=True,
    textprops={'fontsize': 10, 'fontweight': 'bold'},
    pctdistance=0.85
)
ax[0].set_title('Distribuzione Incassi per Negozio', 
                fontsize=14, fontweight='bold')

# Subplot 2: Distribuzione incassi per prodotto
ax[1].pie(
    df_prodotti['incasso_totale'], 
    labels=df_prodotti['prodotto'], 
    autopct='%1.1f%%',
    colors=colori_prodotti,
    shadow=True,
    textprops={'fontsize': 10, 'fontweight': 'bold'},
    pctdistance=0.85    
)
ax[1].set_title('Distribuzione Incassi per Prodotto', 
                fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "-"*80)
print("GRAFICO 3: Andamento temporale degli incassi")

# Crea una figura con 2 grafici temporali affiancati
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Incasso giornaliero con media mobile settimanale linea di tendenza (media)
ax[0].plot(df_c['data'], df_c['incasso'], 
           label='Incasso giornaliero', color='steelblue',
           marker='o', linewidth=1.5, markersize=4, 
           markerfacecolor='red', markeredgecolor='black', 
           markeredgewidth=0.8)

ax[0].plot(df_c['data'], df_c['incasso_rolling'], 
           label='Media mobile settimanale', color='coral',
           linestyle='-', linewidth=2)

# Aggiunge una linea orizzontale per la media
media_incasso = df_c['incasso'].mean()
ax[0].axhline(y=media_incasso, color='red', linestyle='--', linewidth=1.5, 
              label=f'Media: €{media_incasso:.2f}', alpha=0.7)

ax[0].set_title('Incasso Giornaliero', fontsize=14, fontweight='bold', pad=15)
ax[0].set_xlabel('Data', fontsize=11, fontweight='bold')
ax[0].set_ylabel('Incasso (€)', fontsize=11, fontweight='bold')
ax[0].legend(loc='best', fontsize=10)
ax[0].grid(alpha=0.3, linestyle='--', axis='both')
ax[0].tick_params(axis='x', rotation=45, labelsize=9)
ax[0].tick_params(axis='y', labelsize=10)
ax[0].xaxis.set_major_locator(plt.MaxNLocator(15))

# Subplot 2: Incasso cumulativo
ax[1].fill_between(df_c['data'], df_c['incasso_cumulativo'], 
                   alpha=0.5, color='lightgreen', linewidth=0)
ax[1].plot(df_c['data'], df_c['incasso_cumulativo'], 
           color='darkgreen', linewidth=2.5, label='Incasso Cumulativo')

# Aggiunge una linea per il valore finale
valore_finale = df_c['incasso_cumulativo'].iloc[-1]
ax[1].axhline(y=valore_finale, color='red', linestyle='--', linewidth=2, 
              label=f'Valore finale: €{valore_finale:.2f}', alpha=0.7)

ax[1].set_title('Incasso Cumulativo', fontsize=14, fontweight='bold', pad=15)
ax[1].set_xlabel('Data', fontsize=11, fontweight='bold')
ax[1].set_ylabel('Incasso Cumulativo (€)', fontsize=11, fontweight='bold')
ax[1].tick_params(axis='x', rotation=45, labelsize=9)
ax[1].tick_params(axis='y', labelsize=10)
ax[1].grid(alpha=0.3, linestyle='--', axis='both')
ax[1].legend(loc='best', fontsize=10)
ax[1].xaxis.set_major_locator(plt.MaxNLocator(15))

plt.tight_layout()
plt.show()


print("\nFINE PARTE 5")
print("="*80)

# =======================================================================================================
# PARTE 6 - ANALISI AVANZATA
# =======================================================================================================

print("\n" + "="*80)
print("PARTE 6 - ANALISI AVANZATA")
print("="*80)

# Dizionario che mappa ogni categoria ai suoi prodotti
mappa_categorie = {
    'Informatica': ['Smartphone', 'Laptop', 'Tablet'],
    'Accessori': ['Webcam', 'Mouse', 'Tastiera'],
    'Elettrodomestici': ['TV', 'Lavastoviglie', 'Frigorifero']
}

print("\nMappatura categorie definita:\n")
for categoria, prodotti in mappa_categorie.items():
    print(f"   - {categoria}: {', '.join(prodotti)}")


# Per ogni prodotto nella lista originale, trova la categoria corrispondente
# utilizzando il dizionario di mappatura
categorie_dataset = [
    categoria 
    for prodotto in prodotti_dataset 
    for categoria in mappa_categorie.keys() 
    if prodotto in mappa_categorie[categoria]
]

print(f"\nCategorie assegnate a tutte le {len(categorie_dataset)} vendite")


# Aggiunge una nuova colonna 'categoria' al dataframe
df_c['categoria'] = categorie_dataset 

print(f"\nColonna 'categoria' aggiunta al dataframe")

print("\n" + "-"*80)
print("STATISTICHE PER CATEGORIA")

# Raggruppa i dati per categoria e calcola:
# - incasso_totale
# - quantita_media
df_categoria = df_c.groupby('categoria').agg(
    incasso_totale = ('incasso', 'sum'),
    quantita_media = ('quantita', 'mean')
).round(2).reset_index()

# Stampa i risultati dell'aggregazione
print("\nIncasso totale e quantita' media per categoria:\n")
print(df_categoria)

print("\n" + "-"*80)
print("SALVATAGGIO DATI ANALIZZATI")

# Inizializza il file manager per salvare il dataframe completo
vendite_analizzate = pandas_csv('vendite_analizzate')

# Salva il dataframe con tutte le colonne aggiunte (incasso, categoria, ecc.)
vendite_analizzate.save(df_c)

print(f"\nDataframe analizzato salvato nel file 'vendite_analizzate.csv' nella cartella corrente")

print("\nFINE PARTE 6")
print("="*80)

# =======================================================================================================
# PARTE 7 - ESTENSIONI
# =======================================================================================================

print("\n" + "="*80)
print("PARTE 7 - ESTENSIONI")
print("="*80)


print(f"\n{'>'*10} Normalizzazione dati per analisi percentuale...")

# Calcola la percentuale di incasso per ogni categoria rispetto al totale
df_categoria['incasso_percentuale'] = df_categoria['incasso_totale'].transform(
    lambda x: x/x.sum()
)

# Calcola la percentuale di quantità media per ogni categoria rispetto al totale
df_categoria['quantita_percentuale'] = df_categoria['quantita_media'].transform(
    lambda x: x/x.sum()
) 
print("Colonne percentuali aggiunte al dataframe categorie")

print("\n" + "-"*80)
print("GRAFICO: Confronto incasso e quantita' per categoria")

plt.figure(figsize=(14, 7))

# Crea il grafico a barre per l'incasso percentuale
bars = plt.bar(df_categoria['categoria'], df_categoria['incasso_percentuale'], 
               alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5, 
               label='Incasso Percentuale')

# Aggiunge i valori percentuali sopra ogni barra
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + 0.75 * bar.get_width(), height,
             f'{height*100:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Crea il grafico a linee per la quantità media percentuale
line = plt.plot(df_categoria['categoria'], df_categoria['quantita_percentuale'], 
                color='coral', marker='o', linewidth=3, markersize=10, 
                markerfacecolor='black', markeredgecolor='red', 
                markeredgewidth=2, label='Quantita Media Percentuale')

# Configurazione estetica del grafico
plt.title('Confronto Incasso e Quantita Media per Categoria', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Categoria', fontsize=12, fontweight='bold')
plt.ylabel('Percentuale', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3, linestyle='--', axis='y')
plt.legend(loc='best', fontsize=11, framealpha=0.9)
plt.tick_params(axis='x', labelsize=11)
plt.tick_params(axis='y', labelsize=11)
plt.tight_layout()
plt.show()

print("\n" + "-"*80)
print("ANALISI TOP PRODOTTI PIU' VENDUTI")

def top_n_prodotti(n, df):
    """
    Identifica i primi N prodotti più venduti in base al numero di vendite.
    
    Args:
        n (int): Numero di prodotti da restituire
        df (DataFrame): Dataframe contenente i dati delle vendite
    
    Returns:
        Series: Serie con i prodotti e il conteggio delle vendite, ordinati
    """
    # Raggruppa per prodotto, somma gli incassi e ordina in modo decrescente
    n_prodotti = df.groupby('prodotto')['incasso'].sum().sort_values(ascending=False).head(n)
    return n_prodotti

# Trova i top 5 prodotti più redditizi
n_top = 5
prodotti_top = top_n_prodotti(n_top, df_c)

print(f"\nTop {n_top} prodotti per incasso:")
print("-"*40)
for i, (prodotto, incasso) in enumerate(prodotti_top.items()):
    print(f"{i+1}. {prodotto}: €{incasso:.2f} ")
print("-"*40)

print("\nFINE PARTE 7")
print("="*80)