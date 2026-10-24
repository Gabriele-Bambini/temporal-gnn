# TGL-GRN Continuous — design del prototipo

## Obiettivo

Costruire un progetto nuovo e indipendente che conservi la domanda scientifica utile del lavoro di Sottoriva — prevedere come cambiano espressione e ruolo regolatorio dei geni lungo una GRN temporale — usando gli stessi dataset distribuiti `gastrulation` e `pancreas`, ma senza riutilizzare il loro codice.

Il prototipo deve stabilire rapidamente se nei file aggregati è rimasto un segnale temporale recuperabile. Non deve presentare l'aggregato come dato single-cell e non deve attribuire al modello informazione già eliminata dal binning originale.

## Cosa viene conservato

- geni come nodi identificabili nel tempo;
- GRN diretta e pesata come struttura biologica;
- sequenza di snapshot lungo pseudotempo;
- previsione di quantità future per-gene;
- interesse per hub e turnover regolatorio;
- confronto fra modelli statici e temporali;
- uso degli stessi file `gastrulation` e `pancreas`.

## Cosa viene corretto

1. Gli snapshot da 5 e 7 cellule non vengono trattati come affidabili quanto quelli da centinaia di cellule. `n_cells_bin` diventa confidenza, non feature predittiva.
2. Il campo `sum`, fortemente dipendente dal numero di cellule, viene escluso dal modello principale; rimane disponibile per diagnostica.
3. Il modello non predice la semplice presenza binaria degli archi, task dominato dai gradi. Predice variazioni continue dello stato dei geni e delle quantità strutturali.
4. Il target futuro è costruito esplicitamente dallo snapshot seguente. Le ultime colonne nulle dei target distribuiti non vengono usate.
5. La memoria temporale riceve gradiente. Il confronto principale è fra un aggiornamento diretto graph-temporal e un campo continuo addestrato con Flow Matching.
6. Training, validazione e test sono separati per snapshot interi e ordinati nel tempo.
7. Persistenza, momentum, modello senza grafo e ordine temporale rimescolato sono controlli obbligatori.

## Unità modellata

I file distribuiti non contengono cellule ma statistiche per-gene aggregate per bin. I geni, al contrario delle cellule, hanno identità esattamente allineate fra snapshot. Per questo il matching corretto è gene-a-gene; applicare OT per permutare i geni sarebbe biologicamente errato.

Per ogni gene viene costruito uno stato continuo con:

- media, mediana, deviazione standard e frazione non-zero dell'espressione;
- out-strength, in-strength e PageRank ricalcolati dalla GRN sorgente ridotta;
- pseudotempo `bin_mid`;
- confidenza derivata da `n_cells_bin`.

La somma di espressione non entra nello stato principale.

## Dati e riduzione della GRN

I CSV degli archi sono troppo grandi per la memoria disponibile: circa 25,9 milioni di archi per gastrulation e 11,3 milioni per pancreas. Un lettore streaming conserva, per ogni timestamp, i migliori target per ciascun regolatore e salva una cache compatta. Questa riduzione per-sorgente preserva la copertura dei regolatori meglio di un top-K globale. Il lettore non assume che l'intero file entri in RAM.

Ogni snapshot usa pesi normalizzati robustamente e soltanto il grafo osservato al tempo sorgente. Il futuro non viene usato nel message passing.

## Recupero causale del segnale

Prima del modello viene creato un contesto causale confidence-aware. Per lo snapshot `t` possono contribuire soltanto `t` e gli snapshot precedenti. La finestra si allarga finché raggiunge una massa cellulare minima, con peso maggiore agli snapshot vicini e affidabili. Una diagnostica di change point impedisce di lisciare attraverso un cambiamento netto.

Le osservazioni grezze dello snapshot futuro rimangono il target. Il denoising agisce sull'input e non riscrive la risposta attesa.

## Modelli

### Direct Temporal GNN

Un encoder diretto e pesato aggrega separatamente messaggi entranti e uscenti. Una GRU per-gene riceve una finestra di 3–4 rappresentazioni e viene addestrata con BPTT reale. Il campo predice la derivata dello stato e applica un passo di ampiezza `delta_s`:

`x_next = x + delta_s * v_theta(x, s, G_s)`.

### Graph Flow Matching

Per una coppia osservata gene-a-gene `(x_s, x_next)`, viene campionato un punto intermedio e il modello apprende la velocità per unità di pseudotempo, condizionata sullo stesso stato GRU usato dal modello diretto. In inferenza la funzione viene integrata da `s` a `s + delta_s`. Encoder, memoria, hidden size, seed e budget di ottimizzazione restano uguali fra i due modelli.

### Controlli

- persistenza;
- momentum lineare;
- Flow Matching senza grafo;
- Graph Flow Matching con ordine temporale rimescolato;
- opzionale grafo rimescolato preservando il grado.

## Valutazione

Il test usa gli ultimi snapshot completi e non uno split di nodi nello stesso futuro. Le metriche primarie sono:

- correlazione Spearman dei delta per-gene;
- errore normalizzato sullo stato futuro;
- precisione sui geni con maggiore cambiamento, con enfasi sulla media `log1p`;
- precisione del turnover dei top hub/PageRank;
- vantaggio rispetto alla persistenza;
- vantaggio dell'ordine reale rispetto al controllo temporale rimescolato.

Il verdetto è prudente:

- `signal_recovered`: il modello supera persistenza e null temporale in modo coerente su entrambi i dataset o su più transizioni di test;
- `weak_or_task_specific`: il vantaggio è confinato a una metrica, un dataset o pochi seed;
- `not_recovered`: persistenza o null eguagliano/superano il modello.

Nessuno di questi risultati dimostra una dinamica cellulare o una GRN causale, perché i dati originali a livello di cellula non sono disponibili.

## Interfaccia

Il comando principale sarà:

```text
python -m tgl_grn_continuous run \
  --data-root ../tgl-grn-1CCD/GRN \
  --datasets pancreas gastrulation \
  --quick
```

Produrrà cache compatte, metriche JSON, un report Markdown e figure essenziali senza modificare i repository originali.
