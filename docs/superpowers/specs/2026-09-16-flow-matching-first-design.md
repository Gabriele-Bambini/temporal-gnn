# Flow Matching First: design del nuovo prototipo

Data: 16 settembre 2026

## Decisione

Il nuovo progetto inizierà da un motore di Conditional Flow Matching indipendente dal grafo. La GNN, il prior regolatorio e le maschere sugli archi verranno aggiunti soltanto dopo avere dimostrato che il generatore di base è corretto, riproducibile e valutabile. Il codice esistente in `cfm_grn` rimarrà invariato come prototipo storico; la nuova implementazione vivrà in `cfm_grn_v2` e non copierà codice dai repository degli autori.

## Domanda del primo prototipo

Il primo prototipo deve rispondere a una domanda limitata: un modello di Conditional Flow Matching riesce a trasformare una distribuzione cellulare iniziale nella distribuzione osservata a una condizione o a un tempo successivo, senza usare informazioni del test e con prestazioni almeno competitive con controlli semplici?

Questa fase non sosterrà ancora che il modello impari una GRN, che il percorso generativo sia una traiettoria cellulare reale o che il Flow Matching sia superiore ad altre famiglie generative. Il suo scopo è produrre una base affidabile sulla quale tali ipotesi possano essere testate separatamente.

## Separazione dei due tempi

Il codice userà due variabili distinte. Il simbolo `biological_time` indicherà ore, giorni o stadi misurati nell'esperimento. Il simbolo `flow_time` indicherà la coordinata normalizzata fra zero e uno usata per costruire il probability path del Flow Matching. Nessuna visualizzazione, API o metrica potrà chiamare `flow_time` tempo biologico.

Per una transizione osservata da `biological_time=t0` a `biological_time=t1`, il modello costruirà un percorso generativo parametrizzato da `flow_time`. Il modello potrà ricevere l'intervallo biologico e la condizione come input, ma il percorso interno rimarrà una costruzione matematica.

## Ordine di sviluppo

### Fase 1: Flow Matching minimo e verificabile

La prima fase implementerà un probability path lineare, il target di velocità analitico, un campo vettoriale MLP condizionato e un risolutore ODE. Verranno supportati accoppiamento indipendente e accoppiamento Optimal Transport. L'OT sarà calcolato separatamente entro gruppi compatibili, definiti da condizione, replica e coppia di tempi; non potrà riordinare le cellule target senza riordinare con esse tutti i metadati associati.

La validazione iniziale userà distribuzioni bidimensionali e popolazioni sintetiche multivariate. Questi esempi dovranno permettere di controllare visivamente e quantitativamente che il modello recuperi traslazioni, cambiamenti di varianza, multimodalità semplice e risposte differenti fra condizioni.

### Fase 2: Flow Matching per snapshot single-cell

Dopo i test sintetici verrà aggiunta un'interfaccia dati comune basata su AnnData. L'interfaccia conserverà espressione, tempo biologico, perturbazione, replica, batch, specie e identificatore del campione. Il preprocessing verrà appreso esclusivamente sul training set e serializzato, affinché il test non contribuisca a selezione dei geni, normalizzazione, PCA o accoppiamento OT.

Il primo smoke test reale userà un dataset piccolo, come Klein GSE65525. Il primo benchmark temporale serio userà un singolo braccio di Waddington-OT oppure Hagai E-MTAB-6754. Per Hagai, la prova principale addestrerà sui tempi 0, 2 e 4 ore e manterrà le 6 ore completamente escluse; le unità biologiche, non le singole cellule, definiranno gli split.

### Fase 3: condizionamento trasferibile

Il condizionamento non verrà rappresentato soltanto da una colonna one-hot arbitraria quando si vuole valutare una condizione mai osservata. Per gli stimoli noti nel forecasting temporale sarà ammesso un embedding categoriale addestrato. Per la generalizzazione a nuovi target genetici servirà invece una rappresentazione costruita da informazioni disponibili anche per il target escluso, per esempio caratteristiche del gene, appartenenza a pathway o embedding derivati da un prior fissato senza usare le risposte del test.

Questa fase introdurrà Ding/Pollen GSE284197 soltanto quando tale rappresentazione sarà definita e verificata. Tutte le guide dello stesso TF escluso rimarranno fuori da training e tuning.

### Fase 4: estensione graph-aware

Solo dopo il superamento delle fasi precedenti verranno introdotti una GRN diretta e firmata, un message passing sparso e un gating regolatorio dipendente da tempo biologico, perturbazione e contesto. Il modello graph-aware sarà confrontato con lo stesso generatore senza grafo, con grafo statico e con prior randomizzati o degradati. Questa fase non fa parte della prima implementazione del motore Flow Matching.

## Struttura prevista del codice

`cfm_grn_v2` sarà un pacchetto installabile con una separazione netta fra dati, metodo e valutazione:

- `src/cfm_grn_v2/flows`: probability path, target di velocità, couplings e solver;
- `src/cfm_grn_v2/models`: campo vettoriale MLP e moduli di condizionamento;
- `src/cfm_grn_v2/data`: dataset sintetici, schema dei metadati e adapter AnnData;
- `src/cfm_grn_v2/training`: ciclo di training, configurazioni, checkpoint e riproducibilità;
- `src/cfm_grn_v2/evaluation`: metriche distribuzionali, controlli e report comuni;
- `tests`: test unitari, test di integrazione e test anti-leakage;
- `configs`: configurazioni versionate degli esperimenti;
- `outputs`: risultati derivati, esclusi dal codice sorgente e accompagnati dalla configurazione completa.

Le API principali non dipenderanno da CellFlow, scDFM, StructureFlow o FlowGRN. I confronti esterni useranno processi e ambienti separati, con adapter che convertono soltanto input e output in uno schema comune.

## Baseline e confronto equo

Prima dei repository esterni verranno implementati controlli semplici: persistenza, previsione della media del training, mean-shift condizionale, regressione lineare e MLP deterministico. Il Conditional Flow Matching verrà confrontato con queste baseline usando gli stessi geni, gli stessi split e lo stesso preprocessing.

CellFlow e scDFM saranno i confronti esterni primari per la generazione di popolazioni perturbate. StructureFlow e FlowGRN saranno confrontati soltanto su task compatibili con i loro obiettivi. RegVelo verrà usato successivamente come validazione biologica e non come concorrente diretto sullo stesso task.

Il confronto riporterà qualità predittiva, variabilità tra seed, tempo di training, tempo di inferenza, memoria e numero di parametri. Non verranno confrontati direttamente numeri pubblicati ottenuti su split diversi.

## Metriche

La valutazione primaria riguarderà l'intera distribuzione, usando energy distance e Maximum Mean Discrepancy con protocollo e numerosità dei campioni fissati prima del test. Verranno inoltre riportati Wasserstein marginale medio, errore sulle medie, correlazione degli effetti rispetto al controllo, recupero dei geni differenzialmente espressi e proporzioni di popolazioni quando disponibili.

Le metriche verranno calcolate sia nello spazio di espressione selezionato dal training sia in una rappresentazione di valutazione congelata. Una singola correlazione elevata non sarà considerata prova sufficiente di una buona distribuzione generata.

## Test obbligatori

I test dovranno verificare gli endpoint del probability path, la correttezza del target di velocità, la convergenza del solver, la coerenza fra metadati e permutazioni OT, l'assenza di accoppiamenti fra condizioni incompatibili, la riproducibilità dei seed e l'assenza di test data nel preprocessing.

Test di integrazione dovranno dimostrare che il modello apprende almeno tre trasformazioni sintetiche note e che una condizione non rappresentabile viene rifiutata con un errore esplicito invece di usare una colonna non addestrata. I test scientifici verificheranno inoltre che le metriche peggiorino sotto target permutati e che una baseline di persistenza venga sempre riportata.

## Gestione degli errori

Il caricatore dati dovrà fallire con messaggi espliciti quando mancano tempo, condizione, replica o geni richiesti. L'OT dovrà rifiutare gruppi vuoti o sbilanciamenti non gestibili. I checkpoint includeranno schema dei dati, geni, preprocessing, configurazione e versione del codice; un checkpoint incompatibile non verrà caricato silenziosamente.

In caso di memoria insufficiente, il sistema ridurrà in modo dichiarato il numero di cellule usate per il coupling o userà un coupling approssimato configurabile. Non verranno modificati silenziosamente gene set, split o metriche.

## Criteri di completamento della prima fase

La prima fase sarà completa quando il pacchetto verrà installato in un ambiente pulito, tutti i test passeranno, gli esperimenti sintetici saranno riproducibili da configurazione, il pairing OT sarà condition-aware e verrà prodotto un report che confronti Flow Matching con persistenza, mean-shift, regressione e MLP su almeno tre trasformazioni sintetiche.

Il passaggio ai dati reali richiederà inoltre che la pipeline separi correttamente training, validation e test a livello di campione. Il passaggio alla GNN richiederà un verdetto documentato sul Flow Matching senza grafo; un fallimento del generatore non verrà nascosto aggiungendo complessità strutturale.

## Artefatti previsti

Il lavoro produrrà il pacchetto `cfm_grn_v2`, una suite di test, configurazioni riproducibili, un report del benchmark sintetico, un primo benchmark reale e adapter separati per i repository ufficiali. La proposal formale verrà riscritta soltanto dopo il primo verdetto sperimentale e distinguerà chiaramente ciò che è stato dimostrato da ciò che costituisce lavoro futuro.
