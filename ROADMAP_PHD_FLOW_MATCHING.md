# Roadmap PhD: modelli generativi e attività regolatoria nelle risposte a perturbazioni

Preparata il 12 settembre 2026. Obiettivo immediato: bozza da discutere con Pietro Liò entro tre giorni. Le scelte qui proposte sono provvisorie e vanno confrontate con il supervisore; nessun risultato sperimentale è stato ottenuto per questa roadmap.

## Valutazione del progetto

Il progetto ha potenziale da PhD perché collega generalizzazione delle risposte cellulari, rappresentazioni regolatorie e verifica del loro significato biologico. L'originalità deve essere sostenuta dal problema e dagli esperimenti. Aggiungere un grafo al flow matching è già una direzione esplorata.

Domanda centrale: in quali condizioni l'attività regolatoria appresa migliora la previsione delle distribuzioni di risposta a perturbazioni escluse dall'addestramento, e quali evidenze permettono di interpretarla biologicamente?

Tre ipotesi da verificare separatamente:

1. **Generatore:** il conditional flow matching rappresenta le distribuzioni di risposta in modo utile rispetto a generatori alternativi e baseline semplici.
2. **Struttura:** un prior biologico e la sua modulazione contestuale migliorano la generalizzazione rispetto a modelli comparabili senza grafo o con grafo statico.
3. **Interpretazione:** parte delle attività apprese è stabile e supportata da evidenze indipendenti. Un vantaggio predittivo, da solo, non conferma questa ipotesi.

Titolo di lavoro: *Learning Context-Specific Regulatory Activity for Generalizable Single-Cell Perturbation Prediction*. Il sottotitolo può specificare *A conditional flow-matching approach*.

## Perché considerare flow matching

Il flow matching addestra un campo di velocità su percorsi probabilistici specificati, evitando l'integrazione del generatore nell'obiettivo standard di addestramento. La generazione richiede comunque un procedimento numerico; efficienza e vantaggio rispetto ad altri modelli vanno misurati. La flessibilità dei percorsi e la modellazione di distribuzioni sono motivazioni concrete per studiarlo [1].

CellFlow e scDFM offrono esempi pertinenti alla previsione di fenotipi e risposte single-cell [2,3]. Questi lavori sostengono la plausibilità della scelta, senza dimostrare che funzionerà nel nostro scenario. Il tempo computazionale del flow non è automaticamente tempo biologico; la variabilità dei campioni generati non è automaticamente incertezza epistemica calibrata.

L'evidenza sui modelli perturbazionali richiede attenzione: un benchmark ha trovato baseline semplici molto competitive sui compiti e dataset esaminati [5]. Systema mostra come variazioni sistematiche possano gonfiare le metriche senza catturare bene gli effetti specifici della perturbazione [6].

## Le prossime 72 ore

### Giorno 1: definire e difendere il problema

**Mattina, circa 3 ore**

- Confrontare scDFM, StructureFlow e FlowGRN/FlowGRN+ nelle colonne: dati, output, struttura appresa, condizionamento, tempo biologico, split e validazione degli archi.
- Leggere le sezioni metodologiche necessarie a spiegare la differenza proposta. StructureFlow usa già struttura e dinamica congiunte e maschere note per knockout ideali; il possibile spazio riguarda attività contestuali apprese e il loro sostegno indipendente [4].

**Pomeriggio, circa 3-4 ore**

- Scrivere problema, ipotesi e contributo potenziale in una pagina.
- Scelta iniziale consigliata: un sottoinsieme del dataset Replogle K562 per predire geni mai perturbati nel training, mantenendo un solo contesto cellulare. Confermare accesso, metadati, qualità delle perturbazioni e copertura del prior prima di fissare il protocollo.
- Distinguere gene mai perturbato da combinazione nuova di geni già osservati. Norman può servire successivamente per il secondo scenario, senza confondere le due conclusioni.
- Specificare come rappresentare un gene escluso: informazioni funzionali o di rete disponibili indipendentemente dalle sue risposte nel test. Un embedding categorico senza informazione trasferibile non risolve il problema.

**Consegna della giornata:** una pagina scientifica, una tabella di confronto e una scheda del dataset con le informazioni ancora da verificare.

### Giorno 2: rendere il metodo valutabile

**Mattina, circa 3-4 ore**

- Disegnare il modello minimo: popolazione di controllo, target della perturbazione, prior TF-gene, generatore condizionato e maschera di attività.
- Prima formulazione: A_eff(P,C) = A_prior × M_theta(P,C), con prodotto elemento per elemento e maschera continua regolarizzata. La maschera è condivisa al livello di condizione nel primo prototipo.
- Addestrare il generatore sull'espressione; apprendere la maschera indirettamente tramite l'obiettivo predittivo/distribuzionale. Definire vincoli, regolarizzazione e struttura dei messaggi. Non introdurre una loss di flow sugli archi priva di endpoint osservati o di un procedimento esplicito di inferenza latente.
- Chiamare la maschera un'ipotesi funzionale del modello. Il collegamento ad attività regolatoria è parte della ricerca.

**Pomeriggio, circa 3 ore**

- Scrivere split, baseline, metriche e controlli.
- Distinguere distribuzioni di cellule non appaiate da traiettorie osservate della stessa cellula.
- Stabilire i risultati che sosterrebbero o indebolirebbero ciascuna delle tre ipotesi.
- Preparare la figura del progetto e la tabella del primo anno.

**Consegna della giornata:** metodo leggibile, protocollo del primo esperimento, una figura e rischi con alternative.

### Giorno 3: scrivere la bozza per Liò

**Mattina, circa 4 ore**

Scrivere circa 1.700-2.000 parole, esclusi riferimenti, come obiettivo editoriale per questa discussione:

| Parte | Parole indicative |
|---|---:|
| Abstract | 150 |
| Problema e motivazione biologica | 200 |
| Lavori vicini e differenza proposta | 300 |
| Domande e ipotesi | 150 |
| Metodo e apprendimento | 400 |
| Valutazione e dati | 300 |
| Primo anno e programma successivo | 250 |
| Rischi e pertinenza al gruppo | 150 |

**Pomeriggio, circa 2-3 ore**

- Controllare tutte le citazioni, distinguendo articoli e preprint.
- Verificare la coerenza fra domanda, dati, metodo e metriche.
- Preparare una presentazione orale di cinque minuti.
- Preparare tre punti da discutere con Liò: contributo metodologico più interessante, contesto biologico prioritario, dati e collaborazioni realisticamente accessibili.

**Consegna finale:** bozza inglese, figura, tabella dei confronti, cronoprogramma e bibliografia verificata. Un risultato preliminare è facoltativo se dati e strumenti sono già pronti; non è un requisito di questa consegna. La roadmap non autorizza invii o contatti automatici.

## Primo anno: testare presto il contributo centrale

| Periodo | Attività | Risultato verificabile |
|---|---|---|
| Mesi 1-2 | Audit del dataset e del prior; split per gene perturbato; preprocessing dal training; baseline controllo, lineare e MLP. | Pipeline riproducibile, scheda dei dati, protocollo bloccato prima del test. |
| Mesi 3-4 | Prototipi piccoli di conditional flow matching e CVAE; primo confronto senza grafo, grafo statico e maschera appresa. | Primo test della novità e misure iniziali di qualità, stabilità e costo. |
| Mesi 5-6 | Controlli di capacità e prior randomizzati; curve rispetto a quantità di dati e qualità del prior. | Individuazione dei regimi in cui il grafo e il gating aiutano o falliscono. |
| Mesi 7-9 | Robustezza delle maschere, simulazioni con struttura nota, riferimenti biologici indipendenti. | Separazione fra beneficio predittivo e sostegno alle ipotesi regolatorie. |
| Mesi 10-12 | Repliche, valutazione finale esclusa dal tuning, eventuale secondo dataset compatibile e stesura. | Benchmark, implementazione documentata e manoscritto se i risultati sostengono un contributo. |

Il primo anno deve già produrre un verdetto sul nucleo del progetto. L'audit della letteratura continua in parallelo allo sviluppo; non occupa un anno intero.

## Il primo esperimento decisivo

1. **Scenario:** una sola linea cellulare; interi target di perturbazione esclusi dal training e dal tuning. Niente split casuale delle cellule della stessa perturbazione fra training e test per sostenere generalizzazione a nuovi interventi.
2. **Scala:** sottoinsieme gestibile di geni, indicativamente 500-2.000, preservando TF, target perturbati e copertura necessaria. Il numero definitivo dipende dal dataset e dalle risorse. La selezione non usa le risposte del test.
3. **Confronti:** generatori conditional flow matching e CVAE con input e rappresentazioni comparabili; per ciascuno, ove tecnicamente sostenibile, assenza di grafo, grafo statico e maschera appresa. Aggiungere prior randomizzati e baseline non generative.
4. **Equità:** budget di ricerca degli iperparametri, dimensione e capacità comparabili; documentare differenze residue. I soli risultati tra architetture molto diverse non isolano il contributo del flow matching.
5. **Esito primario:** distanza fra distribuzioni, per esempio energy distance, su una rappresentazione comune fissata dal training. Fissare dimensione dei campioni, normalizzazione e procedura prima del test. Le baseline deterministiche sono confrontate sulle statistiche che possono predire; eventuali versioni distributive vanno dichiarate.
6. **Esiti complementari:** effetti medi rispetto al controllo e discriminazione delle perturbazioni; analisi ispirata a Systema per verificare che il modello catturi effetti specifici [6]. Riportare costo e variabilità dei risultati.
7. **Struttura:** sul sintetico, recupero della struttura; sul reale, compatibilità con evidenze indipendenti e sensibilità alle fonti del prior. Una maschera sul prior non può recuperare archi assenti dal suo supporto.
8. **Repliche:** ripetere split e seed entro il budget. Quantificare variabilità fra perturbazioni e repliche biologiche quando disponibili; le cellule non sono repliche indipendenti della generalizzazione a interventi nuovi.

Non fissare una soglia arbitraria di miglioramento per dichiarare successo. Concordare la differenza praticamente rilevante usando rumore, variabilità delle baseline e costo, prima di esaminare il test finale. Gli approfondimenti suggeriti dai risultati restano esplorativi fino a una nuova verifica indipendente.

## Tre decisioni che guidano il seguito

| Decisione | Evidenza richiesta | Se l'evidenza manca |
|---|---|---|
| Il flow matching merita un ruolo centrale? | Qualità distribuzionale o compromesso qualità/costo utile rispetto al generatore alternativo. | Tenere la domanda biologica e usare il generatore più adatto; documentare i limiti del flow matching. |
| La struttura contestuale aggiunge valore? | Beneficio rispetto a grafo statico e controlli, con capacità comparabile. | Verificare i regimi già pianificati; se non emerge un vantaggio, ridurre il ruolo del gating. |
| Le maschere hanno significato biologico? | Stabilità e sostegno indipendente, oltre alle prestazioni. | Descriverle come pesi funzionali; valutare se moduli aggregati offrano ipotesi più robuste. |

Un esito negativo può essere scientificamente utile se è robusto e chiarisce condizioni e limiti. Non garantisce una pubblicazione.

## Anni successivi

**Anno 2: contributo metodologico.** Studiare identificabilità, robustezza rispetto al prior e incertezza, scegliendo una priorità sulla base del primo anno. Estendere a un secondo contesto o a combinazioni nuove. Introdurre una distribuzione latente delle maschere solo dopo avere definito inferenza e valutazione. Dati temporali diventano necessari se la domanda riguarda una sequenza biologica di attivazione.

**Anno 3: contributo biologico e tesi.** Selezionare un problema oncologico delimitato con una collaborazione e dati appropriati: per esempio risposta a una classe di farmaci in un sistema cellulare definito. Una risposta trascrizionale acuta non dimostra resistenza; questa richiede fenotipi, follow-up o condizioni resistenti adeguati. Validare su dati indipendenti e pianificare scrittura e consolidamento negli ultimi mesi.

**Eventuale quarto anno:** dipende dal percorso e dal finanziamento concordati. Può sostenere validazioni aggiuntive, ma i contributi essenziali non dovrebbero dipendere da un anno ancora non confermato.

## Cosa rende forte la proposta

Una domanda che rimanga importante anche se cambia la famiglia del generatore; un contributo riconoscibile rispetto ai lavori più vicini; un primo anno fattibile con dati pubblici; criteri espliciti per separare previsione e interpretazione. La novità complessiva resta da verificare con un audit mirato e con Liò.

## Fonti di riferimento

1. Lipman et al., *Flow Matching for Generative Modeling*. [Articolo](https://arxiv.org/abs/2210.02747).
2. Klein et al., *CellFlow enables generative single-cell phenotype modeling with flow matching*, preprint 2025. [Fonte primaria](https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1).
3. Yu et al., *scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction*, 2026. [Testo](https://arxiv.org/html/2602.07103v1).
4. El Rimawi-Fine et al., *Simulation-free Structure Learning for Stochastic Dynamics*, preprint 2025. [StructureFlow](https://arxiv.org/html/2510.16656v1).
5. Ahlmann-Eltze et al., *Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines*, Nature Methods, 2025. [Articolo](https://www.nature.com/articles/s41592-025-02772-6).
6. Viñas Torné et al., *Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation*, pubblicato online 2025, volume 2026. [Articolo](https://www.nature.com/articles/s41587-025-02777-8).
7. Wu et al., *PerturBench: Benchmarking Machine Learning Models for Cellular Perturbation Analysis*. [Articolo](https://arxiv.org/abs/2408.10609).
8. University of Cambridge, *Research Proposal*. Il limite indicato di 3.000 parole riguarda la candidatura formale, distinta dalla bozza per il supervisore. [Indicazioni](https://www.cst.cam.ac.uk/admissions/phd/research-proposal).
