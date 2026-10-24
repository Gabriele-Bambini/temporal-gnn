# Progetto di dottorato — scelta della nicchia e disegno (17 settembre 2026)

Documento di lavoro. Fonti verificate in `sources/LITERATURE_SWEEP_2026-09-17.md`. Nessun esperimento eseguito per questo documento. Le decisioni marcate **[DECISIONE]** spettano a te e vanno discusse con Pietro Liò.

## 1. Verdetto sul punto di partenza

Il paper Dileo–Sottoriva (arXiv 2605.28659) è un benchmark di temporal GNN su sequenze di GRN inferite per bin di pseudotempo. Tre fatti misurati o verificati lo rendono un punto di partenza da **reinquadrare**, non da estendere:

- I dati distribuiti sono GRN aggregate per bin (SCENIC), senza matrici cellula×gene: nessun modello generativo di popolazioni può essere addestrato o valutato su di essi.
- Con ricorrenza degli archi 0,87, EdgeBank fa già 0,88 AUPRC; la nostra autopsia mostra che i modelli "temporali" girano con stato nullo e che la ricorrenza vera non aiuta. Il segnale è la struttura del grafo, non la dinamica.
- Gli autori stessi ammettono che una ChebNet statica eguaglia il modello temporale e propongono come futuro solo un'altra architettura (DyGFormer).

Le sorgenti cell-level esistono ed hanno tempo reale: gastrulation eritroide = Pijuan-Sala 2019 (E-MTAB-6967, 9 stadi embrionali E6.5–E8.5); pancreas = Bastidas-Ponce 2019 (GSE132188). Il modo giusto di "modellare questo sistema con flow matching" è **ricostruire il problema a livello di cellula**, usare i bin di Dileo solo come benchmark secondario di rewiring, e trasformare la domanda da "prevedere lo snapshot successivo" a "apprendere quando e in quale contesto le interazioni regolatorie agiscono".

## 2. La nicchia aperta (convergenza di cinque sweep indipendenti)

Nessun metodo pubblicato al 17 settembre 2026 fa tutte queste cose insieme, e nessuno fa le prime due:

1. **Struttura regolatoria variabile nel tempo dentro un modello di flow/bridge.** StructureFlow, APPEX, SF²M apprendono una struttura statica; PFM (Flatiron, apr 2026) non ha struttura e dichiara "GNN su interazioni note" e "masked sparse layers" come lavoro futuro; DynNet/RegVelo/FLeCS usano una matrice unica; GRAVITY fa rewiring per stato ma senza tempo.
2. **Addestramento su dati interventistici con tempo-dopo-perturbazione** con drift parametrizzato da GRN. Esistono ora dataset adatti (RENGE 4 tempi post-KO; RAF1:ER 6/12/18 h; CD4 Rest/8h/48h a 22M cellule) e nessun generatore li usa così.
3. Crescita/morte (WFR-FM, BranchSBM) combinata con struttura.
4. Identificabilità dell'attività variabile da marginali + interventi (Guan 2026 chiude solo il caso a struttura fissa).
5. Valutazione: GGE (Rubbi, Di Francesco, Lotfollahi, **Liò**, mar 2026) dichiara la valutazione temporale come future work; nessun benchmark unisce recupero di struttura dinamica e fit distribuzionale; le sequenze di GRN non hanno protocolli persistence-aware.

Motivazione teorica indipendente: "Mechanisms Matter" (mag 2026) prova che tutti i virtual cell model falliscono quando le matrici di interazione divergono fra contesti. L'attività contestuale delle interazioni non è un dettaglio architetturale: è la variabile che decide la trasportabilità.

## 3. La tua idea, migliorata

Idea originale: `A_eff = A_prior ⊙ M_θ(x, t, c)` dentro un CFM controllo→perturbato. Critiche già emerse (12 set): la maschera non ha endpoint osservati, il tempo del flow non è tempo biologico, la novità rispetto a scDFM/StructureFlow non era netta, il prototipo perdeva contro la persistenza.

Versione aggiornata, in cinque mosse:

1. **Il generatore diventa una dinamica di popolazione, non un trasporto a endpoint.** Flow matching multi-marginale, simulation-free, con crescita (stile WFR-FM/PFM), su tempo biologico osservato. Il tempo del flow coincide con il tempo biologico solo perché i marginali sono ancorati a prelievi reali; altrimenti resta coordinata generativa e va dichiarato.
2. **La maschera diventa un processo di attività `m_ij(t, x, c) ∈ [0,1]`** sugli archi del supporto prior, parametrizzato a basso rango per scalare a migliaia di geni, con regolarizzazione di sparsità, di regolarità temporale e di regime (pochi cambi netti, non deriva continua). È l'oggetto scientifico del dottorato: **quando** un'interazione agisce.
3. **L'attività riceve osservazioni.** Tre ancore indipendenti risolvono la critica dell'endpoint mancante: (a) accessibilità dei motivi TF in multiome temporale come osservazione rumorosa dell'attività in uscita di un TF; (b) tempi di propagazione degli effetti di KO (RENGE, RAF1) come vincolo sull'attività degli archi a valle; (c) binding misurato (CUT&RUN atlas, ChIP-Atlas) come supporto. Nessuna delle tre da sola è verità causale; insieme rendono l'attività falsificabile.
4. **Gli interventi entrano come cambi di meccanismo** (clamp del gene perturbato e rimozione degli archi entranti), con propagazione lungo il grafo nel tempo biologico. Generalizzazione testata su interventi esclusi e su tempi esclusi, non su cellule rimescolate.
5. **Controlli che oggi mancano in letteratura:** persistenza, mean-shift, lineare, replay non parametrico (FreeFM di Lim 2026, che dimostra che il FM può limitarsi a ricombinare transizioni osservate), nessun grafo, grafo statico, prior randomizzato, tempo rimescolato, seed multipli con intervallo di confidenza.

Titolo di lavoro: **Regulatory-activity flow matching: learning when regulatory interactions act from time-resolved and perturbed single-cell populations** (sigla provvisoria RA-FM).

Domanda centrale: in quali condizioni l'attività temporale e contestuale delle interazioni regolatorie è apprendibile come parte di un generatore di dinamica cellulare, e quando migliora la previsione di popolazioni future, la generalizzazione a interventi e contesti non visti, e l'accordo con evidenze regolatorie indipendenti?

Tre ipotesi, testate separatamente:

- **H1 generatore.** Un flow matching multi-marginale con crescita e interventi supera persistenza, lineare e replay non parametrico su tempi terminali esclusi e su interventi esclusi.
- **H2 struttura.** Il drift sparso su supporto prior con attività `m(t,x,c)` supera nessun-grafo, grafo statico e prior randomizzato; l'attività variabile supera l'attività fissa specificamente sulle transizioni ad alto rewiring.
- **H3 interpretazione.** Le traiettorie di attività sono stabili fra seed e split, predicono evidenze cromatiniche e tempi di propagazione dei KO non usati in training, e sono identificabili sotto condizioni esplicitate; dove non lo sono, il modello riporta classi di equivalenza, non punti.

## 4. Confronto con i vicini più stretti

| Vicino | Cosa condivide | Cosa manca a loro |
|---|---|---|
| PFM (Flatiron 2026) | FM multi-marginale con crescita, interpretazione meccanicistica | nessun prior/GNN (loro future work), nessun intervento in training, solo interpolazione, 24 TF |
| StructureFlow (Tong/Zhang 2025) | struttura + interventi + SB multi-marginale | struttura statica, nessun prior, nessuna crescita, workshop-level |
| DynNet (Nat Commun 2026) | prior GRN come maschera in una SDE | 36 geni, quasi-stazionarietà, nessun fitting di perturbazioni |
| GRNvelo (MSB 2026) | GRN + crescita + KO | PINN deterministico, tempo latente, <100k cellule |
| GRAVITY (2026) | GRN cell-state-specific (rewiring) | nessun tempo né perturbazione |
| DynaVelo (Leslie 2026) | multiome, attività TF, validazione LOF/rescue | attività a livello nodo, non popolazione generativa, tempo latente |
| MixFlow/SP-FM (Liò, Lotfollahi 2026) | FM condizionato, OOD a condizioni non viste | statico; nessun tempo né struttura → **punto di aggancio naturale con il gruppo target** |
| Dileo–Sottoriva (2026) | GRN che cambia lungo lo sviluppo, hub | non generativo, aggregati per bin, pseudotempo, nessuna perturbazione |

Rischio principale: il gruppo PFM o i gruppi Theis/Tong possono aggiungere una GNN in pochi mesi. Difesa: velocità sul prototipo (H1+H2 su sintetico e RENGE entro 9 mesi), differenziazione su ancoraggio, interventi con tempo, teoria e benchmark, e collaborazione con Human Technopole sui dati.

## 5. Modello (bozza per la discussione)

Stato `x ∈ R^G` (log-normalizzato; variante su conteggi con Count-FM in anno 2), tempo biologico `t`, condizione `c` (intervento, contesto).

- Dinamica: `dX = f_θ(X,t,c) dt + Σ(X) dW`, crescita `g_θ(X,t,c)` (regressione congiunta velocità + crescita, simulation-free).
- Drift strutturato: `f_i(x,t,c) = φ_i( x_i , Σ_{j∈Pa(i)} m_ij(t,x,c) · w_ij · ψ(x_j) )` con `Pa(i)` dal supporto prior (SCENIC+/CollecTRI/ChIP-Atlas, fissato prima del test), `w_ij` pesi firmati appresi, `m_ij = σ(u_i(t,x,c)ᵀ v_j(t,x,c) + b_ij)` a basso rango.
- Regolarizzazione dell'attività: L1 su `m`, penalità di variazione totale in `t` (regimi), penalità di coerenza con le ancore cromatiniche quando disponibili.
- Interventi: `do(x_k = 0)` o clamp, archi entranti in `k` disattivati, effetti propagati in `t`.
- Addestramento: coupling OT multi-marginale sbilanciato entro gruppi compatibili (condizione, replica, coppia di tempi); nessun coupling fra condizioni diverse (errore già commesso nel primo prototipo).
- Inferenza: integrazione ODE/SDE da un marginale osservato a tempi non visti; campionamento di popolazioni; lettura delle traiettorie `m(t)` come ipotesi regolatorie.

Questo è un disegno da validare, non un risultato.

## 6. Dati, a livelli

- **Livello 0, verità nota:** scMultiSim in modalità GRN time-varying, dyngen, SERGIO/BoolODE, GRouNdGAN-Toolkit. Qui si misura il recupero dell'attività variabile e le condizioni di identificabilità.
- **Livello 1, tempo reale senza perturbazione:** Waddington-OT (39 tempi), Farrell zebrafish (12), gastrulation Pijuan-Sala (9 stadi reali: sorgente cell-level di Dileo), pancreas Bastidas-Ponce (pseudotempo, confronto con i bin di Dileo).
- **Livello 2, perturbazione × tempo:** RENGE GSE213069 (23 TF KO, 4 tempi), RAF1:ER GSE250559 (22 TF KO, 6/12/18 h), CD4 GSE314342 (genome-scale, Rest/8h/48h), Hagai E-MTAB-6754 (stimolo × 0/2/4/6 h), Ding/Pollen GSE284197 (44 TF, D0/D7).
- **Livello 3, ancoraggio cromatina:** Fleck 2023 organoidi cerebrali (11 tempi multiome + GLI3 KO; accessione da verificare), gastrulation multiome GSE205117, scTIE GSE223041, GC B cells di DynaVelo.
- **Livello 4, cancro (anno 3):** Watermelon GSE150949 (D0/3/7/14, 56.419 cellule, barcode di lineage), TraCe-seq, ReSisTrace; il benchmark DRMREF 2026 (26 dataset, 760k cellule) mostra che la predizione della resistenza pre-trattamento è a livello casuale: bersaglio irrisolto.
- **Validazione esterna:** RegVelo zebrafish NC + Perturb-seq; CUT&RUN atlas E10.5/E11.5.

Dato mancante in letteratura: un Perturb-seq genome-scale con ≥3 prelievi dopo l'inizio della perturbazione. Proporre la generazione (Cas12a degradabile + Flex) come collaborazione, non come requisito.

## 7. Valutazione (fissata prima del test)

- Distribuzioni: energy distance, MMD, W2 in PCA-50 e nello spazio DEG, con GGE (compatibilità diretta con il gruppo di Liò); perturbation-effect correlation.
- Tempo: forecast di tempi terminali esclusi (non interpolazione), coerenza degli embedding e fedeltà di lineage (scTimeBench).
- Struttura: AUPR e recall del rewiring contro verità sintetica; sulle sequenze reali, task di **rewiring puro** (nascite/morti di archi, turnover degli hub) con negativi storici e soffitto di persistenza per snapshot; PageRank invece di out-degree.
- Ancore: correlazione fra `m(t)` e accessibilità dei motivi / binding misurato, su dati non usati in training.
- Controlli: persistenza, mean-shift, lineare, FreeFM, nessun grafo, grafo statico, prior randomizzato o degradato, tempo rimescolato; ≥3 seed con intervallo di confidenza; costo e tempo di training riportati.
- Regola: nessuna soglia di successo arbitraria; la differenza praticamente rilevante si concorda prima del test finale usando rumore e variabilità delle baseline.

## 8. Struttura del dottorato (36 mesi + eventuale quarto anno)

| Periodo | Work package | Risultato verificabile |
|---|---|---|
| Mesi 1–6 | **WP0 Benchmark e valutazione.** Costruire *TempGRN-Bench*: verità sintetica time-varying + ancore reali (RENGE, RAF1, Fleck, GSE205117) + protocolli persistence-aware; estendere GGE con metriche temporali e strutturali in accordo con Rubbi/Lotfollahi. Prototipo RA-FM v0 su sintetico. | Pacchetto e nota di benchmark (workshop o Bioinformatics); verdetto preliminare su H1. |
| Mesi 4–15 | **WP1 Metodo.** RA-FM v1: multi-marginale + crescita + interventi + attività a basso rango; ablation completa; baseline esterne (PFM, WFR-FM, StructureFlow, DynNet, GRNvelo, CellFlow, scDFM) via adapter. Test primari: tempo terminale escluso (Waddington-OT, gastrulation reale) e interventi esclusi (RENGE, RAF1). | Articolo di metodo (ICLR/NeurIPS o rivista); verdetto su H1 e H2 entro il mese 12. |
| Mesi 12–24 | **WP2 Ancoraggio e identificabilità.** Ancore cromatiniche (multiome), test di trasportabilità fra contesti (CD4, Hagai multi-specie), teoria: condizioni di identificabilità dell'attività variabile da marginali + interventi, estendendo Guan 2026 e Zweig 2026; report di classi di equivalenza. | Articolo teoria+metodo; verdetto su H3. |
| Mesi 20–36 | **WP3 Applicazione.** Cancro con Human Technopole: dinamica dei persister (Watermelon con coupling vincolato dai barcode; ReSisTrace per pre-esistente vs indotto) oppure sviluppo (organoidi GLI3 KO). Tesi. | Articolo biologico; tesi. |

Gate decisionali: (mese 9) se H1 fallisce sul reale, la domanda resta e il generatore cambia famiglia; (mese 12) se l'attività variabile non batte l'attività fissa, ridurre il ruolo del gating e spostare peso su WP0/WP2; (mese 24) scegliere l'applicazione in base ai dati realmente accessibili.

## 9. Perché è forte per Liò e per Sottoriva

- Liò è coautore di GGE (mar 2026) e di MixFlow (gen 2026): il gruppo lavora esattamente su generatori di espressione e sulla loro valutazione, e ha dichiarato la valutazione temporale come lavoro futuro. Il WP0 è un contributo immediato e a basso rischio per il gruppo; RA-FM estende MixFlow con tempo e struttura. Liò pubblica inoltre su sheaf/topologia e su dinamiche: un'estensione con trasporto per-arco (sheaf) dell'attività è una direzione da tenere in riserva, non per il primo anno.
- Sottoriva (Human Technopole) ha due lavori direttamente adiacenti: Dileo–Sottoriva (sequenze di GRN) e NeuroVelo (Neural ODE + interazioni). Il WP3 e la ricostruzione cell-level dei loro dataset creano una collaborazione naturale sui dati e sull'evoluzione tumorale.

## 10. Rischi e risposte

| Rischio | Risposta |
|---|---|
| Scooping su "PFM + GNN" | prototipo entro 9 mesi; differenziazione su ancoraggio, interventi con tempo, teoria, benchmark |
| Il FM ricombina transizioni osservate (Lim 2026) | FreeFM come baseline; solo tempi terminali esclusi come test di forecast |
| Prior sbagliato o incompleto | prior randomizzati/degradati; supporto morbido con L1 fuori-supporto come ablation |
| Pochi tempi (3–4) nei dataset di perturbazione | sintetico + più dataset in pooling; proposta di dataset multi-tempo come collaborazione |
| Non identificabilità | teoria in WP2; riportare classi di equivalenza; ancore osservate |
| Calcolo | A100 Bocconi sufficiente fino a ~5k geni; attività a basso rango; simulation-free |
| Varianza fra seed dello stesso ordine delle differenze fra modelli (misurata nell'autopsia) | ≥3 seed, IC, differenze appaiate |

## 11. Alternative se non condividi la scelta **[DECISIONE]**

- **A, valutazione-prima (più sicura, allineata a Liò):** WP0 come primo anno intero (GGE temporale + TempGRN-Bench + critica persistence-aware delle temporal GNN), metodo dal secondo anno.
- **B, Sottoriva-prima:** dinamica dei persister con coupling vincolato dai lineage (Watermelon, ReSisTrace, DRMREF); GRN come prior secondario. Più biologia, meno metodo.
- **C, teoria-prima:** identificabilità di struttura variabile da marginali + interventi; esperimenti solo sintetici nel primo anno. Più rischio di isolamento dai dati.

Raccomandazione: la traccia principale (WP0→WP1→WP2→WP3) con A come fallback se il prototipo non dà verdetto entro il mese 9.

## 12. Prossime due settimane

1. **[DECISIONE]** confermare la traccia principale o un'alternativa.
2. Riscrivere la bozza inglese per Liò (≤2 pagine) con domanda, tre ipotesi, WP0 come contributo immediato al loro GGE, e la tabella dei vicini.
3. Prototipo secondo la spec "Flow Matching first" (`cfm_grn_v2`): motore multi-marginale sbilanciato + baseline FreeFM su Waddington-OT e Hagai; poi RENGE con interventi. Nessuna GNN prima del verdetto H1.
4. Scaricare e auditare RENGE (GSE213069) e RAF1:ER (GSE250559): tempi, cellule per condizione, controlli, copertura del prior.
5. Ricostruire la gastrulation cell-level (E-MTAB-6967) con stadi reali e confrontarla con i bin di Dileo: prima misura di quanto pseudotempo e tempo reale divergono.
6. Contatti: Rubbi/Lotfollahi per l'estensione di GGE; lab Sottoriva per dati e collaborazione. Opzionale: VCC 2026 (chiude 5 novembre) come stress test zero-shot, solo se non sottrae tempo al prototipo.

## 13. Research statement (bozza inglese, ~350 parole)

**Regulatory-activity flow matching: learning when regulatory interactions act from time-resolved and perturbed single-cell populations**

Cells change state by switching regulatory interactions on and off, yet current generative models of single-cell dynamics either ignore regulatory structure or assume a single static network over the whole process. Recent work shows that flow-matching models with biophysical drifts generalise better than unconstrained ones, that models with identical fit can encode different mechanisms, and that virtual cell models fail precisely when interaction mechanisms differ across contexts. The open question is whether the time- and context-dependent *activity* of regulatory interactions can be learned as part of a population-dynamics model, and whether it is identifiable and biologically supported.

I propose regulatory-activity flow matching (RA-FM): a simulation-free, multi-marginal flow-matching model with growth and interventions whose drift is a sparse, signed message-passing field over a prior transcription-factor–target support, modulated by a learned low-rank activity process m(t, x, c). Interventions enter as mechanism changes that propagate through the graph in biological time. Activity is anchored to independent observations, namely motif accessibility from temporal multiome data, knockout-propagation timing from time-resolved Perturb-seq, and measured binding atlases, so that learned activity is falsifiable rather than a free latent.

Three hypotheses are tested separately: that the generator forecasts genuinely unobserved terminal time points and held-out interventions better than persistence, linear and non-parametric replay controls; that prior-supported, time-varying activity improves over no-graph, static-graph and randomised-prior controls, specifically on rewiring-rich transitions; and that learned activity trajectories are stable, predictive of unseen chromatin and perturbation evidence, and identifiable under stated conditions.

The programme has four parts: a benchmark with ground-truth time-varying networks and persistence-aware rewiring tasks, extending the GGE evaluator with temporal and structural metrics; the method with full ablations against PFM, StructureFlow, DynNet, GRNvelo, CellFlow and scDFM; identifiability theory for time-varying activity from marginals and interventions; and an application to therapy-induced regulatory rewiring in cancer using lineage-barcoded drug time courses, in collaboration with Human Technopole. Public datasets with physical time and perturbations (RENGE, RAF1:ER, CD4 CRISPRi, temporal multiome, Watermelon) make the first two years feasible without new data.
