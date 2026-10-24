# Fonti per NotebookLM: flow matching, varianti, perturbazioni e reti regolatorie

Selezione del 12 settembre 2026 per preparare una bozza di progetto da discutere con Pietro Liò. I numeri corrispondono alle righe di NOTEBOOKLM_FONTI_FLOW_MATCHING_URL.txt. Il file URL è una lista da copiare nell'importazione delle fonti, non contiene il testo degli articoli. L'importazione in NotebookLM non è stata eseguita o verificata.

Per il primo giro, partire da **1, 2, 3, 5, 8, 10, 12, 14**. Poi ampliare con le altre fonti. Preferire il testo completo o il PDF all'abstract. NotebookLM supporta URL PDF e caricamenti PDF; una pagina HTML importa il testo della pagina specificata, non automaticamente le pagine collegate: [guida ufficiale](https://support.google.com/notebooklm/answer/16215270?hl=it).

## Basi e varianti matematiche

1. **Lipman et al. — Flow Matching Guide and Code (2024).** Guida per costruire la mappa generale: distribuzioni, campi di velocità, percorsi probabilistici, obiettivi e sampling. [PDF](https://arxiv.org/pdf/2412.06264).
2. **Lipman et al. — Flow Matching for Generative Modeling (ICLR 2023).** Articolo fondativo: perché si può apprendere il campo tramite regressione e come si genera integrando il modello. [PDF](https://arxiv.org/pdf/2210.02747).
3. **Tong et al. — Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport (TMLR 2024).** Conditional flow matching, accoppiamento fra campioni e OT-CFM; cruciale per distribuzioni cellulari non appaiate. [PDF](https://arxiv.org/pdf/2302.00482).
4. **Liu, Gong e Liu — Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (ICLR 2023).** Interpolazioni, rectification/reflow e riduzione del costo di generazione. [PDF](https://arxiv.org/pdf/2209.03003).
5. **Tong et al. — Simulation-Free Schrödinger Bridges via Score and Flow Matching (AISTATS 2024).** [SF]²M: connessione fra flow matching, score matching, trasporto entropico e dinamiche stocastiche da campioni non appaiati. [PDF ufficiale](https://proceedings.mlr.press/v238/tong24a/tong24a.pdf).
6. **Gat et al. — Discrete Flow Matching (preprint iniziale 2024).** Modelli per variabili discrete e tassi di transizione, distinti dai campi continui per l'espressione. [PDF](https://arxiv.org/pdf/2407.15595).
7. **DeFoG: Discrete Flow Matching for Graph Generation (preprint iniziale 2024).** Applicazione del flow discreto alla generazione di grafi. Serve a distinguere generare archi discreti dall'usare un grafo dentro un generatore continuo. [PDF](https://arxiv.org/pdf/2410.04263).

## Applicazioni al progetto

8. **Klein et al. — CellFlow enables generative single-cell phenotype modeling with flow matching (preprint 2025).** Previsione di distribuzioni cellulari condizionate da perturbazioni complesse. Questo è il CellFlow di Klein, DOI 10.1101/2025.04.11.648220, distinto da lavori omonimi sulla morfologia cellulare. [PDF ETH](https://www.research-collection.ethz.ch/server/api/core/bitstreams/4f42b78a-5f94-4234-a8b7-644864ce7137/content), [articolo](https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1).
9. **Roohani, Huang e Leskovec — Predicting transcriptional outcomes of novel multigene perturbations with GEARS (online 2023, Nature Biotechnology 2024).** Uso di grafi biologici come prior e distinzione fra diversi scenari di perturbazioni escluse. È un confronto GNN, non un modello di flow matching. [PDF](https://www.nature.com/articles/s41587-023-01905-6.pdf).
10. **Yu et al. — scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction (ICLR 2026).** Flow matching, MMD, attenzione differenziale e grafo di coespressione come maschera. Fonte centrale per il confronto con la proposta. [PDF](https://arxiv.org/pdf/2602.07103).
11. **Tong e Pang — FlowGRN: Scalable and Dropout-Robust Gene Regulatory Network Inference via Flow Matching-Based Trajectory Reconstruction (ACM-BCB 2025).** Ricostruzione delle traiettorie con flow/score matching, seguita da inferenza della rete con dynGENIE3. [PDF originale](https://orbilu.uni.lu/bitstream/10993/66745/1/BCB25.pdf).
12. **El Rimawi-Fine et al. — Simulation-free Structure Learning for Stochastic Dynamics / StructureFlow (versione completa preprint 2025).** Apprendimento congiunto di struttura e dinamiche condizionate. Studiare soprattutto grafo stazionario, maschere note di knockout e interventi esclusi: sono confronti essenziali per la novità. [PDF](https://arxiv.org/pdf/2510.16656).

## Valutazione critica

13. **Wu et al. — PerturBench: Benchmarking Machine Learning Models for Cellular Perturbation Analysis (NeurIPS 2025; preprint iniziale 2024).** Dataset, baseline, metriche e problemi di valutazione dei modelli perturbazionali. [PDF](https://arxiv.org/pdf/2408.10609).
14. **Viñas Torné et al. — Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation (online 2025, volume 2026).** Come distinguere effetti specifici delle perturbazioni da variazioni sistematiche che possono gonfiare le metriche. [PDF](https://www.nature.com/articles/s41587-025-02777-8.pdf).
15. **Ahlmann-Eltze et al. — Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines (Nature Methods 2025).** Per capire perché servono baseline forti e conclusioni circoscritte ai compiti e dataset effettivamente studiati. Non è una prova generale contro tutti i modelli generativi o il flow matching. [PDF](https://www.nature.com/articles/s41592-025-02772-6.pdf).

## Approfondimenti facoltativi

- **Chen e Lipman — Flow Matching on General Geometries (ICLR 2024).** Riemannian flow matching per spazi non euclidei; utile per ampliare la tassonomia, meno urgente per il primo prototipo. [PDF](https://arxiv.org/pdf/2302.03660).
- **Tong e Pang — FlowGRN+: Improving Gene Regulatory Network Inference by Spline Fitting and Manifold Projection in Conditional Flow Matching (technical report 2026).** Estensione di FlowGRN da leggere dopo la versione originale. [PDF](https://arxiv.org/pdf/2608.10407).

## Prompt per la spiegazione

Agisci come un docente che mi prepara a discutere un progetto PhD con Pietro Liò. Conosco machine learning e GNN, ma voglio costruire una comprensione solida del flow matching. Usa le fonti caricate, cita i passaggi pertinenti e segnala le informazioni non coperte.

1. Parti da un esempio intuitivo di popolazioni cellulari controllo e perturbate. Spiega distribuzione, campo di velocità, percorso probabilistico, coupling, addestramento e generazione. Poi introduci gradualmente le equazioni e definisci ogni simbolo.
2. Confronta flow matching, conditional flow matching, OT-CFM, rectified flow, score-and-flow matching/Schrödinger bridges e discrete flow matching. Se è caricata la fonte pertinente, aggiungi Riemannian flow matching. Spiega quali sono formulazioni, quali scelte di percorso o coupling, quali tipi di spazio e quali aspetti possono essere combinati.
3. Distingui il condizionamento sugli endpoint usato nell'obiettivo CFM dal condizionamento biologico su perturbazione, dose o contesto. Distingui inoltre tempo del flow e tempo biologico, campioni non appaiati e traiettorie individuali osservate.
4. Confronta CellFlow, GEARS, scDFM, FlowGRN e StructureFlow in una tabella: input, output, ruolo del grafo, dati temporali richiesti, obiettivo, generalizzazione testata e limiti.
5. Usa PerturBench, Systema e lo studio sulle baseline lineari per spiegare come valutare una proposta senza confondere prestazione media, risposta specifica, qualità distribuzionale e recupero di meccanismi.
6. Valuta questa ipotesi: apprendere maschere di attività contestuale su un prior TF-gene all'interno di un generatore di risposte cellulari. Separa ciò che esiste già, ciò che resta un'ipotesi di novità e ciò che richiede evidenza indipendente. Spiega come si potrebbe apprendere una maschera non osservata e quali problemi di identificabilità rimangono. Non dichiarare causalità o originalità assoluta senza supporto.
7. Concludi con dieci domande di autovalutazione e un ordine di studio realistico per tre giorni.

La roadmap e il report personale possono essere aggiunti come documenti di lavoro da criticare; non vanno trattati come evidenza scientifica a favore della proposta.
