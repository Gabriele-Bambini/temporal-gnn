# RA-FM Fase 3 — drift strutturato con gate di attività (spec, 2026-09-18)

Stato: disegno pronto, implementazione rimandata per decisione del committente (prima consolidare il passo 1 sui dati reali, Fase 2). Codice di riferimento: `cfm_grn_v2` v0.1.7 (motore con cammini spline e clock `duration`, gate G1 superato).

## Idea in una frase

Il campo di velocità v(x, t) è una rete regolatoria firmata su un supporto prior (regolatore → bersaglio), in cui ogni arco ha un'attività m_ji(t, c) ∈ (0, 1) a basso rango. Il tempo biologico assoluto entra SOLO attraverso i gate; il resto del campo è autonomo. I gate sono l'oggetto scientifico: dicono quali regolazioni agiscono quando.

## Modello (v0, sintetico)

- Regolatore j → bersaglio i solo se `support[j, i]` (prior fissato prima del test; sintetico: dal simulatore; reale: CollecTRI/DoRothEA firmati, SCENIC+/CellOracle da motivi+ATAC; GRNBoost2 solo come baseline).
- Attività del regolatore: ψ_j(x_j) = sigmoid(softplus(a_j)·x_j + b_j) — monotona, saturante, per gene (famiglia di Hill senza fissarne l'esponente).
- Messaggio: s_i = Σ_j support_ji · sign_ji · softplus(w_ji) · m_ji(t, c) · ψ_j(x_j); segno dal prior; dove il prior non ha segno, tanh(θ_ji) appreso.
- Velocità: v_i = softplus(γ_i) · softplus(β_i + s_i) − softplus(λ_i) · x_i + κ_i (produzione − decadimento, nello spazio standardizzato).
- Gate: m_ji(t, c) = sigmoid(c_ji + E_ji · z(t, c)), z = MLP([t/scala, embedding condizione]) ∈ R^r, r ∈ {2, 4}; `gate_mode` ∈ {none (m=1), static (E=0), dynamic}. Inizializzazione: gate aperti (c = 2), E ≈ 0 → si parte da una rete statica.
- Regolarizzatori (griglia): L1 sull'attività media, variazione totale in t su una griglia di 16 tempi (pochi cambi netti).
- Training: cammini spline (v0.1.6) — il campione (x, v, u) dà anche il tempo biologico u da passare ai gate. Predizione: integrazione sull'orizzonte con t = t0 + s.
- Solo `path="spline"` + `clock="duration"` in v0.

## Controlli strutturali (oltre a quelli di Fase 1)

`grn_static_gates` (E=0), `grn_no_gates` (m=1), `grn_random_prior` (rewiring che conserva i gradi in entrata e in uscita e i segni: la lezione dell'audit), `mlp_no_graph` (motore di Fase 1), tempo rimescolato.

## Metriche di recupero dei gate (verità sintetica: `activity[t, e]` su 73 archi, 42 con switch)

- `gate_aupr`: average precision dell'escursione appresa max_t m − min_t m come punteggio per "arco che cambia" (verità: escursione > 0.5); gate costanti → AUPR = prevalenza (caso).
- `gate_corr`: correlazione media fra m_e(t) e attività vera sugli archi che cambiano.
- `gate_direction_acc`: segno del cambio (accensione/spegnimento) corretto.
- `gate_switch_mae`: errore sul tempo dello switch (tempo del cambio massimo fra tempi osservati consecutivi).

## Gate G3 (dal piano)

Recupero del rewiring sopra il caso (AUPR > prevalenza con IC95), ED di interpolazione: gate dinamici < gate statici < prior randomizzato e ≤ motore senza grafo (IC95 appaiati), tempo rimescolato peggiore. Fallback: se il recupero fallisce ma la previsione regge → non identificabile dai soli marginali: aggiungere ancore (osservazioni simulate di m, poi ATAC reale).

## File previsti

`data/prior.py` (GRNPrior: support, signs, from_weights, randomized degree-preserving, to/from_dict), `models/grn_field.py` (GatedGRNField, `uses_bio_time=True`, `gates(t, c)`, `gate_regularizers`), `evaluation/gates.py` (gate_recovery), modifiche a config (field, gate_mode, gate_rank, gate_l1, gate_tv), trainer (build_networks con prior, spline_loss con bio_time, predict con offset t0, checkpoint con prior, `FlowModel.gates`), search (prior e controlli strutturali, colonne gate_*), report (CONTROLS, tabella gate), runner (`--tasks gates`, griglia GATE_GRID 16 config). Versione 0.2.0.
