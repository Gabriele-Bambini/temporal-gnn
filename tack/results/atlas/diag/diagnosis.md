# Perché l'OT non vince: domanda, dati o architettura? (2026-10-25, ESPLORATIVO)

Tutto qui è esplorativo: non cambia il verdetto pre-registrato (H1 0/7, H2 1/6).
Script: `diag_oracle.py` (locale, `diag_oracle.json`), `tack/hpc/diag_gap.py` (HPC job 718148, `diag_gap.json`;
gap 1 riproduce esattamente confirm.json). Pool = pool biologico di embrioni (colonna `pool` dei metadati: alcuni
"campioni" sono repliche tecniche dello stesso pool; pool indipendenti per stadio 3,1,3,2,5,2,2,2,4).

## Verdetto
1. **Dati / disegno (causa principale a 6 ore).** L'etichetta di stadio non è il tempo vero dell'embrione.
   - Orologio per pool (classificatore leave-one-pool-out sui 30 PC): Spearman 0.95 con lo stadio nominale, ma errore medio
     0.68 stadi; 14 pool su 24 sembrano più vicini a un altro stadio; dentro E7.75 i due pool distano 1.5 stadi (4.38 vs 5.88).
   - Un altro pool vero dello stesso stadio è più lontano del mix 50/50 dei vicini in 9/14 pool (atlante) e 6/6 (eritroidi):
     la differenza tra embrioni supera il cambiamento tra stadi.
   - L'età stimata del pool predice quale vicino gli somiglia (peso di miscela oracolo): Spearman 0.79, p = 0.0007, n = 14.
   - Pochi pool indipendenti, lotti di sequenziamento legati allo stadio (E7.25, E8.25 tutti lotto 2); E8.0-E8.5 sono tutti TS12.
2. **Domanda (a 6 ore è una domanda di miscela, non di movimento).**
   - Ripesando in modo ottimale le cellule degli stadi vicini (oracolo nel campione) lo stadio di mezzo si ricostruisce quasi
     perfettamente (ED_U <= 0 in tutti i gruppi dell'atlante): tutti gli stati esistono già nei vicini, manca solo il peso giusto.
   - Aggiungere punti trasportati dall'OT a questo oracolo non aiuta mai: contributo del trasporto -0.013..-0.018 (atlante),
     -0.004..-0.066 (eritroidi).
   - Il peso giusto dipende dall'età del singolo embrione (punto 1), non dalla sua etichetta.
3. **Architettura (secondaria, conta solo con salti grandi).**
   - Separazione centri/raggio cresce lentamente col salto: 0.36-0.54 (6 h), 0.56-0.78 (12 h), 0.76-0.98 (18 h), 1.05 (24 h).
     L'atlante intero a 24 ore si muove quanto gli eritroidi in 6.
   - ot_pairs (McCann senza rumore) non vince MAI, a nessun salto: stringe la nuvola.
   - Le versioni con rumore/crescita iniziano a battere il mix quando il salto cresce: uot 3/5 a 12 h; sb_half 3/3 a 18 h
     (-0.3 / -4.7 / -7.5 %) e 1/1 a 24 h (-7.5 %). Guadagni piccoli.
   - A 18-24 h un vero pool dello stesso stadio sta 3 volte più vicino di qualsiasi interpolazione (pavimento/mix 0.21-0.36):
     lì c'è spazio, ma i percorsi curvano (distanza dalla corda 0.28-0.42 raggi) e quasi tutti i tipi cellulari nascono
     (tipi condivisi con >= 50 cellule: da 3-12 a 1-2). Una retta tra due istantanee non basta.

## In una frase
A 6 ore il problema sono i dati e la domanda: embrioni dello stesso "stadio" hanno età diverse e lo stadio di mezzo è una
miscela di stati già presenti, quindi trasportare cellule non serve. L'architettura diventa il collo di bottiglia solo
con salti di 18-24 ore, e lì servirebbero percorsi curvi, crescita e rumore (non la retta McCann).

## Cosa seguirebbe (non lanciato)
- Cambiare variabile: tempo = età stimata dell'embrione, non lo stadio (modello a miscela per età + composizione).
- Dati: una sola linea, campionamento denso con età per embrione (somiti, o marcatura metabolica), più pool per stadio,
  lotti incrociati col tempo.
- Architettura, solo per salti larghi: ponte di Schrödinger multi-marginale con crescita (usa k-2..k+2), non coppie McCann.
