# Atlante di gastrulazione: i due segnali eritroidi si replicano? (2026-10-25)

> **TL;DR.** No. Il verdetto pre-registrato è confermato e nel codice non c'è nessun errore.
> Sull'atlante il semplice mix 50/50 dei due stadi vicini batte l'interpolazione OT in 7 stadi su 7 (6 sconfitte solide, 1 pareggio a E8.25).
> La previsione con ripesatura OT (uot_rw) vince 1 volta su 6. Il segnale eritroide era vero, ma solo per quei dati.
> La linea si chiude come fonte di risultati positivi e si scrive come risultato negativo e di metodo.

Tutto ciò che va oltre il ricalcolo del verdetto è **ESPLORATIVO**: aiuta a capire, non cambia il verdetto.
Fonti: `judge/judge.md` e gli script in `tack/results/atlas/judge/` (nomi tra parentesi).

## 1. Cosa abbiamo testato e perché

Sulla linea eritroide (cellule che diventano globuli rossi) avevamo trovato due segnali **a posteriori**, cioè guardando già i dati:
- **(a) Interpolare** (= ricostruire uno stadio nascosto partendo da quello prima e quello dopo) con il **trasporto ottimo, OT**
  (= accoppiare ogni cellula "di prima" con una "di dopo" simile, nel modo più economico) e mettere un punto a metà strada.
  Sugli eritroidi questo (`ot_pairs`) batteva il mix 50/50 in 3 stadi su 4: 1.908/1.287/0.942/1.198 contro 2.206/1.041/1.154/1.473.
- **(b) Prevedere** lo stadio di domani dando più peso alle cellule di oggi che l'OT vede in crescita (`uot_rw`), invece di "domani = oggi".

Un segnale trovato a posteriori può essere fortuna. L'unico modo onesto per saperlo: riprovarlo una volta, con regole fisse, su cellule nuove.

## 2. Come

- **Pre-registrazione** (= regole scritte e congelate *prima* di vedere i risultati): `tack/prereg/2026-10-25-atlas-confirmation-prereg.md`.
  La sua impronta (sha256 7cb27ba3…) era già in decisions.md prima del risultato. Le impronte dei 4 file di codice congelati combaciano (`j_verdict.py`, `j_products.py`).
- **Controllo di parità**: prima di congelare, lo stesso codice sui dati eritroidi ridava i vecchi numeri (`parity_erythroid.json`). Quindi era il codice che aveva vinto.
- **Cellule indipendenti**: atlante Pijuan-Sala 2019 (topo), 9 stadi da E6.5 a E8.5, uno ogni 6 ore, **senza** la linea del sangue (nessuna cellula eritroide riusata).
  Al massimo 3000 cellule per stadio. Pool di embrioni (= gruppi di embrioni sequenziati insieme) per stadio: 3,1,6,3,6,4,4,3,4.
- **Misura**: ED_U, "distanza di energia" = quanto due nuvole di punti si somigliano (0 = identiche, più basso = meglio),
  nello spazio PCA (= i 2000 geni riassunti in 30 assi principali).
- **Regola**: A batte B se è *strettamente* più basso in almeno 5 stadi su 7 (interpolazione) o 4 su 6 (previsione) **e** ha media più bassa, contro **ogni** rivale. Pareggio = sconfitta.
- **I rivali**: `prev` = stadio prima così com'è; `next` = stadio dopo; `union` = mix 50/50 dei due; `indep` = punto a metà tra coppie *a caso*;
  `uot` = OT che permette a cellule di nascere e morire; `persist` = domani uguale a oggi; `meanshift` = sposto tutto come all'ultimo passo; `tilt_rw` = ripesatura senza OT.

## 3. Risultati pre-registrati (definitivi)

| ipotesi | cosa dice | esito | vinte/pari/perse per rivale | media ED_U |
|---|---|---|---|---|
| H1 (principale) | ot_pairs batte prev, next, union | **FALLISCE** 0/7 (ne servono 5) | prev 6/0/1, next 6/0/1, union **0/0/7** | 0.6151 vs 1.2072 / 0.9369 / **0.5073** |
| H1b | uot batte prev, next, union | **FALLISCE** 2/7 (E8.0, E8.25) | prev 6/0/1, next 5/0/2, union 3/0/4 | 0.6149 vs union 0.5073 |
| H1c | ot_pairs batte indep | **PASSA** 7/7 | indep 7/0/0 | 0.6151 vs 1.0063 |
| H2 (principale) | uot_rw batte persist, meanshift | **FALLISCE** 1/6 (ne servono 4) | persist 1/3/2, meanshift 6/0/0 | 0.8154 vs persist 0.7708 |
| H2b | uot_rw batte tilt_rw | **FALLISCE** 2/6 | 2/2/2 | 0.8154 vs 0.7297 |

| stadio nascosto | E6.75 | E7.0 | E7.25 | E7.5 | E7.75 | E8.0 | E8.25 |
|---|---|---|---|---|---|---|---|
| union (mix 50/50) | **1.476** | **0.343** | **0.644** | **0.357** | **0.248** | **0.276** | **0.208** |
| ot_pairs (OT a metà strada) | 1.732 | 0.352 | 0.725 | 0.519 | 0.403 | 0.349 | 0.226 |
| indep (coppie a caso) | 1.978 | 0.772 | 1.249 | 1.011 | 0.888 | 0.607 | 0.539 |

| stadio da prevedere | E7.25 | E7.5 | E7.75 | E8.0 | E8.25 | E8.5 |
|---|---|---|---|---|---|---|
| persist (domani = oggi) | 0.500 | 1.640 | 0.468 | 0.767 | 0.912 | 0.339 |
| uot_rw (ripesatura OT) | 0.500 | 1.640 | 1.043 | 0.767 | **0.596** | 0.346 |

- L'OT batte "solo prima" e "solo dopo" in 6 stadi su 7. È il mix 50/50 a fermarlo, ogni volta.
- Anche nello spazio dei geni (misura secondaria) ot_pairs batte il mix 0/7: media 0.7156 contro 0.388 (`j_verdict.py`).
- A E7.25, E7.5 ed E8.0 uot_rw è *identico* a persist: la scelta automatica dei parametri ha deciso "non ripesare" (λ = 0).
  Se contassimo quei pareggi come vittorie (non è la regola) farebbe 4/6, ma H2 fallirebbe comunque sulla media (0.8154 contro 0.7708).

## 4. Cosa hanno trovato i controlli

**Bug? No.** Questo è l'unico controllo che potrebbe contare (e anche allora servirebbe una nuova pre-registrazione).
- Codice congelato rifatto girare senza toccarlo: tutti i 1492 numeri di confirm.json tornano, differenza massima 4.5e-13 (`j_diff.py`).
- I 5 verdetti ricalcolati con una regola riscritta da zero: identici (`j_verdict.py`). Quattro controllori separati (verdict, audit, floor, mechanism): nessun bug.
- Dati verificati cellula per cellula: stadi, cellule nascoste e pool giusti; la PCA non ha mai visto lo stadio nascosto (`j_products.py`).
- Unico punto debole: solo 3-8 geni di proliferazione tra i 2000, usati da uot per stimare la crescita. Non cambia nulla:
  con 96 geni uot batte il mix 2/7, senza stima di crescita 1/7; il mix resta 0.507 (`j_eps_cc.py`).

**È sfortuna? No.** *(ESPLORATIVO, `j_noise.py`)*
- Confronto rifatto contro 12 campioni diversi di cellule vere: OT contro mix resta 0/7 in 11 casi, 1/7 in uno.
- Su 220 estrazioni casuali l'OT vince nel 0 / 8 / 0 / 0 / 0 / 0.5 / 35 % dei casi (da E6.75 a E8.25).
- Togliendo un pool di embrioni alla volta il segno non cambia mai da E7.0 a E8.0.
- Bilancio: 6 sconfitte solide + 1 pareggio nel rumore (E8.25). Non è "per un pelo".
  Un vantaggio OT grande come quello eritroide (13-19 % sotto il mix) è escluso in 6 stadi su 7.
- "Pavimento di rumore" (= il mix era già bravo come un vero replicato, quindi non c'era spazio): **respinto**.
  Il mix sta circa 1.5 volte sopra un replicato vero in entrambi i dataset (mediana 1.51 atlante, 1.71 eritroidi; `j_lopo.py`),
  eppure sugli eritroidi l'OT vinceva. E "niente spazio" darebbe pareggi, non 6 sconfitte.

**Perché il mix vince.** *(ESPLORATIVO)* Immagina due nuvole di punti, "prima" e "dopo".
- Se si sovrappongono molto, lo stadio di mezzo è un **miscuglio**: alcune cellule ancora "come prima", altre già "come dopo".
  Il mix 50/50 è proprio questo. Mettere punti a metà strada stringe la nuvola e peggiora.
- Se sono lontane, il mix ha due gruppi con un buco in mezzo. Lo stadio vero sta nel buco, e lì i punti a metà strada aiutano.
- Il numero chiave: distanza tra i centri delle due nuvole diviso il raggio della nuvola di mezzo.
  Atlante 0.37-0.55, eritroidi 1.18-1.92: nessuna sovrapposizione (`j_restore.py`). È un'ipotesi nata qui (11 unità non indipendenti), non una scoperta.
- Prova senza modelli (`j_mixture.py`), con **oracolo** (= scelte fatte guardando la risposta, solo per capire): anche col tempo migliore l'OT perde
  contro il miglior miscuglio 0/7 sull'atlante. Sugli eritroidi l'OT a metà strada batte il miglior miscuglio 3/4.
- Altre spiegazioni, controllate:
  - *Nuvola OT troppo stretta*: vero ma parziale. Larghezza 0.911 contro 1.038 del mix; ridandogli larghezza vince solo 1/7 (E8.25, 0.195 vs 0.208) (`j_restore.py`).
  - *Tempi sbagliati*: lo stadio di mezzo spesso non è a metà (tempo migliore da 0.1 a 1.0), ma anche col tempo giusto l'OT perde (`j_mixture.py`).
  - *Tipi cellulari che nascono o spariscono*: solo 2.2-13.6 % delle cellule, e non separa i dataset (`j_comp.py`).
  - *Dentro un singolo tipo cellulare*: OT batte il mix 10/33 sull'atlante e 1/5 sugli eritroidi (`j_within.py`).

**Previsione.** *(ESPLORATIVO, `j_forecast.py`)*
- Anche scegliendo i parametri col senno di poi, uot_rw migliora persist di −1.0 / +1.0 / −2.5 / +0.9 / **+34.6** / +1.2 % (da E7.25 a E8.5). Conta solo E8.25.
- Il **backtest** (= provo i parametri sul passo prima, dove so già la risposta) non indovina il passo dopo:
  correlazione mediana 0.21 sull'atlante, 0.90 sugli eritroidi. Per E7.75 un guadagno di 0.016 nel backtest è costato 0.575 nella previsione vera.
- Quindi falliscono sia il metodo (poco guadagno anche nel caso migliore) sia la scelta dei parametri.
  Sugli eritroidi la crescita vista ieri continuava domani; sull'atlante no.

## 5. Cosa significa per il risultato eritroide

- **Era vero, non fortuna**: lì l'OT batte il mix nel 95 / 100 / 100 % delle estrazioni (stadi eritroidi k2, k4, k5; `j_noise.py`).
- **Ma era una base debole per un'idea generale**:
  - trovato a posteriori, con 4 unità: tirando una moneta, "almeno 3 su 4" esce il 31 % delle volte;
  - scelto tra almeno 6 varianti di OT e 4 misure, dopo aver cambiato la misura principale;
  - nello spazio dei geni si capovolge di circa 2 volte (mix 0.72/0.53/0.48/0.55 contro OT 1.39/1.25/1.11/1.29);
  - vale solo in 9 pool su 15 e in 1 unità su 5 dentro i tipi cellulari;
  - coppie **a caso** a metà strada danno già il 33 / 60 / 70 % del guadagno (`j_ery_indep.py`);
  - su cellule nuove si capovolge: 0/7.
- **Etichetta onesta**: effetto vero ma legato alla geometria eritroide (una sola linea che matura in fretta: ogni 6 ore le cellule si spostano di 1.2-1.9 raggi).
  Falsa pista per l'idea "l'OT batte il mix sugli embrioni nel tempo". Non un falso positivo da rumore.
- **Previsione eritroide**: +9.4 / 19.6 / 21.6 % su persist (origini o3-o5), ma trovata cercando tra 24 famiglie *dopo* che il test pre-registrato era fallito (1/4).
  Anche tilt_rw, senza OT, vinceva a o4-o5. Cavalca l'espansione tardiva dei globuli rossi. Sull'atlante: 1/6.

## 6. Cosa fare dopo

1. **Chiudere** questa linea come fonte di risultati positivi. Non rianalizzare l'atlante per "salvarla".
2. **Scriverla come risultato negativo e di metodo** (sezione controlli del preprint sul benchmark): il mix 50/50 batte l'OT 7/7;
   "OT batte coppie a caso" (H1c) è un'asticella troppo bassa; ogni test di interpolazione deve riportare union, indep e il numero di separazione.
   Prima di dire che il mix come rivale è un'idea nuova, controllare la letteratura.
3. **Solo se lo decidi tu**: nuova pre-registrazione su un terzo dataset, con l'ipotesi ristretta "l'OT batte il mix solo se la separazione è > ~1".
   Servono ≥ 4 pool per stadio, lotti di sequenziamento incrociati col tempo, stadi sopra e sotto la soglia. Probabilità a priori bassa. Richiede download e HPC.
4. **Previsione**: abbandonare uot_rw. Tenere al massimo il controllo "il backtest predice il passo dopo?" (0.21 contro 0.90).
5. **Registrare** l'esito di questi controlli in decisions.md / STATE come ESPLORATIVO. Il verdetto resta quello della sezione 3.

Limite di disegno: l'atlante ha E6.75 con un solo pool, E7.25 ed E8.25 con 3, e lotto di sequenziamento in parte legato allo stadio.
Non salva H1 (la sconfitta resiste togliendo un pool alla volta), ma limita la lettura per singolo pool.

## 7. Domande di autoverifica

1. Perché l'OT batte "solo prima" e "solo dopo" 6 volte su 7, ma perde contro il mix 7 su 7?
   → Sull'atlante lo stadio di mezzo è un miscuglio dei vicini, che si sovrappongono (separazione 0.37-0.55): il mix lo copia, i punti a metà strada stringono la nuvola.
2. Se contassimo i pareggi di uot_rw come vittorie, H2 passerebbe?
   → No: 4/6, ma la media resta peggiore (0.8154 contro 0.7708).
3. Quale singolo numero separa eritroidi e atlante?
   → Distanza tra i centri / raggio: 1.18-1.92 contro 0.37-0.55.
