# Pregi, difetti e un approccio nuovo
## Come modellare nel tempo una popolazione di cellule (generativo, quasi continuo, interpretabile)

1 ottobre 2026. Analisi fatta con 17 agenti indipendenti:
- 7 lettori: i tuoi progetti, le tue idee, i due articoli, lo stato dell'arte, i limiti teorici;
- 3 progettisti, ciascuno con un'angolazione diversa;
- 6 critici avversari;
- 1 giudice.

I dettagli tecnici completi sono in `docs/superpowers/analysis/` (vedi l'appendice in fondo).

---

## In breve

1. **Il problema vero non è l'architettura: è che dalle sole "foto" di cellule molte storie diverse danno le stesse foto.** Se vuoi manopole che significano qualcosa, devi (a) costruire un modello in cui quelle manopole si possono ricavare dai dati, e (b) controllarle con un dato che il modello non ha mai visto.
2. **I tuoi progetti finora: ottimo metodo di verifica, oggetto scientifico sbagliato.**
   - cfm_grn_v2, rispetto ai tuoi tre requisiti:

     | Requisito | Voto |
     |---|---|
     | Generativo | 2/5 |
     | Continuo | 2/5 |
     | Interpretabile | **0/5** |

   - Sui dati nuovi mai visti: 1 sconfitta e 1 pareggio contro il "rivale banale" (la media fra i due giorni vicini). Non per sfortuna: per una ragione algebrica.
   - La famiglia tgl-grn usava riassunti per gruppo invece delle cellule, quindi non poteva arrivare all'obiettivo nemmeno in teoria.
3. **Chronocell aggiunge molto:** è il pezzo da cui partire, riscritto da te. **L'articolo di Zecchina e colleghi aggiunge poco adesso:** è utile al secondo anno come strumento di lettura, e come contatto in Bocconi.
4. **Il mio approccio:** trattare la popolazione come una città con l'anagrafe.
   - Ogni cellula ha un'"età", cioè le ore da quando è uscita dal serbatoio di cellule staminali.
   - Tutte invecchiano alla stessa velocità del tempo reale.
   - Ogni giorno ne "nascono" un certo numero.
   - I geni si accendono e si spengono con velocità misurate in ore.

   Da qui si genera la popolazione di qualunque istante, e ogni manopola ha un significato.
5. **Prova decisiva:** il modello impara solo dall'RNA totale e poi deve indovinare l'RNA "nuovo", marcato chimicamente nelle ultime 2 ore e nascosto durante l'addestramento. Se orologio e velocità sono veri ci azzecca; se sono inventati sbaglia. La "media fra due foto" questa prova non può nemmeno tentarla.
6. **Probabilità oneste:**

   | Esito | Probabilità |
   |---|---|
   | Un risultato positivo presentabile entro 9 mesi (teoria + simulazioni + almeno una conferma su dati veri) | circa 60% |
   | Tutti e tre i requisiti pienamente soddisfatti sui dati veri | circa 20–25% |

---

## 1. Il problema, messo a nudo

### Cosa hai
- **Le "foto":** fotografie di una popolazione di cellule prese in pochi giorni (di solito 5–10).
- **Una sola foto per cellula:** ogni foto distrugge le cellule che fotografa, quindi **non vedi mai la stessa cellula due volte.**
- **Cellule fuori sincrono:** dentro la stessa foto alcune sono "in anticipo" e altre "in ritardo".
- **La popolazione cambia:** nel frattempo le cellule si dividono, muoiono e prendono strade diverse.
- **Giorno e lotto si confondono:** spesso ogni giorno è stato processato in un lotto diverso di laboratorio, quindi un cambiamento può venire dal tempo o dal lotto.

### Cosa vuoi
Una macchina che:
- **(G) generi** popolazioni realistiche, fatte di conteggi veri e non di numeri "normalizzati";
- **(C)** lo faccia **in qualunque istante**, con un **orologio vero** (ore e giorni), non un orologio inventato dal modello;
- **(I)** abbia **manopole con un significato biologico**: velocità di produzione e di distruzione dell'RNA, quante cellule iniziano il percorso e quando, quali strade prendono, a che età cambiano stato, quali programmi genici sono accesi.

### I due nodi da sciogliere

**Nodo 1: le foto non bastano per capire "chi è andato dove".**
- Pensa a due foto aeree di una piazza, una alle 10 e una alle 11: non ti dicono chi è andato dove.
- Un vortice di gente che gira in tondo è invisibile: la foto è identica.
- Se la piazza è più piena, non sai se sono arrivati da fuori o se sono "nati" lì.

La teoria lo dimostra (Weinreb 2018; Lavenant 2024; Guan 2024–2026). Una manopola si ricava dalle sole foto **solo** se il modello è costruito in un certo modo, oppure se hai dati in più: codici a barre di parentela, RNA marcato, perturbazioni. Leggere a posteriori una rete neurale "a scatola nera" **non** dà manopole vere.

**Nodo 2: il metro di giudizio usato finora premia la media.**
- Il punteggio usato in cfm_grn_v2 dipende per il 90–99% dal "baricentro" della nuvola di cellule, e solo per il resto dalla sua forma.
- Fare la media fra le due foto vicine indovina il baricentro quasi alla perfezione.
- In più, i tuoi esperimenti hanno dimostrato con l'algebra che qualunque percorso costruito "unendo i punti" fra le foto ha il baricentro inchiodato a una media delle foto.
- Risultato: nessun modello sofisticato poteva battere chiaramente il rivale banale con quel metro. **Serve un metro diverso**, che premi la forma, le proporzioni dei tipi cellulari, i conteggi veri e soprattutto la previsione di qualcosa che il modello non ha visto.

---

## 2. Cosa insegnano i tuoi progetti

### Pregi da tenere, perché funzionavano

| Cosa | Da dove viene |
|---|---|
| Separare sempre i due orologi (quello del modello e quello biologico) | cfm_grn_v2 |
| Dividere i dati per giorni interi, mai mescolando cellule dello stesso giorno fra addestramento e prova | cfm_grn_v2 |
| Confrontarsi sempre con rivali banali: media fra giorni, "niente cambia", interpolazione semplice, tempo mescolato a caso | tutti |
| Misurare il "tetto" prima di addestrare (quanto bene può andare al massimo questa strada?). È stato lo strumento più utile in assoluto. | cfm_grn_v2 |
| Regole decise **prima** di vedere i risultati, applicate dal codice; conferma su dati nuovi | cfm_grn_v2 (Job D/E) |
| Negativi "furbi" e controlli che possono uccidere un'ipotesi | audit Dileo (0,94 → 0,50) |
| Cambiare i **dati** prima dell'architettura. Segno delle interazioni e velocità misurate hanno spostato i risultati; 12 architetture diverse no. | cfm_grn v3/v4, tgl-grn |
| Codice riutilizzabile: metriche, controlli, lettori dei dataset, schemi di divisione, test anti-"sbirciata" | cfm_grn_v2, tgl-grn-continuous |

### Difetti da non ripetere

| Cosa | Da dove viene |
|---|---|
| Lavorare su riassunti per gruppo invece che sulle cellule | tutta la famiglia tgl-grn, compreso tgl-grn-continuous (30/09) |
| Fare di una rete "a scatola nera" deterministica l'oggetto scientifico: non si interpreta e non sa biforcarsi (una cellula non può scegliere fra due destini) | cfm_grn_v2 |
| Percorsi che "uniscono i punti" fra le foto: baricentro inchiodato alla media | cfm_grn_v2 |
| Lavorare su numeri normalizzati invece che sui conteggi: geni rarissimi diventano picchi assurdi (fino a 263.822 volte la deviazione) | cfm_grn_v2 |
| Rimandare sempre la parte scientifica: la parte "regolatoria" di RA-FM non è mai stata costruita | RA-FM fase 3 |
| Scegliere le correzioni dopo aver guardato i dati di prova; 12 versioni in una settimana | cfm_grn_v2 v0.1.x |
| Annunciare un positivo prima dei controlli | tgl-grn AUTOPSY §10.2 |
| Moduli "temporali" che in realtà non usavano il tempo | tgl-grn (stato sempre a zero) |
| Cambiare domanda di continuo senza fissare uno scopo (PURPOSE.md è ancora vuoto) | tutti |
| Lasciare che la RAM del portatile decida la scienza, quando c'è l'HPC | tgl-grn-continuous |
| Spendere settimane di calcolo su domande laterali (meccanismo della seconda fase di addestramento) | Job A/D |

---

## 3. Pregi e difetti delle tue idee

| Idea | Pregio | Difetto | Decisione |
|---|---|---|---|
| Generatore a flusso continuo (cuore di RA-FM) | genera popolazioni intere, scala bene | il baricentro è inchiodato, l'interpretabilità è zero, è deterministico | **lascia** come cuore; congela v0.1.10 come rivale di confronto |
| Crescita e morte delle cellule | necessaria per popolazioni vere | dalle sole foto si ricava solo in parte | **modifica**: solo differenze relative, verificate coi codici a barre |
| Moto con un po' di caso, non deterministico | serve per la forma corretta della nuvola | mai implementato | **modifica**: il caso viene da età miste + rumore dei conteggi |
| Interazioni fra geni (con segno) prese da un database noto (CollecTRI) | parametri con significato | nella versione attuale la "distruzione" agiva su numeri normalizzati, quindi non era biologica | **modifica**: dentro la velocità di produzione, sui conteggi veri |
| "Interruttori" per ogni singola interazione che cambiano nel tempo | è la domanda che ti appassiona | con 4–5 giorni il momento esatto di ogni interruttore è deciso dalle impostazioni del modello, non dai dati | **lascia** a livello di singola interazione; **sostituisci** con pochi "programmi" genici che si accendono con l'età, ognuno con geni-bersaglio esclusivi |
| Ancoraggio alla cromatina (multiome) | dato indipendente | il database di partenza è costruito dalla cromatina stessa, quindi la verifica sarebbe circolare | **anno 2**, solo a livello di regolatore |
| Tempi di propagazione dei knockout (RENGE, RAF1, CD4) | vera informazione causale | solo RENGE è davvero "tempo dopo knockout" ed è piccolo; RENGE (il metodo) è già un concorrente | **per ora no** |
| Batteria di controlli | più severa della letteratura | — | **tieni ed estendi** |
| Teoria su cosa si può ricavare | è il contributo più difendibile | — | **tieni**, rifocalizzata (vedi §5) |
| Benchmark temporale (TempGRN-Bench / estensione di GGE) | aggancio naturale a Liò | — | **modifica**: diventa il banco di prova sui conteggi |
| Programma "motore" (Job D/E, regole sull'orizzonte) | rigoroso | ortogonale alla tesi, costoso, probabilità basse | **ferma**; scrivi la scoperta algebrica sul baricentro come breve nota metodologica |
| Audit di Dileo | economico, nessuno l'ha fatto, mostra rigore | impatto limitato | **tieni** come nota breve (2–3 settimane al massimo) |
| Famiglia tgl-grn (5 copie) | test anti-sbirciata riutilizzabili | non può raggiungere l'obiettivo | **archivia** una copia canonica e chiudi |

---

## 4. I due articoli aggiungono qualcosa?

### Chronocell (Fang, Gorin, Pachter, PLoS Comput Biol 2025): **sì, molto (3,5/5)**

**Cosa fa.**
- Dà a ogni cellula un "tempo di processo", cioè quanto è avanzata nel suo percorso.
- Spiega i conteggi di RNA con produzione e distruzione a velocità costanti dentro ogni tratto.
- Dimostra che queste manopole si possono ricavare dai dati.
- Ha verificato le velocità di distruzione contro misure indipendenti di laboratorio.

**Cosa ti dà.**
- Il "decodificatore": dall'età e dalla strada di una cellula ai suoi conteggi, in forma esatta.
- Il lavoro sui conteggi veri invece che sui numeri normalizzati, il che elimina i picchi dei geni rari.
- I test di stabilità.
- Un punto preciso dove inserire l'orologio vero.

**Trappole.**
- Assume che le cellule siano distribuite in modo uniforme lungo il percorso. Il loro stesso articolo mostra che se non è così fallisce completamente.
- Non usa i giorni veri di prelievo.
- Non ha regolazione fra geni.
- Funziona su pochi geni (24–100).

Il tuo contributo sarebbe proprio sostituire l'ipotesi "uniforme" con l'**anagrafe** (§5).

### Badalotti, Baldassi, Mézard, Scardecchia, Zecchina (Phys Rev Research 2026): **poco, per ora (1,5/5 come metodo)**

**Cosa fa.** Reti neurali ricorrenti asimmetriche con neuroni binari. Un'"auto-attivazione" di ogni unità crea tantissimi stati stabili raggiungibili, e una regola di apprendimento locale li stabilizza. È fisica statistica applicata alla classificazione di immagini.

**Perché non serve adesso.**
- Nessun tempo reale.
- Nessuna popolazione.
- Nessuna generazione di dati.
- Unità binarie.
- La regola di apprendimento impara correlazioni, che è proprio la trappola dell'audit Dileo.

**Quando può servire (anno 2).** Quando avrai programmi genici con retroazione, i loro strumenti diventano utili per *leggere* il modello:
- un "indice di impegno" (quanto un regolatore si auto-sostiene);
- il censimento degli stati stabili;
- la robustezza agli scossoni;
- le biforcazioni nel tempo reale;
- la previsione di "isteresi": dopo un knockout temporaneo, la cellula resta cambiata?

**Valore pratico.** Sono in Bocconi, dove hai l'HPC: una conversazione con loro è realistica. Ma conta come rete di contatti, non come prova scientifica.

**Trappole.** "Attrattore = tipo cellulare" preso alla lettera; usare la loro regola di apprendimento; usare le loro soglie numeriche, che valgono solo per reti casuali.

---

## 5. Il mio approccio: "l'anagrafe delle cellule" (nome tecnico provvisorio: TACK)

### L'idea in parole povere
Pensa a una città di cui hai foto aeree in anni diversi. In ogni foto ci sono persone di tutte le età mescolate. Se sai:
- quante persone **entrano** in città ogni anno,
- che **tutti invecchiano di un anno all'anno**,
- e come cambia una persona con l'età (bambino → ragazzo → adulto, magari su strade diverse),

allora puoi ricostruire la foto di **qualunque** anno intermedio, e ogni numero del modello ha un significato.

Per le cellule:
- **"entrare in città"** = uscire dal serbatoio di staminali e iniziare il percorso di differenziamento;
- **"età"** = ore trascorse da allora;
- **"come cambia con l'età"** = quali geni si accendono e si spengono, con velocità di produzione e di distruzione dell'RNA misurate **in ore**;
- **"strade diverse"** = i destini possibili (per esempio neurale o mesodermico), su un albero dichiarato in anticipo.

### Perché risolve i problemi di prima

| Problema di prima | Come lo risolve |
|---|---|
| Il baricentro è inchiodato alla media fra due foto | La previsione in un istante nuovo nasce dal mescolare età diverse e dalla cinetica, non dall'unire i punti fra le foto. Non è una media. |
| Le cellule fuori sincrono nella stessa foto | Sono la normalità: hanno età diverse. Diventano un'informazione, non un disturbo. |
| Numeri normalizzati e picchi dei geni rari | Si lavora sui conteggi veri, con un rumore realistico. |
| Interpretabilità zero | Ogni manopola ha un'unità di misura e un significato. |
| "Non sa biforcarsi" | La strada è una scelta casuale dichiarata, con proporzioni che si possono stimare. |
| Il metro premia la media | Il metro principale diventa "indovinare ciò che non hai visto" (sotto). |

### Come genera una popolazione all'istante T (anche fra due giorni misurati)
1. Calcola, dall'anagrafe, quante cellule hanno ciascuna età e ciascuna strada all'istante T.
2. Pesca una cellula: strada, poi età.
3. Calcola i conteggi attesi dei suoi geni dalla cinetica, poi pesca conteggi realistici.

Oltre l'ultimo giorno misurato il modello **dice esplicitamente** che sta facendo un'ipotesi, e mostra tre scenari invece di una previsione falsamente sicura.

### Le manopole (cosa è interpretabile, e a quali condizioni)
- **Velocità di distruzione dell'RNA di ogni gene, in ore.** Solo per i geni in cui i dati la fissano davvero; per gli altri solo i rapporti.
- **Flusso di ingresso nel tempo:** quante cellule iniziano il percorso, e quando.
- **Proporzioni delle strade** e **crescita relativa** fra le strade.
- **Età a cui si cambia stato**, riportate come intervalli.
- **Secondo stadio (dal mese 6–7, solo se il primo funziona):** pochi **programmi genici con nome**, guidati da regolatori del database. Ognuno deve avere almeno 3 geni-bersaglio suoi esclusivi, contati prima di addestrare; altrimenti non lo si può nominare onestamente. Qui rientra la tua domanda originale: "quale regolatore si accende quando".

**Cosa il modello dichiara di NON sapere:**
- il percorso della singola cellula;
- i "vortici" (per esempio il ciclo cellulare, trattato come disturbo noto);
- il futuro oltre i dati;
- nascita e morte separate;
- le interazioni fuori dal database.

### Il contributo teorico (il pezzo più sicuro)
Gli agenti hanno trovato una condizione precisa per sapere **quando** le foto prese in giorni veri bastano a fissare le velocità **in ore**:
- serve che l'andamento degli ingressi nel tempo non abbia una certa forma "troppo regolare" (tecnicamente: non sia esponenziale né gaussiano), oppure che ci sia un inizio netto noto, come il giorno in cui parte il protocollo;
- serve che le cellule cambino abbastanza in fretta da essere "tracciabili".

Se queste condizioni mancano, il modello deve **rifiutarsi** di dare numeri in ore. Questo va dimostrato e verificato con simulatori che **devono** fallire nei casi sbagliati. Probabilità di riuscita di questa parte: circa 80%.

### Su quali dati (in ordine; tutti da scaricare sull'HPC, con il tuo permesso)

| Ordine | Dati | A cosa servono |
|---|---|---|
| 1 | **Simulatori** propri, costruiti in modo **diverso** dal modello, più dyngen | verificare che le manopole si ritrovino e che il modello si rifiuti quando deve |
| 2 | **Eritrociti di gastrulazione** (pacchetto scVelo, ~10.000 cellule, 7 stadi) | rifare Chronocell e verificare di riprodurre i loro numeri prima di confrontarsi |
| 3 | **sci-fate** (GSE131351; cellule A549 + desametasone, 0–10 ore ogni 2 ore, RNA marcato) | test "orologio noto": il modello deve trovare da solo l'inizio a 0 ± 1 ora |
| 4 | **sci-FATE2** (GSE236512; Maizels, Snell, Briscoe; 44.713 cellule staminali di topo → neurale/mesoderma; giorni 3–8; 3 repliche complete processate insieme, quindi niente confusione giorno/lotto; una quarta replica ogni 5 ore fra giorno 3 e 4; RNA marcato per 2 ore prima del prelievo) | **dataset principale e prova decisiva** |
| 5 | **Riprogrammazione con CellTag** (Biddy/Morris; 8 giorni, codici a barre di parentela) | controllare proporzioni delle strade e crescita contro i cloni veri |

Da verificare appena scaricato: l'elenco esatto dei giorni di sci-FATE2 e quali tabelle ci sono nei file (la pagina GEO era bloccata da un captcha).

### Come si giudica: i voti principali, nessuno dominato dal baricentro
1. **La prova decisiva.** Il modello vede solo l'RNA totale e deve indovinare, cellula per cellula, quanto RNA è "nuovo" (marcato nelle ultime 2 ore). Lo confronti con:
   - il modello con ipotesi "uniforme" (Chronocell);
   - il modello con "orologio = giorno di prelievo ± rumore" (VeloVAE);
   - un modello con velocità tutte uguali;
   - il tempo mescolato a caso.
2. **Le velocità di distruzione stimate contro quelle misurate dal laboratorio:** stessa classifica? Stessa scala in ore?
3. **Generazione a istanti intermedi veri:** alleni sui giorni 3 e 5–8 e prevedi il giorno 4 e i prelievi ogni 5 ore della quarta replica. Lo misuri come "quanto ti avvicini a un oracolo" che conosce già la risposta, usando per tutti gli stessi pezzi tranne l'orologio. Così vince solo chi ha l'orologio giusto, non chi ha il decodificatore migliore.
4. **Calibrazione:** la quantità di zeri, la variabilità per gene e le proporzioni dei tipi cellulari devono stare dentro la differenza fra repliche.

### Criteri di stop (decisi prima, applicati dal codice)

| Quando | Controllo | Se fallisce |
|---|---|---|
| Settimana 3 | Sui dati di allenamento c'è davvero margine fra il miglior rivale e l'oracolo? | Si toglie la promessa "generativa" invece di inseguirla per mesi |
| Mese 3 | Sui simulatori le manopole si ritrovano? | Si scrive solo l'articolo teorico, che è comunque pubblicabile |
| Mese 4 | Sul test "orologio noto" | Il problema è un errore di codice, non scientifico: solo correzioni |
| Mese 6 | Se le ore non si fissano su sci-FATE2 | Si riportano solo i rapporti, più la teoria sul perché |
| Sempre | Se un dataset è instabile | Nessuna affermazione su di lui |

### Piano in 9 mesi

| Mese | Cosa |
|---|---|
| 0 (prime 3 settimane) | Compilare PURPOSE.md (lo scopo, una volta per tutte). Scrivere e verificare la condizione teorica. Schede dei dati. Lista dei download da approvare. Banco di prova con i rivali. Primo controllo di margine. |
| 1–2 | Il cuore del modello (decodificatore, anagrafe, stima). Test che le formule esatte coincidano con la simulazione. Riproduzione di Chronocell sugli eritrociti. |
| 2–3 | Simulatori, compresi quelli "che devono fallire". Curve di rottura quando le ipotesi sono violate. |
| 4 | Test "orologio noto" (sci-fate). Confronto fra ipotesi sugli eritrociti. Congelamento delle regole per sci-FATE2. |
| 5–6 | sci-FATE2: orologio e generazione. Confronto diretto con VeloVAE (prima riprodotto nel tuo banco). |
| 7 | Programmi genici con nome (lista di regolatori decisa prima, per esempio Sox2, Tbxt, Cdx2, Olig2, Nkx2-2, Foxa2). |
| 8 | Riprogrammazione con codici a barre. Nota sull'audit Dileo nei tempi morti della coda HPC. |
| 9 | Preprint: teoria + simulatori + validazione su dati veri + banco di prova. Proposta per Liò con gli agganci per l'anno 2: programmi con retroazione, strumenti di Zecchina, cromatina, crescita con codici a barre. |

**Calcolo:** circa 250–350 ore di A100 in totale. Una stima del modello costa 0,5–2 ore. Con il vincolo di 1 job alla volta si mettono più stime in fila dentro lo stesso job. Il codice è originale, circa 5–7 mila righe, più i pezzi riusati.

### Probabilità oneste

| Esito | Probabilità |
|---|---|
| Teoria + simulatori (compresi i fallimenti previsti) | 0,8 |
| Test "orologio noto" superato | 0,75 |
| Velocità in ore confermate su sci-FATE2 | 0,4–0,5 |
| Generazione intermedia migliore dei rivali | 0,4 |
| Programmi con nome che battono quelli casuali | 0,3 |
| **Positivo presentabile entro il mese 9** | **0,6–0,65** |
| Tutto pieno (G+C+I) | 0,2–0,25 |

### Concorrenti, e cosa resterebbe nuovo

| Concorrente | Cosa fa già | Cosa gli manca |
|---|---|---|
| **VeloVAE** (Gu, …, Welch, PLoS Comput Biol 2026, verificato) | usa i giorni di prelievo come indizio, ha già usato sci-FATE2 | niente anagrafe, crescita o ingressi; nessuna teoria su quando le ore si possono ricavare |
| **Velvet / VelvetSDE** (Maizels, Snell, Briscoe; gli autori di sci-FATE2) | dinamica neurale con caso sullo stesso dataset | scatola nera, nessuna velocità in ore con significato |
| **Chronocell** | il punto di partenza | ipotesi uniforme, nessun giorno vero |
| **CardamomOT** (Maugé, Ventre 2026) | meccanistico + trasporto + generativo su serie temporali | rischio di sovrapposizione, da leggere bene subito |
| **PFM, StructureFlow, Cell-MNN, DeepRUOT** | generatori potenti | interpretabilità solo a posteriori, o parametri senza unità di misura |

**Cosa resta nuovo:**
1. La popolazione ad anagrafe accoppiata alla cinetica sui conteggi.
2. La condizione teorica che dice quando le ore si possono ricavare, con un modello che si rifiuta quando non si può.
3. La prova sull'RNA nuovo nascosto, contro i due tipi di orologio dei concorrenti.
4. La generazione a istanti intermedi veri.

### Cosa ho cambiato io rispetto alla sintesi degli agenti
- **Più leggero all'inizio.** La sintesi propone sei "certificati" e molte regole. È lo stesso rischio di cfm_grn_v2, dove il processo era pesante rispetto agli effetti. Io partirei con **una sola prova decisiva** (l'RNA nuovo nascosto) e **un solo controllo di margine** (settimana 3). Gli altri certificati li aggiungerei solo quando servono a un'affermazione precisa.
- **La tua domanda sulla regolazione non sparisce, viene rimandata con un motivo.** Prima l'orologio deve funzionare, poi i programmi. Se l'orologio non regge, i programmi non avrebbero significato comunque.
- **Velvet va aggiunto subito alla lista dei concorrenti:** la sintesi non lo nominava, eppure è dello stesso gruppo che ha prodotto il dataset principale.

---

## 6. Prossimi passi concreti (servono decisioni tue)

1. **Decidi** se adottare questa direzione. Se sì, la scrivo in PURPOSE.md e nel registro decisioni.
2. **Fermare** i job e le copie che non servono più: programma "motore" e famiglia tgl-grn. tgl-grn-continuous è di Codex ed è ancora "in corso": chiudiamolo con un paragrafo di verdetto.
3. **Permessi di download sull'HPC**, in ordine:
   - eritrociti scVelo (piccolo);
   - sci-fate GSE131351;
   - sci-FATE2 GSE236512 (dimensione da verificare);
   - CellTag di Biddy/Morris (circa 1,5 GB).
4. **Prima settimana:** leggere CardamomOT e Velvet in dettaglio, per escludere una sovrapposizione prima di scrivere codice.

---

## Appendice tecnica: dove sono i dettagli

Cartella `docs/superpowers/analysis/`:

| File | Contenuto |
|---|---|
| `2026-10-01-synthesis-TACK.md` | Sintesi del giudice: equazioni, condizione teorica, regole di decisione, controlli, fonti |
| `2026-10-01-dossiers.md` | I 7 dossier: cfm_grn_v2, famiglia tgl-grn, inventario delle idee, i due articoli, stato dell'arte, limiti teorici |
| `2026-10-01-design-A-kinetic-clock.md` | Progetto A con le due critiche |
| `2026-10-01-design-B-identifiable-sde.md` | Progetto B con le due critiche |
| `2026-10-01-design-C-evidence-first.md` | Progetto C con le due critiche |

**Riferimenti non verificati, da controllare prima di citarli:** Navigo, GRNvelo (non trovati), il numero esatto di giorni e le tabelle di sci-FATE2, la disponibilità dei conteggi spliced/unspliced su sci-fate.

**Fonti principali:**
- Chronocell: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11760028/
- Badalotti et al. 2026: https://journals.aps.org/prresearch/abstract/10.1103/lbr6-91sn (arXiv 2509.05041)
- VeloVAE: https://pmc.ncbi.nlm.nih.gov/articles/PMC13021174/
- sci-FATE2 / Velvet: https://www.biorxiv.org/content/10.1101/2023.07.06.547989.full.pdf
- GSE236512: https://www.omicsdi.org/dataset/geo/GSE236512
- CardamomOT: https://cardamomot.readthedocs.io/en/latest/
- Guan et al. (identificabilità): https://proceedings.mlr.press/v300/guan26a.html, https://arxiv.org/abs/2410.22729
