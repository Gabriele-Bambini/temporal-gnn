# RA-FM, Fase 1: motore CFM + benchmark sintetico con attività regolatoria variabile — piano di implementazione

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** costruire `cfm_grn_v2`, un motore originale di conditional flow matching per snapshot single-cell non appaiati, e verificarlo contro controlli obbligatori su un benchmark sintetico con GRN la cui attività per arco varia nel tempo biologico (verità nota), con cross-validation per time point e grid search annidata.

**Architecture:** pacchetto `src/` con quattro strati separati — dati (schema, simulatori, split, preprocessing addestrato solo sul training), flussi (path lineare, coupling OT entro gruppi compatibili, solver ODE in `flow_time`), modelli (campo vettoriale MLP + encoder del contesto che riceve `biological_time`), valutazione (metriche distribuzionali, controlli, report con verdetto). Nessuna GNN, nessuna maschera in questa fase: il verdetto sul generatore precede la struttura.

**Tech Stack:** Python ≥3.11, NumPy, SciPy (assegnamento di Hungarian per l'OT a minibatch), PyTorch CPU/GPU, scikit-learn (controlli), pandas, pytest. Nessuna dipendenza da torchcfm, CellFlow, scDFM, StructureFlow, FlowGRN, RegVelo.

**Spec:** `docs/superpowers/specs/2026-09-16-flow-matching-first-design.md` (fasi 1–4, due tempi, controlli, test obbligatori) + `docs/PHD_PROJECT_2026-09-17.md` §5–7 (modello, dati, valutazione) + audit `tgl-grn-1CCD/logs-baseline/NEGATIVE_SAMPLING_AUDIT_2026-09-17.md` §6.

## Global Constraints

- Codice originale: i repository degli autori sono baseline via adapter, mai copiati.
- Ordine: motore FM senza grafo → snapshot reali → condizionamento trasferibile → prior GRN / maschera. Nessuna GNN prima del verdetto G1.
- Due clock separati nel codice e nelle API: `flow_time` (s ∈ [0,1], costruzione matematica) e `biological_time` (ore/giorni/stadi). Nessuna metrica o figura chiama `flow_time` tempo biologico.
- Split per unità biologiche intere (time point, replicato, target di perturbazione), mai per cellule casuali. Preprocessing addestrato solo sul training e serializzato.
- Controlli obbligatori riportati sempre: persistenza, media del training, mean-shift, lineare, MLP deterministico, replay FreeFM, tempo rimescolato; poi (Fase 3) nessun grafo, grafo statico, prior randomizzato. ≥3 seed con IC95 e differenze appaiate.
- Nessuna soglia di successo arbitraria: la differenza rilevante si confronta con il rumore (noise floor) e con la variabilità fra seed, fissati prima del test.
- Il coupling OT non attraversa mai condizioni, replicati o coppie di tempi diverse; una permutazione del target permuta anche i suoi metadati.
- Un checkpoint contiene schema, geni, preprocessing, configurazione e versione; incompatibile → errore, mai caricamento silenzioso.
- Ogni ridisegno = nuova versione di configurazione + rerun dell'intera suite di controlli; il test finale si guarda una volta sola per configurazione.

---

## Parte A — Roadmap "bulletproof" (fasi, gate, ridisegno, cross-validation, grid search)

### A.1 Fasi e gate

| fase | settimane | cosa si costruisce | gate di uscita (misurato, non opinabile) | se il gate fallisce (ridisegno on the road) |
|---|---|---|---|---|
| 0 | 0 | Ricontrollo novità (agente in corso), spec congelata, questo piano | nessun paper con lo stesso claim (attività per arco variabile dentro un FM multi-marginale/interventional ancorato) | differenziare su ancore/interventi/teoria; se scoop totale → alternativa A (valutazione-prima) |
| 1 | 1–3 | `cfm_grn_v2`: motore CFM, toy 2D, simulatore GRN con attività variabile, split, controlli, grid search annidata, report | **G1**: toy (sanity) — FM indistinguibile dal target al test di permutazione (ED ≤ `null_q95`) su ≥3/4 trasformazioni e sotto il proprio controllo a tempo rimescolato; GRN sintetico (benchmark, dalla v0.1.5) — FM batte tutti i controlli sul time point interno escluso (interpolazione, t=3: differenza appaiata IC95 < 0 sull'energy distance contro persistenza, interpolazione delle medie, interpolante OT/McCann, mappe lineare/MLP, replay della velocità) e batte il controllo a tempo rimescolato; il forecast del time point terminale è riportato come diagnostica (estrapolazione pura fuori dal supporto: il replay a velocità costante è quasi ottimo per costruzione); test anti-leakage verdi | in ordine: (1) coupling OT vs indipendente, (2) σ_min, (3) capacità/profondità, (4) embedding di Δt, (5) più passi ODE; ogni cambio = config v+1; se dopo 5 cicli FM ≤ persistenza → il generatore cambia famiglia (bridge stocastico) prima di toccare la struttura |
| 2 | 3–6 | Dati reali senza struttura: Klein GSE65525 (smoke), Waddington-OT (un braccio) o Hagai E-MTAB-6754 (0/2/4 h → 6 h), gastrulation cell-level con stadi reali (Pijuan-Sala E-MTAB-6967) | **G2**: forecast del tempo terminale escluso ≥ controlli (IC95), controllo tempo rimescolato peggiora, FreeFM non batte FM; risultati stabili in PCA-50 e nello spazio dei geni | crescita/sbilanciamento (proliferazione), spazio latente vs geni, coupling sbilanciato, regolazione σ; se G2 fallisce su tutti i dataset → verdetto negativo documentato, non nascosto con la GNN |
| 3 | 6–12 | Struttura: prior GRN diretto e firmato, message passing sparso, gate statici poi variabili m_ij(t,x) (RA-FM v0) — prima sul sintetico con verità nota | **G3**: recupero del rewiring sintetico (AUPR su m(t)) sopra prior statico e prior randomizzato; forecast reale non peggiore di Fase 2; gate variabili > gate statici (IC95) | se recupero fallisce ma forecast ok → non identificabile da soli marginali: aggiungere ancore (osservazioni simulate di m, poi ATAC reale); se gate variabili ≈ statici → ridurre il gating, spostare peso su benchmark/teoria |
| 4 | mesi 3–6 | Interventi: RENGE GSE213069 / RAF1:ER GSE250559 (tempo dopo knockout), CV per target di perturbazione, operatore do sul drift; ancore multiome (Argelaguet 2022 gastrulation) | **G4**: target esclusi predetti meglio delle baseline lineari (Ahlmann-Eltze) e di FM senza interventi; m(t) correlata con accessibilità su dati non usati | se G4 fallisce → il claim si restringe a "forecast + rewiring sintetico" e il preprint diventa benchmark + metodo parziale |

Preprint 1 (audit + protocollo) indipendente da queste fasi, pronto quando il lab Sottoriva è informato. Preprint 2 (metodo) dopo G3 con almeno un dataset reale.

### A.2 Protocollo di cross-validation dei dati (fissato prima di ogni test)

1. **Unità di split**: time point interi (forecast), replicati biologici (toy e dataset con più repliche), target di perturbazione con tutte le guide (Fase 4). Mai cellule casuali della stessa unità fra training e test.
2. **Forecast (diagnostico dalla v0.1.5, vedi punto 3)**: K time point ordinati t₁…t_K. Test = t_K (mai usato in selezione). Training = t₁…t_{K-1}. Validazione interna = t_{K-1} predetto da t_{K-2} con modello addestrato su t₁…t_{K-2}. Il test finale parte dalle cellule osservate a t_{K-1} e predice t_K con contesto (t_{K-1}, t_K), quindi estrapolazione in tempo biologico.
3. **Interpolazione (primario per il GRN dalla v0.1.5, K ≥ 4)**: leave-one-inner-timepoint-out; il time point escluso t_k è predetto dalle cellule a t_{k-1} con orizzonte t_k − t_{k-1}; la transizione che scavalca il buco (t_{k-1} → t_{k+1}) è osservata; validazione annidata su un secondo time point interno (il più lontano da t_k), refit su tutti i time point osservati; controlli: persistenza, interpolazione delle medie, interpolante OT (McCann) lungo le coppie della transizione che scavalca, mappe lineare/MLP sulle coppie osservate con lo stesso orizzonte, replay della velocità in arrivo; riportato separato dal forecast, mai mescolato.
4. **Replicati**: fit su r₁, validazione su r₂, test su r₃ quando esistono; con 2 replicati, entrambi nel training e split solo per tempo.
5. **Perturbazioni (Fase 4)**: fold raggruppati per gene target; un TF escluso esce con tutte le sue guide da training e tuning.
6. **Preprocessing**: selezione geni, standardizzazione, PCA e coupling OT addestrati solo sulle cellule di fit; serializzati nel checkpoint; il test non contribuisce a nulla (test automatico: cambiare le cellule di test non cambia i pesi).
7. **Modello finale (v0.1.7: nessun refit)**: il modello finale è quello fermato dall'early stopping sul fit soltanto; la validazione serve solo a selezione e arresto. Il refit su fit+validazione con il numero di iterazioni trasferito è fragile quando la curva di validazione non è monotona (run completo forecast: 2 seed su 5 refittati a 500 iterazioni → ED 13). I controlli usano tutti i dati osservati (fit+validazione) e lo stesso spazio di valutazione: svantaggio conservativo per il FM. `refit_on_all=True` resta disponibile come opzione.
8. **Seed**: ≥3 seed per modello; metriche aggregate prima per seed, poi media ± IC95 (t di Student); differenze appaiate per seed contro il miglior controllo.
9. **Noise floor e test di permutazione**: floor = ED fra due metà indipendenti del target (attesa 0 con la U-statistica, riportato per informazione); criterio di indistinguibilità = quantile 95% dell'ED fra due split casuali di predizione ∪ target, alle stesse numerosità della valutazione (`null_q95`): sotto quel valore la predizione non è distinguibile dal target.
10. **Metriche**: energy distance (primaria, U-statistica: attesa 0 a distribuzioni uguali per ogni numerosità), MMD-RBF con banda fissata sul training, W2 per assegnamento su ≤500 cellule, errore sulle medie, W1 medio per gene; calcolate nello spazio standardizzato dal training; in Fase 2 anche in PCA-50 congelata.

### A.3 Grid search (annidata, stesso budget per tutti)

- Griglia di Fase 1 per il FM (v0.1.3): toy `hidden ∈ {128, 256}`, `lr ∈ {1e-3, 3e-4}`, `sigma_min ∈ {1e-3, 1e-2}`, `coupling ∈ {ot, independent}` → 16 configurazioni; GRN stessa griglia + `pair_mode ∈ {consecutive, all}` + `clock ∈ {flow, duration}` con `time_conditioning=delta` fisso (64 configurazioni) + configurazioni `path=spline` (clock duration, OT concatenato); v0.1.7: `lr=1e-3` fisso, 12 configurazioni lineari campionate + tutte le 4 spline = 16, selezione dell'interpolazione sulla media di due fold di validazione (`interpolation_val_folds`); × 2 seed di ricerca; selezione = media dell'energy distance di validazione. Budget in `--max-configs` (campionamento casuale della griglia) quando il tempo non basta; il budget si dichiara nel report.
- Controllo MLP deterministico: `hidden ∈ {64, 128, 256}` selezionato con lo stesso criterio; lineare: `alpha ∈ {0.1, 1, 10}`. Persistenza, mean-shift, media, FreeFM non hanno iperparametri.
- Early stopping sulla validazione con pazienza (v0.1.3: 8000 iterazioni, valutazione ogni 250, pazienza 8); validazione e modello finale usano la media mobile esponenziale dei pesi (`ema_decay=0.999`) e il gradiente è troncato a norma 1; il numero di iterazioni scelto è quello del refit.
- Tutti i run della ricerca finiscono in `search.csv`; il test finale non entra mai nella selezione.
- Fase 2: stessa griglia + `n_ode_steps ∈ {20, 50}` e spazio (geni vs PCA-50); Fase 3: + rango dei gate, L1 sui gate, penalità di variazione totale in t.

### A.4 Ridisegno on the road: regole

1. Ogni modifica al metodo o ai dati è una nuova versione (`configs/vN.json`) con motivazione in `docs/superpowers/decisions.md` (data, osservazione, ipotesi, cambio, esito).
2. Si ridisegna guardando validazione e diagnostica (loss, traiettorie, ED per gruppo), mai il test.
3. Dopo ogni ridisegno gira l'intera suite: controlli, seed, noise floor. Un miglioramento senza controlli non conta.
4. Se un gate fallisce due volte con le opzioni della tabella A.1, si scrive il verdetto negativo e si passa al fallback della fase, non si aggiunge complessità.
5. La suite di test (`pytest`) deve restare verde a ogni versione; un test rosso blocca la versione.

### A.5 Audit dei dati reali (Fase 2+), scheda per dataset

Per ogni dataset prima di usarlo: accessione verificata e checksum del download; tabella tempi × condizioni × replicati con conteggi; controlli e batch; QC (cellule per unità ≥ 200 o eccezione documentata); geni selezionati sul training (HVG) e copertura del prior (Fase 3); nota su tempo reale vs pseudotempo. Scheda in `cfm_grn_v2/data_cards/<dataset>.md`. Dataset ammesso solo con ≥3 time point per il forecast (≥2 per lo smoke test) e ≥2 unità biologiche.

### A.6 Calcolo

Fase 1 gira in locale (CPU, 16 thread; run completo ≤ 2 h). Fase 2+ su Bocconi (`ssh bocconi`, partizione `stud`, 1 job alla volta, loop sequenziale nel job; vedi memoria `bocconi-hpc-infra`). Risultati sempre con configurazione completa e versione del codice.

---

## Parte B — Fase 1: struttura dei file

- `cfm_grn_v2/pyproject.toml`: metadati, dipendenze, pytest con `pythonpath = ["src"]` (fuori da pytest: `PYTHONPATH=src` oppure `pip install -e .`).
- `cfm_grn_v2/src/cfm_grn_v2/__init__.py`: versione.
- `cfm_grn_v2/src/cfm_grn_v2/flows/path.py`: path lineare, target di velocità, loss CFM, `flow_time`.
- `cfm_grn_v2/src/cfm_grn_v2/flows/coupling.py`: coupling indipendente e OT esatto a minibatch, coupling entro gruppi con coerenza dei metadati.
- `cfm_grn_v2/src/cfm_grn_v2/flows/solver.py`: integratore Euler / midpoint / RK4 in `flow_time`.
- `cfm_grn_v2/src/cfm_grn_v2/models/conditioning.py`: registro delle condizioni (errore su condizione ignota), encoder del contesto da `biological_time` e condizione.
- `cfm_grn_v2/src/cfm_grn_v2/models/vector_field.py`: embedding sinusoidale di `flow_time`, campo vettoriale MLP.
- `cfm_grn_v2/src/cfm_grn_v2/data/schema.py`: `SnapshotSet` con validazione.
- `cfm_grn_v2/src/cfm_grn_v2/data/synthetic_toys.py`: quattro trasformazioni 2D note.
- `cfm_grn_v2/src/cfm_grn_v2/data/splits.py`: split per tempo, replicato, gruppo; asserzioni di disgiunzione.
- `cfm_grn_v2/src/cfm_grn_v2/data/preprocessing.py`: standardizzazione addestrata solo sul training, serializzabile.
- `cfm_grn_v2/src/cfm_grn_v2/evaluation/metrics.py`: metriche distribuzionali, protocollo, noise floor.
- `cfm_grn_v2/src/cfm_grn_v2/data/synthetic_grn.py`: simulatore GRN con attività per arco variabile nel tempo, campionamento distruttivo, knockout.
- `cfm_grn_v2/src/cfm_grn_v2/evaluation/controls.py`: persistenza, media, mean-shift, lineare, MLP, replay FreeFM, coppie OT per i controlli.
- `cfm_grn_v2/src/cfm_grn_v2/training/config.py`: configurazione e griglie.
- `cfm_grn_v2/src/cfm_grn_v2/training/trainer.py`: transizioni, task di forecast, `FlowModel`, `train_flow` con early stopping e controllo a tempo rimescolato, checkpoint.
- `cfm_grn_v2/src/cfm_grn_v2/training/search.py`: grid search annidata e benchmark di forecast con controlli.
- `cfm_grn_v2/src/cfm_grn_v2/evaluation/report.py`: aggregazione per seed, IC95, differenze appaiate, verdetto, report markdown.
- `cfm_grn_v2/src/cfm_grn_v2/experiments/run_synthetic.py`: CLI della Fase 1 (toy + GRN, ricerca, controlli, report).
- `cfm_grn_v2/tests/`: un file di test per modulo + `test_leakage.py` + `test_integration_toys.py`.

Convenzione: ogni blocco di codice nei task inizia con `# file: <percorso>`; lo script `scripts/materialize_plan.py` scrive i blocchi sui file (un blocco per file), così il codice del piano e quello su disco coincidono.

---

## Parte C — Task della Fase 1

### Task 1: Scaffold del pacchetto

**Files:**
- Create: `cfm_grn_v2/pyproject.toml`, `cfm_grn_v2/src/cfm_grn_v2/__init__.py`, `cfm_grn_v2/tests/conftest.py`
- Test: `cfm_grn_v2/tests/test_import.py`

**Interfaces:**
- Produces: `cfm_grn_v2.__version__: str`.

- [ ] **Step 1: test di import (RED)**

```python
# file: cfm_grn_v2/tests/test_import.py
def test_package_exposes_version():
    import cfm_grn_v2
    assert isinstance(cfm_grn_v2.__version__, str)
```

- [ ] **Step 2: eseguire `python -m pytest tests/test_import.py -q` dentro `cfm_grn_v2/` → FAIL (modulo assente)**

- [ ] **Step 3: scaffold**

```toml
# file: cfm_grn_v2/pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "cfm_grn_v2"
version = "0.1.11"
description = "Original conditional flow matching engine for unpaired single-cell snapshots (flow_time is not biological_time)"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "scipy>=1.11", "torch>=2.1", "scikit-learn>=1.3", "pandas>=2.0"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
markers = ["slow: long-running integration tests"]
```

```python
# file: cfm_grn_v2/src/cfm_grn_v2/__init__.py
"""cfm_grn_v2: original conditional flow matching engine for single-cell snapshot dynamics.

Two clocks: ``flow_time`` (generative path coordinate in [0, 1]) and ``biological_time``
(measured hours/days/stages). They are never interchanged.
"""
__version__ = "0.1.11"
```

```python
# file: cfm_grn_v2/tests/conftest.py
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    yield
```

- [ ] **Step 4: `python -m pytest tests/test_import.py -q` → 1 passed**

### Task 2: Path lineare e loss CFM

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/flows/__init__.py`, `cfm_grn_v2/src/cfm_grn_v2/flows/path.py`
- Test: `cfm_grn_v2/tests/test_path.py`

**Interfaces:**
- Produces: `sample_flow_time(batch_size, generator=None, device=None) -> Tensor[B]`; `interpolate(x0, x1, flow_time, sigma_min=1e-3, noise=None) -> Tensor`; `velocity_target(x0, x1) -> Tensor`; `cfm_loss(vector_field, x0, x1, context, flow_time, sigma_min=1e-3, target_scale=1.0) -> Tensor` dove `vector_field(x_s, flow_time, context)`; `SplinePath(times, trajectories[m, n, D], bc_type="natural")` con `.evaluate(u, idx) -> (x, v)` e `.sample(rng, b) -> (x, v, u)`; `spline_loss(vector_field, x, v, context, sigma_min=1e-3) -> Tensor`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_path.py
import pytest
import torch

import numpy as np

from cfm_grn_v2.flows.path import SplinePath, cfm_loss, interpolate, sample_flow_time, spline_loss, velocity_target


def test_path_endpoints_without_noise():
    x0 = torch.zeros(4, 3)
    x1 = torch.ones(4, 3)
    assert torch.allclose(interpolate(x0, x1, torch.zeros(4), sigma_min=0.0), x0)
    assert torch.allclose(interpolate(x0, x1, torch.ones(4), sigma_min=0.0), x1)


def test_velocity_target_is_displacement():
    x0 = torch.randn(5, 2)
    x1 = torch.randn(5, 2)
    assert torch.allclose(velocity_target(x0, x1), x1 - x0)


def test_flow_time_in_unit_interval():
    s = sample_flow_time(1000, generator=torch.Generator().manual_seed(0))
    assert s.shape == (1000,) and float(s.min()) >= 0.0 and float(s.max()) < 1.0


def test_cfm_loss_zero_for_exact_field():
    x0 = torch.randn(8, 2)
    shift = torch.tensor([1.0, -2.0])
    x1 = x0 + shift
    field = lambda x, s, c: shift.expand_as(x)
    loss = cfm_loss(field, x0, x1, context=None, flow_time=torch.rand(8), sigma_min=0.0)
    assert float(loss) < 1e-12


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        interpolate(torch.zeros(2, 3), torch.zeros(3, 3), torch.zeros(2))


def test_spline_path_hits_knots_and_recovers_constant_velocity():
    times = np.array([0.0, 1.0, 2.0, 3.5])
    v = np.array([[1.0, -2.0], [0.5, 0.5], [-1.0, 3.0]])                  # three trajectories, constant velocity
    traj = np.stack([k * v + np.arange(3)[:, None] for k in times])       # (4, 3, 2)
    path = SplinePath(times, traj)
    x, vel = path.evaluate(times, [0, 1, 2, 0])
    assert np.allclose(x, traj[np.arange(4), [0, 1, 2, 0]])
    assert np.allclose(vel, v[[0, 1, 2, 0]], atol=1e-8)
    xs, vs, us = path.sample(np.random.default_rng(0), 64)
    assert xs.shape == (64, 2) and vs.shape == (64, 2) and us.min() >= 0.0 and us.max() <= 3.5
    two = SplinePath([0.0, 2.0], traj[[0, 2]])                             # two knots: linear segment
    assert np.allclose(two.evaluate([1.0], [1])[1], v[1])
    with pytest.raises(ValueError):
        SplinePath([0.0, 0.0], traj[:2])
    field = lambda x, s, c: torch.as_tensor(v[1], dtype=x.dtype).expand_as(x)
    xt, vt = torch.as_tensor(xs[:8], dtype=torch.float32), torch.as_tensor(np.tile(v[1], (8, 1)), dtype=torch.float32)
    assert float(spline_loss(field, xt, vt, None, sigma_min=0.0)) < 1e-12


def test_cfm_loss_target_scale_turns_displacement_into_velocity():
    x0 = torch.randn(8, 2)
    shift = torch.tensor([2.0, -4.0])
    x1 = x0 + shift
    field = lambda x, s, c: (shift / 2.0).expand_as(x)          # velocity per unit time for a horizon of 2
    assert float(cfm_loss(field, x0, x1, None, torch.rand(8), sigma_min=0.0, target_scale=0.5)) < 1e-12
    assert float(cfm_loss(field, x0, x1, None, torch.rand(8), sigma_min=0.0)) > 1.0


def test_spline_path_linear_centre():
    """center="linear": the path mean follows straight segments between the knot means, the deviation of each
    trajectory from the mean is the spline's, the path still passes through every observed cell at the knots,
    and the velocity is the derivative of the state."""
    import numpy as np
    from cfm_grn_v2.flows.path import SplinePath
    rng = np.random.default_rng(0)
    t = np.array([0.0, 1.0, 3.0, 4.0])
    traj = rng.normal(size=(4, 50, 3)) + np.array([0.0, 1.0, 4.0, 4.5])[:, None, None]
    sp_ = SplinePath(t, traj, center="spline")
    li = SplinePath(t, traj, center="linear")
    idx = np.arange(50)
    for k, tk in enumerate(t):
        x, _ = li.evaluate(np.full(50, tk), idx)
        assert np.allclose(x, traj[k], atol=1e-9)                                  # through the observed cells
    u = np.full(50, 2.0)
    xl, _ = li.evaluate(u, idx)
    xs, _ = sp_.evaluate(u, idx)
    m = traj.mean(1)
    assert np.allclose(xl.mean(0), m[1] + 0.5 * (m[2] - m[1]), atol=1e-9)        # straight centre
    assert np.allclose(xl - xl.mean(0), xs - xs.mean(0), atol=1e-9)              # spline shape
    eps = 1e-6
    for uu in (0.4, 2.2, 3.7):
        x0, v0 = li.evaluate(np.full(50, uu), idx)
        x1, _ = li.evaluate(np.full(50, uu + eps), idx)
        assert np.allclose((x1 - x0) / eps, v0, atol=1e-4)
    with np.testing.assert_raises(ValueError):
        SplinePath(t, traj, center="quadratic")
```

- [ ] **Step 2: `python -m pytest tests/test_path.py -q` → FAIL (import)**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/flows/__init__.py
```

```python
# file: cfm_grn_v2/src/cfm_grn_v2/flows/path.py
"""Linear probability path and the conditional flow matching objective; cubic-spline paths (v0.1.6).

``flow_time`` (s in [0, 1]) parametrises the generative path between a source sample x0
and a target sample x1. It is a mathematical construction, not biological time: the
latter only reaches the model through the conditioning ``context``.
"""
from __future__ import annotations

import numpy as np
import torch


def sample_flow_time(batch_size: int, generator: torch.Generator | None = None, device=None) -> torch.Tensor:
    """Uniform flow_time in [0, 1)."""
    return torch.rand(batch_size, generator=generator, device=device)


def interpolate(x0: torch.Tensor, x1: torch.Tensor, flow_time: torch.Tensor, sigma_min: float = 1e-3,
                noise: torch.Tensor | None = None) -> torch.Tensor:
    """x_s = (1 - s) x0 + s x1 + sigma_min * eps."""
    if x0.shape != x1.shape:
        raise ValueError(f"x0 {tuple(x0.shape)} and x1 {tuple(x1.shape)} must have the same shape")
    s = flow_time.view(-1, *([1] * (x0.dim() - 1))).to(x0.dtype)
    if noise is None:
        noise = torch.randn_like(x0)
    return (1.0 - s) * x0 + s * x1 + sigma_min * noise


def velocity_target(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Conditional velocity of the linear path (constant along the segment)."""
    return x1 - x0


def cfm_loss(vector_field, x0: torch.Tensor, x1: torch.Tensor, context, flow_time: torch.Tensor,
             sigma_min: float = 1e-3, target_scale: float = 1.0) -> torch.Tensor:
    """Mean squared error between v(x_s, s, context) and the conditional velocity target_scale * (x1 - x0)
    (target_scale = 1/dt turns the displacement into a velocity per unit of biological time)."""
    x_s = interpolate(x0, x1, flow_time, sigma_min)
    return ((vector_field(x_s, flow_time, context) - float(target_scale) * velocity_target(x0, x1)) ** 2).mean()


class SplinePath:
    """Cubic spline (natural boundary) through chained pseudo-trajectories: times (m,), trajectories (m, n, D).

    Sampling a biological time u ~ U[t_0, t_{m-1}] gives a state on the curve and the curve's biological
    velocity, the regression target of an autonomous field v(x, condition). Curved paths let the field
    learn the curvature between observed snapshots, which straight (McCann) interpolants cannot express.
    """

    def __init__(self, times, trajectories, bc_type: str = "natural", center: str = "spline"):
        from scipy.interpolate import CubicSpline
        if bc_type not in ("natural", "not-a-knot", "clamped"):
            raise ValueError(f"unknown spline boundary condition {bc_type!r}")
        if center not in ("spline", "linear"):
            raise ValueError(f"unknown path centre {center!r} (expected 'spline' or 'linear')")
        t = np.asarray(times, dtype=np.float64)
        y = np.asarray(trajectories, dtype=np.float64)
        if t.ndim != 1 or y.ndim != 3 or len(t) != y.shape[0]:
            raise ValueError("times must have shape (m,) and trajectories (m, n, D) with the same m")
        if len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("times must be strictly increasing with at least two entries")
        self.times, self.n, self.dim = t, int(y.shape[1]), int(y.shape[2])
        self.coef = CubicSpline(t, y, axis=0, bc_type=bc_type).c        # (4, m-1, n, D): c0 dt^3 + ... + c3
        self.center = center
        if center == "linear":
            # The mean over trajectories of a spline through chained cells is the spline through the knot means
            # (the spline is linear in its knots). "linear" swaps that curve for straight segments between the
            # knot means and keeps each trajectory's deviation from it: the spline supplies the shape, the data
            # the centre. The two centres agree at the knots, so the path still passes through observed cells.
            means = y.mean(1)                                            # (m, D)
            self.mean_coef = CubicSpline(t, means, axis=0, bc_type=bc_type).c
            self.knot_means = means

    def evaluate(self, u, idx):
        """State and velocity of trajectories idx at times u (both arrays of shape (b,))."""
        u, idx = np.asarray(u, dtype=np.float64), np.asarray(idx, dtype=int)
        j = np.clip(np.searchsorted(self.times, u, side="right") - 1, 0, len(self.times) - 2)
        dt = (u - self.times[j])[:, None]
        c = self.coef[:, j, idx, :]                                    # (4, b, D)
        x = ((c[0] * dt + c[1]) * dt + c[2]) * dt + c[3]
        v = (3.0 * c[0] * dt + 2.0 * c[1]) * dt + c[2]
        if self.center == "linear":
            cm = self.mean_coef[:, j, :]                               # (4, b, D)
            xm = ((cm[0] * dt + cm[1]) * dt + cm[2]) * dt + cm[3]
            vm = (3.0 * cm[0] * dt + 2.0 * cm[1]) * dt + cm[2]
            h = (self.times[j + 1] - self.times[j])[:, None]
            a, b = self.knot_means[j], self.knot_means[j + 1]
            x = x - xm + a + (dt / h) * (b - a)
            v = v - vm + (b - a) / h
        return x, v

    def sample(self, rng: np.random.Generator, batch_size: int, window: float = 0.0):
        """(x, v, u) for batch_size random (trajectory, biological time) pairs.

        ``window`` > 0 restricts u to a neighbourhood of the observed times: the interpolant is anchored by
        data at its knots and is a guess in the middle of a long gap, so a windowed sample supervises the field
        only where the path is supported and leaves the interior of long gaps to the field's own smoothness.
        It is meant to be combined with the marginal-matching stage, which constrains the endpoints of those
        gaps from the data (see flows/marginal.py)."""
        idx = rng.integers(0, self.n, size=batch_size)
        if window and window > 0:
            k = rng.integers(0, len(self.times), size=batch_size)
            lo = np.maximum(self.times[k] - float(window), self.times[0])
            hi = np.minimum(self.times[k] + float(window), self.times[-1])
            u = rng.uniform(lo, hi)
        else:
            u = rng.uniform(self.times[0], self.times[-1], size=batch_size)
        x, v = self.evaluate(u, idx)
        return x, v, u


def spline_loss(vector_field, x: torch.Tensor, v: torch.Tensor, context, sigma_min: float = 1e-3,
                time: torch.Tensor | None = None) -> torch.Tensor:
    """MSE between the field at a noisy state on the curve and the curve's biological velocity. ``time`` is the
    (scaled) biological time of each state for a non-autonomous field; None for the autonomous one."""
    x_n = x + sigma_min * torch.randn_like(x)
    t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device) if time is None else time
    return ((vector_field(x_n, t, context) - v) ** 2).mean()
```

- [ ] **Step 4: `python -m pytest tests/test_path.py -q` → 5 passed**

### Task 3: Coupling entro gruppi compatibili

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/flows/coupling.py`
- Test: `cfm_grn_v2/tests/test_coupling.py`

**Interfaces:**
- Produces: `independent_coupling(n0, n1, rng, size=None) -> (i0, i1)`; `chain_ot(xs: list[ndarray], rng, max_cells=1000, mode="ot"|"random") -> ndarray[m, n, D]` (pseudo-traiettorie per assegnamenti OT concatenati; `random` = ablation senza OT); `ot_coupling(x0, x1, rng, max_size=512) -> (i0, i1)` (assegnamento esatto a costo euclideo quadratico su un sottocampione bilanciato); `couple_within_groups(x0, meta0, x1, meta1, keys=("condition","replicate"), method="ot", rng=None, max_size=512) -> (i0, i1)`; `IncompatibleCouplingError`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_coupling.py
import numpy as np
import pytest

from cfm_grn_v2.flows.coupling import (IncompatibleCouplingError, chain_ot, couple_within_groups, independent_coupling,
                                       ot_coupling)


def test_chain_ot_recovers_identifiable_trajectories():
    rng = np.random.default_rng(0)
    base = 10.0 * np.arange(6)[:, None] * np.ones((6, 2))
    v = np.array([0.3, -0.2])
    xs = [rng.permutation(base + k * v) for k in range(4)]
    out = chain_ot(xs, rng, max_cells=100)
    assert out.shape == (4, 6, 2)
    for k in range(1, 4):
        assert np.allclose(out[k] - out[0], k * v)
    with pytest.raises(IncompatibleCouplingError):
        chain_ot([xs[0]], rng)
    rnd = chain_ot(xs, np.random.default_rng(0), max_cells=100, mode="random")
    assert rnd.shape == (4, 6, 2) and not all(np.allclose(rnd[k] - rnd[0], k * v) for k in range(1, 4))
    with pytest.raises(ValueError):
        chain_ot(xs, rng, mode="nearest")


def test_ot_coupling_recovers_translation_pairs():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(50, 2))
    x1 = x0 + 5.0
    i0, i1 = ot_coupling(x0, x1, rng, max_size=50)
    assert len(i0) == 50 and len(np.unique(i0)) == 50 and len(np.unique(i1)) == 50
    assert np.allclose(x0[i0] + 5.0, x1[i1])


def test_ot_cost_not_worse_than_independent():
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(200, 4))
    x1 = rng.normal(size=(200, 4)) + 1.0
    i0, i1 = ot_coupling(x0, x1, np.random.default_rng(2), max_size=200)
    j0, j1 = independent_coupling(200, 200, np.random.default_rng(3))
    assert ((x0[i0] - x1[i1]) ** 2).sum(1).mean() <= ((x0[j0] - x1[j1]) ** 2).sum(1).mean()


def test_independent_coupling_size():
    i0, i1 = independent_coupling(30, 50, np.random.default_rng(0), size=20)
    assert len(i0) == len(i1) == 20 and i0.max() < 30 and i1.max() < 50


def test_groups_never_mix():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(40, 3))
    x1 = rng.normal(size=(40, 3))
    meta0 = {"condition": ["a"] * 20 + ["b"] * 20, "replicate": ["r1"] * 40}
    meta1 = {"condition": ["b"] * 20 + ["a"] * 20, "replicate": ["r1"] * 40}
    i0, i1 = couple_within_groups(x0, meta0, x1, meta1, rng=rng)
    assert (np.asarray(meta0["condition"])[i0] == np.asarray(meta1["condition"])[i1]).all()
    assert len(i0) == 40


def test_missing_group_raises():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(10, 2))
    x1 = rng.normal(size=(10, 2))
    meta0 = {"condition": ["a"] * 10, "replicate": ["r1"] * 10}
    meta1 = {"condition": ["b"] * 10, "replicate": ["r1"] * 10}
    with pytest.raises(IncompatibleCouplingError):
        couple_within_groups(x0, meta0, x1, meta1, rng=rng)
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/flows/coupling.py
"""Couplings between source and target minibatches.

A coupling is always computed WITHIN a compatible group (same condition, replicate and
biological time pair). Any permutation applied to the target is applied to its metadata
through the returned index arrays, so cells and labels can never drift apart.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class IncompatibleCouplingError(ValueError):
    """Raised when a source group has no compatible target group."""


def independent_coupling(n0: int, n1: int, rng: np.random.Generator, size: int | None = None):
    """Random pairs of indices (no optimal transport)."""
    n = min(n0, n1) if size is None else min(n0, n1, size)
    return rng.choice(n0, n, replace=False), rng.choice(n1, n, replace=False)


def ot_coupling(x0: np.ndarray, x1: np.ndarray, rng: np.random.Generator, max_size: int = 512):
    """Exact minibatch optimal transport (squared Euclidean cost) via the Hungarian assignment.

    Both sides are subsampled to n = min(len(x0), len(x1), max_size) so the problem is balanced.
    Returns index arrays (i0, i1) such that x0[i0[k]] is paired with x1[i1[k]].
    """
    n = min(len(x0), len(x1), max_size)
    if n < 1:
        raise IncompatibleCouplingError("cannot couple empty sets")
    i0 = rng.choice(len(x0), n, replace=False)
    i1 = rng.choice(len(x1), n, replace=False)
    a = np.asarray(x0[i0], dtype=np.float64)
    b = np.asarray(x1[i1], dtype=np.float64)
    cost = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    row, col = linear_sum_assignment(cost)
    return i0[row], i1[col]


def chain_ot(xs: list, rng: np.random.Generator, max_cells: int = 1000, mode: str = "ot") -> np.ndarray:
    """Chain assignments across an ordered list of snapshots x_0, ..., x_{m-1} so that row k of the result
    follows one pseudo-trajectory: every snapshot is subsampled to n = min(len(x_k), max_cells) cells and
    consecutive snapshots are matched by the exact OT (Hungarian, squared Euclidean cost) assignment, or,
    with mode="random", by a random permutation (the ablation that removes optimal transport).
    Returns an array of shape (m, n, genes)."""
    if mode not in ("ot", "random"):
        raise ValueError(f"unknown chain mode {mode!r} (expected 'ot' or 'random')")
    if len(xs) < 2:
        raise IncompatibleCouplingError("chaining needs at least two snapshots")
    n = min(min(len(x) for x in xs), max_cells)
    if n < 1:
        raise IncompatibleCouplingError("cannot chain empty snapshots")
    cur = np.asarray(xs[0], dtype=np.float64)[rng.choice(len(xs[0]), n, replace=False)]
    out = [cur]
    for x in xs[1:]:
        nxt = np.asarray(x, dtype=np.float64)[rng.choice(len(x), n, replace=False)]
        if mode == "ot":
            cost = (cur * cur).sum(1)[:, None] + (nxt * nxt).sum(1)[None, :] - 2.0 * cur @ nxt.T
            row, col = linear_sum_assignment(cost)
            order = np.empty(n, dtype=int)
            order[row] = col
        else:
            order = rng.permutation(n)
        nxt = nxt[order]
        out.append(nxt)
        cur = nxt
    return np.stack(out)


def group_keys(meta: dict, keys: tuple) -> np.ndarray:
    cols = [np.asarray(meta[k]).astype(str) for k in keys]
    return np.array(["|".join(t) for t in zip(*cols)])


def couple_within_groups(x0, meta0, x1, meta1, keys=("condition", "replicate"), method="ot",
                         rng: np.random.Generator | None = None, max_size: int = 512):
    """Couple source and target cells group by group. Returns (i0, i1) with equal group keys."""
    rng = np.random.default_rng() if rng is None else rng
    g0 = group_keys(meta0, keys)
    g1 = group_keys(meta1, keys)
    out0, out1 = [], []
    for g in np.unique(g0):
        idx0 = np.flatnonzero(g0 == g)
        idx1 = np.flatnonzero(g1 == g)
        if len(idx1) == 0:
            raise IncompatibleCouplingError(f"group {g!r} has source cells but no target cells")
        if method == "ot":
            a, b = ot_coupling(x0[idx0], x1[idx1], rng, max_size)
        elif method == "independent":
            a, b = independent_coupling(len(idx0), len(idx1), rng, size=max_size)
        else:
            raise ValueError(f"unknown coupling method {method!r}")
        out0.append(idx0[a])
        out1.append(idx1[b])
    i0 = np.concatenate(out0)
    i1 = np.concatenate(out1)
    if not (g0[i0] == g1[i1]).all():
        raise IncompatibleCouplingError("internal error: coupled cells across groups")
    return i0, i1
```

- [ ] **Step 4: `python -m pytest tests/test_coupling.py -q` → 5 passed**

### Task 4: Solver ODE in `flow_time`

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/flows/solver.py`
- Test: `cfm_grn_v2/tests/test_solver.py`

**Interfaces:**
- Produces: `integrate(vector_field, x0, context, n_steps=20, method="rk4", return_trajectory=False, t_end=1.0)` → `x1` oppure `(x1, trajectory[n_steps+1, B, D])`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_solver.py
import math

import pytest
import torch

from cfm_grn_v2.flows.solver import integrate


def test_constant_field_translates():
    x0 = torch.zeros(3, 2)
    c = torch.tensor([1.0, -1.0])
    x1 = integrate(lambda x, s, ctx: c.expand_as(x), x0, None, n_steps=10, method="euler")
    assert torch.allclose(x1, c.expand_as(x0), atol=1e-6)


@pytest.mark.parametrize("method,tol", [("euler", 0.2), ("midpoint", 1e-2), ("rk4", 1e-4)])
def test_linear_field_accuracy(method, tol):
    x0 = torch.ones(4, 3)
    x1 = integrate(lambda x, s, ctx: x, x0, None, n_steps=20, method=method)
    assert float((x1 - math.e * x0).abs().max()) < tol


def test_trajectory_shape_and_endpoints():
    x0 = torch.randn(5, 2)
    x1, traj = integrate(lambda x, s, ctx: torch.zeros_like(x), x0, None, n_steps=7, return_trajectory=True)
    assert traj.shape == (8, 5, 2)
    assert torch.allclose(traj[0], x0) and torch.allclose(traj[-1], x1)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        integrate(lambda x, s, ctx: x, torch.zeros(1, 1), None, method="leapfrog")


def test_t_end_scales_the_horizon():
    x0 = torch.zeros(3, 2)
    c = torch.tensor([1.0, -1.0])
    x1 = integrate(lambda x, s, ctx: c.expand_as(x), x0, None, n_steps=10, method="rk4", t_end=2.5)
    assert torch.allclose(x1, 2.5 * c.expand_as(x0), atol=1e-6)
    with pytest.raises(ValueError):
        integrate(lambda x, s, ctx: x, torch.zeros(1, 1), None, t_end=0.0)
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/flows/solver.py
"""Fixed-step ODE integration of dx/ds = v(x, s, context) over s in [0, t_end] (t_end = 1 for the flow clock,
the biological horizon for the duration clock)."""
from __future__ import annotations

import torch


def integrate(vector_field, x0: torch.Tensor, context, n_steps: int = 20, method: str = "rk4",
              return_trajectory: bool = False, t_end: float = 1.0):
    if method not in ("euler", "midpoint", "rk4"):
        raise ValueError(f"unknown ODE method {method!r}")
    if t_end <= 0:
        raise ValueError("t_end must be positive")
    x = x0
    grid = torch.linspace(0.0, float(t_end), n_steps + 1, device=x0.device, dtype=x0.dtype)
    traj = [x0]
    for k in range(n_steps):
        s0, h = grid[k], grid[k + 1] - grid[k]
        t0 = s0.expand(x.shape[0])
        if method == "euler":
            x = x + h * vector_field(x, t0, context)
        elif method == "midpoint":
            k1 = vector_field(x, t0, context)
            x = x + h * vector_field(x + 0.5 * h * k1, t0 + 0.5 * h, context)
        else:
            k1 = vector_field(x, t0, context)
            k2 = vector_field(x + 0.5 * h * k1, t0 + 0.5 * h, context)
            k3 = vector_field(x + 0.5 * h * k2, t0 + 0.5 * h, context)
            k4 = vector_field(x + h * k3, t0 + h, context)
            x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if return_trajectory:
            traj.append(x)
    if return_trajectory:
        return x, torch.stack(traj)
    return x
```

- [ ] **Step 4: `python -m pytest tests/test_solver.py -q` → 6 passed**

### Task 5: Condizionamento e campo vettoriale

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/models/__init__.py`, `cfm_grn_v2/src/cfm_grn_v2/models/conditioning.py`, `cfm_grn_v2/src/cfm_grn_v2/models/vector_field.py`
- Test: `cfm_grn_v2/tests/test_models.py`

**Interfaces:**
- Produces: `ConditionRegistry(names)` con `.names`, `.encode(list[str]) -> LongTensor`, `len()`, `.to_dict()/.from_dict()`, `UnknownConditionError`; `ContextEncoder(registry, cond_dim=8, time_scale=1.0, time_conditioning="delta")` (`delta` = solo Δt, `absolute` = t0, t1, Δt, `none`) con `forward(cond_idx, bio_t0, bio_t1) -> Tensor[B, out_dim]`, `.out_dim`; `sinusoidal_embedding(flow_time, dim)`; `MLPVectorField(dim, context_dim, hidden=256, depth=3, time_embed_dim=32, use_time=True)` con `forward(x, flow_time, context)` (`use_time=False`: campo autonomo, `flow_time` ignorato).

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_models.py
import inspect

import pytest
import torch

from cfm_grn_v2.models.conditioning import ConditionRegistry, ContextEncoder, UnknownConditionError
from cfm_grn_v2.models.vector_field import MLPVectorField, sinusoidal_embedding


def test_registry_roundtrip_and_unknown():
    reg = ConditionRegistry(["stim", "ctrl", "ctrl"])
    assert reg.names == ["ctrl", "stim"] and len(reg) == 2
    assert reg.encode(["stim", "ctrl"]).tolist() == [1, 0]
    with pytest.raises(UnknownConditionError):
        reg.encode(["never_seen"])
    assert ConditionRegistry.from_dict(reg.to_dict()).names == reg.names


def test_context_encoder_never_receives_flow_time():
    params = inspect.signature(ContextEncoder.forward).parameters
    assert "flow_time" not in params and "bio_t0" in params and "bio_t1" in params


def test_context_encoder_output():
    enc = ContextEncoder(ConditionRegistry(["a", "b"]), cond_dim=4, time_scale=10.0, time_conditioning="absolute")
    out = enc(torch.tensor([0, 1, 1]), torch.tensor([0.0, 5.0, 5.0]), torch.tensor([5.0, 10.0, 10.0]))
    assert out.shape == (3, enc.out_dim) == (3, 7)
    assert torch.allclose(out[1], out[2])
    assert torch.allclose(out[0, -3:], torch.tensor([0.0, 0.5, 0.5]))


def test_context_encoder_delta_mode_ignores_absolute_time():
    enc = ContextEncoder(ConditionRegistry(["a"]), cond_dim=2, time_scale=1.0, time_conditioning="delta")
    a = enc(torch.tensor([0]), torch.tensor([0.0]), torch.tensor([1.0]))
    b = enc(torch.tensor([0]), torch.tensor([5.0]), torch.tensor([6.0]))
    assert enc.out_dim == 3 and torch.allclose(a, b)
    with pytest.raises(ValueError):
        ContextEncoder(ConditionRegistry(["a"]), time_conditioning="sinusoidal")


def test_sinusoidal_embedding_shape():
    emb = sinusoidal_embedding(torch.tensor([0.0, 0.5, 1.0]), 16)
    assert emb.shape == (3, 16) and not torch.allclose(emb[0], emb[1])


def test_autonomous_vector_field_ignores_flow_time():
    torch.manual_seed(0)
    vf = MLPVectorField(dim=4, context_dim=2, hidden=8, depth=1, time_embed_dim=8, use_time=False)
    x, c = torch.randn(5, 4), torch.randn(5, 2)
    assert torch.allclose(vf(x, torch.zeros(5), c), vf(x, torch.ones(5), c))
    assert vf.net[0].in_features == 6


def test_vector_field_shapes_and_context_required():
    vf = MLPVectorField(dim=5, context_dim=3, hidden=16, depth=2, time_embed_dim=8)
    x = torch.randn(6, 5)
    out = vf(x, torch.rand(6), torch.randn(6, 3))
    assert out.shape == (6, 5)
    out.sum().backward()
    assert vf.net[0].weight.grad is not None
    with pytest.raises(ValueError):
        vf(x, torch.rand(6), None)


def test_vector_field_scalar_time_input():
    """time_embed="scalar": the field sees the time as one raw feature (smooth in biological time) instead of
    the high-frequency sinusoidal flow-time embedding."""
    torch.manual_seed(0)
    f = MLPVectorField(3, 0, hidden=16, depth=2, use_time=True, time_embed="scalar")
    assert f.net[0].in_features == 3 + 1
    x = torch.randn(5, 3)
    a, b = f(x, torch.zeros(5)), f(x, torch.ones(5))
    assert a.shape == (5, 3) and not torch.allclose(a, b)
    with pytest.raises(ValueError):
        MLPVectorField(3, 0, time_embed="fourier")
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/models/__init__.py
```

```python
# file: cfm_grn_v2/src/cfm_grn_v2/models/conditioning.py
"""Conditioning context built from biological_time and condition labels.

The context never contains ``flow_time``: the generative clock enters the vector field
separately, so the two clocks cannot be confused inside the model.
"""
from __future__ import annotations

import torch
from torch import nn


class UnknownConditionError(KeyError):
    """Raised when a condition label was not present at training time."""


class ConditionRegistry:
    """Maps condition names to embedding indices; refuses unknown names loudly."""

    def __init__(self, names):
        self.names = sorted({str(n) for n in names})
        self._index = {n: i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def encode(self, names) -> torch.Tensor:
        idx = []
        for n in names:
            if str(n) not in self._index:
                raise UnknownConditionError(f"condition {n!r} not in registry {self.names}; "
                                            "a condition unseen in training cannot be represented")
            idx.append(self._index[str(n)])
        return torch.tensor(idx, dtype=torch.long)

    def to_dict(self) -> dict:
        return {"names": list(self.names)}

    @classmethod
    def from_dict(cls, d: dict) -> "ConditionRegistry":
        return cls(d["names"])


class ContextEncoder(nn.Module):
    """context = [embedding(condition), time features].

    time_conditioning:
      "delta"    -> [(t1 - t0)/scale]                      time-homogeneous dynamics; extrapolates to unseen times
      "absolute" -> [t0/scale, t1/scale, (t1 - t0)/scale]  non-autonomous; cannot extrapolate past training times
      "none"     -> []
    """

    MODES = ("delta", "absolute", "none")

    def __init__(self, registry: ConditionRegistry, cond_dim: int = 8, time_scale: float = 1.0,
                 time_conditioning: str = "delta"):
        super().__init__()
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if time_conditioning not in self.MODES:
            raise ValueError(f"time_conditioning must be one of {self.MODES}, got {time_conditioning!r}")
        self.registry = registry
        self.cond_dim = int(cond_dim)
        self.time_scale = float(time_scale)
        self.time_conditioning = time_conditioning
        self.cond_embed = nn.Embedding(len(registry), self.cond_dim)

    @property
    def n_time_features(self) -> int:
        return {"delta": 1, "absolute": 3, "none": 0}[self.time_conditioning]

    @property
    def out_dim(self) -> int:
        return self.cond_dim + self.n_time_features

    def forward(self, cond_idx: torch.Tensor, bio_t0: torch.Tensor, bio_t1: torch.Tensor) -> torch.Tensor:
        t0 = (bio_t0.to(torch.float32) / self.time_scale)[:, None]
        t1 = (bio_t1.to(torch.float32) / self.time_scale)[:, None]
        parts = [self.cond_embed(cond_idx)]
        if self.time_conditioning == "delta":
            parts.append(t1 - t0)
        elif self.time_conditioning == "absolute":
            parts += [t0, t1, t1 - t0]
        return torch.cat(parts, dim=-1)
```

```python
# file: cfm_grn_v2/src/cfm_grn_v2/models/vector_field.py
"""MLP vector field v(x, flow_time, context) -> dx/ds."""
from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(flow_time: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features of flow_time in [0, 1] (scaled by 1000 like diffusion timesteps)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=flow_time.device, dtype=torch.float32) / max(half, 1))
    args = flow_time.to(torch.float32)[:, None] * 1000.0 * freqs[None, :]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLPVectorField(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden: int = 256, depth: int = 3, time_embed_dim: int = 32,
                 use_time: bool = True, time_embed: str = "sinusoidal"):
        super().__init__()
        if time_embed not in ("sinusoidal", "scalar"):
            raise ValueError(f"unknown time_embed {time_embed!r} (expected 'sinusoidal' or 'scalar')")
        self.dim, self.context_dim, self.time_embed_dim = int(dim), int(context_dim), int(time_embed_dim)
        self.use_time = bool(use_time)            # False: autonomous field v(x, context), flow_time ignored
        self.time_embed = time_embed              # "scalar": the time as one raw feature (biological time, v0.1.11)
        t_width = (self.time_embed_dim if time_embed == "sinusoidal" else 1) if self.use_time else 0
        layers, width = [], self.dim + t_width + self.context_dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, self.dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, flow_time: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        parts = [x]
        if self.use_time:
            parts.append(flow_time.to(torch.float32)[:, None] if self.time_embed == "scalar"
                         else sinusoidal_embedding(flow_time, self.time_embed_dim))
        if self.context_dim > 0:
            if context is None:
                raise ValueError("this vector field expects a context tensor (biological_time + condition)")
            parts.append(context)
        return self.net(torch.cat(parts, dim=-1))
```

- [ ] **Step 4: `python -m pytest tests/test_models.py -q` → 6 passed**

### Task 6: Schema dei dati

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/data/__init__.py`, `cfm_grn_v2/src/cfm_grn_v2/data/schema.py`
- Test: `cfm_grn_v2/tests/test_schema.py`

**Interfaces:**
- Produces: `SnapshotSet(x, biological_time, condition, replicate, gene_names, name="unnamed", require_two_timepoints=True)` con `n_cells`, `n_genes`, `timepoints()`, `conditions()`, `replicates()`, `mask(times=None, conditions=None, replicates=None) -> bool[N]`, `subset(mask)`, `summary() -> DataFrame`; `SchemaError`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_schema.py
import numpy as np
import pytest

from cfm_grn_v2.data.schema import SchemaError, SnapshotSet


def _snaps():
    x = np.random.default_rng(0).normal(size=(6, 2)).astype(np.float32)
    return SnapshotSet(x, [0, 0, 0, 1, 1, 1], ["a", "a", "b", "a", "b", "b"], ["r1"] * 6, ["g1", "g2"])


def test_valid_snapshot_set():
    s = _snaps()
    assert s.n_cells == 6 and s.n_genes == 2
    assert s.timepoints().tolist() == [0.0, 1.0] and s.conditions() == ["a", "b"] and s.replicates() == ["r1"]


def test_mask_and_subset():
    s = _snaps()
    m = s.mask(times=[1.0], conditions=["b"])
    assert m.sum() == 2
    sub = s.subset(m)
    assert sub.n_cells == 2 and (sub.condition == "b").all()


def test_summary_counts():
    df = _snaps().summary()
    assert df["n_cells"].sum() == 6 and set(df.columns) == {"biological_time", "condition", "replicate", "n_cells"}


def test_length_mismatch_raises():
    with pytest.raises(SchemaError):
        SnapshotSet(np.zeros((3, 2)), [0, 1], ["a"] * 3, ["r"] * 3, ["g1", "g2"])


def test_nan_raises():
    x = np.zeros((4, 2))
    x[0, 0] = np.nan
    with pytest.raises(SchemaError):
        SnapshotSet(x, [0, 0, 1, 1], ["a"] * 4, ["r"] * 4, ["g1", "g2"])


def test_single_timepoint_raises():
    with pytest.raises(SchemaError):
        SnapshotSet(np.zeros((4, 2)), [0, 0, 0, 0], ["a"] * 4, ["r"] * 4, ["g1", "g2"])


def test_duplicate_gene_names_raise():
    with pytest.raises(SchemaError):
        SnapshotSet(np.zeros((4, 2)), [0, 0, 1, 1], ["a"] * 4, ["r"] * 4, ["g", "g"])
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/data/__init__.py
```

```python
# file: cfm_grn_v2/src/cfm_grn_v2/data/schema.py
"""Snapshot data schema: unpaired cells observed at biological times under conditions."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """Raised when a SnapshotSet is malformed."""


@dataclass
class SnapshotSet:
    x: np.ndarray                  # [n_cells, n_genes] float32
    biological_time: np.ndarray    # [n_cells] measured hours / days / stages
    condition: np.ndarray          # [n_cells] str, e.g. "ctrl", "ko_GENE", "stim"
    replicate: np.ndarray          # [n_cells] str, biological unit used for splits
    gene_names: list
    name: str = "unnamed"
    require_two_timepoints: bool = True

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float32)
        if self.x.ndim != 2:
            raise SchemaError(f"x must be 2-D [cells, genes], got shape {self.x.shape}")
        n = self.x.shape[0]
        self.biological_time = np.asarray(self.biological_time, dtype=np.float64).reshape(-1)
        self.condition = np.asarray(self.condition).astype(str).reshape(-1)
        self.replicate = np.asarray(self.replicate).astype(str).reshape(-1)
        self.gene_names = [str(g) for g in self.gene_names]
        for field_name, arr in (("biological_time", self.biological_time), ("condition", self.condition),
                                ("replicate", self.replicate)):
            if arr.shape[0] != n:
                raise SchemaError(f"{field_name} has {arr.shape[0]} entries, x has {n} cells")
        if len(self.gene_names) != self.x.shape[1]:
            raise SchemaError(f"{len(self.gene_names)} gene names for {self.x.shape[1]} columns")
        if len(set(self.gene_names)) != len(self.gene_names):
            raise SchemaError("gene names must be unique")
        if not np.isfinite(self.x).all():
            raise SchemaError("x contains NaN or inf")
        if not np.isfinite(self.biological_time).all():
            raise SchemaError("biological_time contains NaN or inf")
        if self.require_two_timepoints and len(self.timepoints()) < 2:
            raise SchemaError("a SnapshotSet needs at least two biological timepoints")

    @property
    def n_cells(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.x.shape[1])

    def timepoints(self) -> np.ndarray:
        return np.unique(self.biological_time)

    def conditions(self) -> list:
        return sorted(set(self.condition.tolist()))

    def replicates(self) -> list:
        return sorted(set(self.replicate.tolist()))

    def mask(self, times=None, conditions=None, replicates=None) -> np.ndarray:
        m = np.ones(self.n_cells, dtype=bool)
        if times is not None:
            t = np.asarray(list(times), dtype=np.float64)
            m &= np.isclose(self.biological_time[:, None], t[None, :]).any(axis=1)
        if conditions is not None:
            m &= np.isin(self.condition, [str(c) for c in conditions])
        if replicates is not None:
            m &= np.isin(self.replicate, [str(r) for r in replicates])
        return m

    def subset(self, mask: np.ndarray) -> "SnapshotSet":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_cells,):
            raise SchemaError("mask must have one boolean entry per cell")
        return replace(self, x=self.x[mask], biological_time=self.biological_time[mask],
                       condition=self.condition[mask], replicate=self.replicate[mask],
                       require_two_timepoints=False)

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame({"biological_time": self.biological_time, "condition": self.condition,
                           "replicate": self.replicate})
        out = df.value_counts().rename("n_cells").reset_index()
        return out.sort_values(["condition", "replicate", "biological_time"]).reset_index(drop=True)
```

- [ ] **Step 4: `python -m pytest tests/test_schema.py -q` → 7 passed**

### Task 7: Toy 2D con risposta nota

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/data/synthetic_toys.py`
- Test: `cfm_grn_v2/tests/test_toys.py`

**Interfaces:**
- Produces: `TOY_NAMES = ("translation", "variance", "bimodal", "condition_shift")`; `make_toy(name, n_per_group=1000, replicates=("r1","r2","r3"), seed=0) -> SnapshotSet` con `biological_time ∈ {0.0, 1.0}`, condizioni `ctrl` (e `stim` per `condition_shift`), geni `dim0`, `dim1`, `name = "toy_<name>"`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_toys.py
import numpy as np
import pytest

from cfm_grn_v2.data.synthetic_toys import TOY_NAMES, make_toy


@pytest.mark.parametrize("name", TOY_NAMES)
def test_toys_have_two_times_and_replicates(name):
    s = make_toy(name, n_per_group=200, seed=1)
    assert s.timepoints().tolist() == [0.0, 1.0]
    assert s.replicates() == ["r1", "r2", "r3"] and s.n_genes == 2 and s.name == f"toy_{name}"


def test_translation_moves_the_mean():
    s = make_toy("translation", n_per_group=2000)
    m0 = s.x[s.mask(times=[0.0])].mean(0)
    m1 = s.x[s.mask(times=[1.0])].mean(0)
    assert np.allclose(m1 - m0, [3.0, -2.0], atol=0.15)


def test_condition_shift_has_opposite_shifts():
    s = make_toy("condition_shift", n_per_group=2000)
    assert s.conditions() == ["ctrl", "stim"]
    d = {c: s.x[s.mask(times=[1.0], conditions=[c])].mean(0)[0] - s.x[s.mask(times=[0.0], conditions=[c])].mean(0)[0]
         for c in s.conditions()}
    assert d["ctrl"] > 1.5 and d["stim"] < -1.5


def test_unknown_toy_raises():
    with pytest.raises(ValueError):
        make_toy("spiral")
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/data/synthetic_toys.py
"""Two-dimensional toy transitions with known answers (biological_time 0 -> 1).

translation      : N(0, I) -> N((3, -2), I)
variance         : N(0, I) -> N(0, diag(4, 0.25))
bimodal          : N(0, I) -> 0.5 N((3, 3), 0.25 I) + 0.5 N((-3, -3), 0.25 I)
condition_shift  : ctrl -> +2 on dim0, stim -> -2 on dim0 (independent cohorts per time)
Every replicate is an independent cohort, so replicate-level splits are honest.
"""
from __future__ import annotations

import numpy as np

from .schema import SnapshotSet

TOY_NAMES = ("translation", "variance", "bimodal", "condition_shift")


def _sample_t1(name: str, cond: str, rng: np.random.Generator, n: int) -> np.ndarray:
    if name == "translation":
        return rng.normal(0.0, 1.0, size=(n, 2)) + np.array([3.0, -2.0])
    if name == "variance":
        return rng.normal(0.0, 1.0, size=(n, 2)) * np.array([2.0, 0.5])
    if name == "bimodal":
        sign = rng.choice([-1.0, 1.0], size=(n, 1))
        return rng.normal(0.0, 0.5, size=(n, 2)) + sign * 3.0
    if name == "condition_shift":
        shift = np.array([2.0, 0.0]) if cond == "ctrl" else np.array([-2.0, 0.0])
        return rng.normal(0.0, 1.0, size=(n, 2)) + shift
    raise ValueError(f"unknown toy {name!r}; choose one of {TOY_NAMES}")


def make_toy(name: str, n_per_group: int = 1000, replicates=("r1", "r2", "r3"), seed: int = 0) -> SnapshotSet:
    if name not in TOY_NAMES:
        raise ValueError(f"unknown toy {name!r}; choose one of {TOY_NAMES}")
    conditions = ("ctrl", "stim") if name == "condition_shift" else ("ctrl",)
    rng = np.random.default_rng(seed)
    xs, ts, cs, rs = [], [], [], []
    for rep in replicates:
        for cond in conditions:
            x0 = rng.normal(0.0, 1.0, size=(n_per_group, 2))
            x1 = _sample_t1(name, cond, rng, n_per_group)
            xs += [x0, x1]
            ts += [np.zeros(n_per_group), np.ones(n_per_group)]
            cs += [np.array([cond] * n_per_group)] * 2
            rs += [np.array([rep] * n_per_group)] * 2
    return SnapshotSet(np.concatenate(xs), np.concatenate(ts), np.concatenate(cs), np.concatenate(rs),
                       ["dim0", "dim1"], name=f"toy_{name}")
```

- [ ] **Step 4: `python -m pytest tests/test_toys.py -q` → 7 passed**

### Task 8: Split e preprocessing senza leakage

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/data/splits.py`, `cfm_grn_v2/src/cfm_grn_v2/data/preprocessing.py`
- Test: `cfm_grn_v2/tests/test_splits.py`, `cfm_grn_v2/tests/test_preprocessing.py`

**Interfaces:**
- Produces: `forecast_split(times, n_test=1) -> {"train_times", "test_times"}`; `inner_validation_split(train_times) -> {"fit_times", "val_times"}`; `interpolation_folds(times) -> list[dict]`; `replicate_split(replicates, n_val=1, n_test=1, seed=0) -> {"fit","val","test"}`; `grouped_folds(groups, n_folds, seed=0) -> list[{"train_groups","test_groups"}]`; `assert_disjoint(a, b, what="sets")`; `SplitError`; `Standardizer().fit(x).transform(x)`, `.inverse_transform`, `.to_dict/.from_dict`, attributi `mean_`, `std_`; `fit_on_train(x, train_mask) -> Standardizer`; `NotFittedError`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_splits.py
import pytest

from cfm_grn_v2.data.splits import (SplitError, assert_disjoint, forecast_split, grouped_folds,
                                    inner_validation_split, interpolation_folds, replicate_split)


def test_forecast_split_holds_out_terminal_time():
    s = forecast_split([3.0, 1.0, 2.0, 0.0])
    assert s["train_times"] == [0.0, 1.0, 2.0] and s["test_times"] == [3.0]
    inner = inner_validation_split(s["train_times"])
    assert inner["fit_times"] == [0.0, 1.0] and inner["val_times"] == [2.0]


def test_forecast_split_needs_three_times():
    with pytest.raises(SplitError):
        forecast_split([0.0, 1.0])


def test_interpolation_folds_only_inner_times():
    folds = interpolation_folds([0, 1, 2, 3])
    assert [f["test_times"] for f in folds] == [[1.0], [2.0]]
    assert folds[0]["train_times"] == [0.0, 2.0, 3.0]


def test_replicate_split_is_disjoint_and_complete():
    s = replicate_split(["r1", "r2", "r3", "r3"], seed=0)
    assert sorted(s["fit"] + s["val"] + s["test"]) == ["r1", "r2", "r3"]
    assert len(s["test"]) == 1 and len(s["val"]) == 1


def test_grouped_folds_cover_every_group_once():
    folds = grouped_folds(["a", "b", "c", "d", "e"], n_folds=2, seed=1)
    tested = sum((f["test_groups"] for f in folds), [])
    assert sorted(tested) == ["a", "b", "c", "d", "e"]
    for f in folds:
        assert not set(f["train_groups"]) & set(f["test_groups"])


def test_assert_disjoint():
    assert_disjoint([0.0, 1.0], [2.0])
    with pytest.raises(SplitError):
        assert_disjoint([0.0, 1.0], [1.0])
```

```python
# file: cfm_grn_v2/tests/test_preprocessing.py
import numpy as np
import pytest

from cfm_grn_v2.data.preprocessing import NotFittedError, Standardizer, fit_on_train


def test_standardizer_roundtrip_and_serialization():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=(100, 4))
    st = Standardizer().fit(x)
    z = st.transform(x)
    assert np.allclose(z.mean(0), 0, atol=1e-5) and np.allclose(z.std(0), 1, atol=1e-5)
    assert np.allclose(st.inverse_transform(z), x, atol=1e-4)
    st2 = Standardizer.from_dict(st.to_dict())
    assert np.allclose(st2.transform(x), z)


def test_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        Standardizer().transform(np.zeros((2, 2)))


def test_fit_on_train_ignores_test_cells():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    mask = np.zeros(50, dtype=bool)
    mask[:30] = True
    st = fit_on_train(x, mask)
    x2 = x.copy()
    x2[~mask] += 100.0
    st2 = fit_on_train(x2, mask)
    assert np.allclose(st.mean_, st2.mean_) and np.allclose(st.std_, st2.std_)
    with pytest.raises(ValueError):
        fit_on_train(x, np.ones(50, dtype=bool))
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/data/splits.py
"""Splits by whole biological units: timepoints, replicates, perturbation targets. Never by random cells."""
from __future__ import annotations

import numpy as np


class SplitError(ValueError):
    """Raised when a split cannot be built honestly."""


def _sorted_unique_times(times) -> np.ndarray:
    return np.unique(np.asarray(list(times), dtype=np.float64))


def forecast_split(times, n_test: int = 1) -> dict:
    """Hold out the LAST n_test timepoints (extrapolation in biological_time)."""
    t = _sorted_unique_times(times)
    if len(t) < n_test + 2:
        raise SplitError(f"need at least {n_test + 2} timepoints for a forecast split, got {len(t)}")
    return {"train_times": t[:-n_test].tolist(), "test_times": t[-n_test:].tolist()}


def inner_validation_split(train_times) -> dict:
    """Inside the training timepoints, the last one is the validation target."""
    t = _sorted_unique_times(train_times)
    if len(t) < 2:
        raise SplitError("need at least two training timepoints to hold one out for validation")
    return {"fit_times": t[:-1].tolist(), "val_times": [float(t[-1])]}


def interpolation_folds(times) -> list:
    """Leave-one-inner-timepoint-out folds (never the first or the last timepoint)."""
    t = _sorted_unique_times(times)
    if len(t) < 3:
        raise SplitError("interpolation needs at least three timepoints")
    return [{"train_times": np.delete(t, k).tolist(), "test_times": [float(t[k])]} for k in range(1, len(t) - 1)]


def replicate_split(replicates, n_val: int = 1, n_test: int = 1, seed: int = 0) -> dict:
    reps = sorted({str(r) for r in replicates})
    if len(reps) < n_val + n_test + 1:
        raise SplitError(f"need at least {n_val + n_test + 1} replicates, got {len(reps)}")
    order = [str(r) for r in np.random.default_rng(seed).permutation(reps)]
    return {"test": order[:n_test], "val": order[n_test:n_test + n_val], "fit": order[n_test + n_val:]}


def grouped_folds(groups, n_folds: int, seed: int = 0) -> list:
    """K folds over whole groups (e.g. perturbation targets); every group is tested exactly once."""
    g = sorted({str(x) for x in groups})
    if n_folds < 2 or n_folds > len(g):
        raise SplitError(f"n_folds must be between 2 and {len(g)}")
    order = np.random.default_rng(seed).permutation(g)
    return [{"train_groups": sorted(set(g) - set(f.tolist())), "test_groups": sorted(f.tolist())}
            for f in np.array_split(order, n_folds)]


def assert_disjoint(a, b, what: str = "sets") -> None:
    inter = set(a) & set(b)
    if inter:
        raise SplitError(f"{what} overlap: {sorted(inter)[:5]}")
```

```python
# file: cfm_grn_v2/src/cfm_grn_v2/data/preprocessing.py
"""Preprocessing fitted on training cells only and serialised with the model."""
from __future__ import annotations

import numpy as np


class NotFittedError(RuntimeError):
    """Raised when transform is called before fit."""


class Standardizer:
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.mean_ = None
        self.std_ = None

    def fit(self, x) -> "Standardizer":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError("Standardizer.fit needs a 2-D array with at least two rows")
        self.mean_ = x.mean(0)
        self.std_ = np.maximum(x.std(0), self.eps)
        return self

    def _check(self) -> None:
        if self.mean_ is None:
            raise NotFittedError("Standardizer.fit must run on training cells before transform")

    def transform(self, x) -> np.ndarray:
        self._check()
        return ((np.asarray(x, dtype=np.float64) - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, z) -> np.ndarray:
        self._check()
        return (np.asarray(z, dtype=np.float64) * self.std_ + self.mean_).astype(np.float32)

    def to_dict(self) -> dict:
        self._check()
        return {"eps": self.eps, "mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        st = cls(d["eps"])
        st.mean_ = np.asarray(d["mean"], dtype=np.float64)
        st.std_ = np.asarray(d["std"], dtype=np.float64)
        return st


def fit_on_train(x, train_mask) -> Standardizer:
    """Fit on the training cells only; refuses a mask that hides nothing."""
    train_mask = np.asarray(train_mask, dtype=bool)
    x = np.asarray(x)
    if train_mask.shape != (len(x),):
        raise ValueError("train_mask must have one boolean entry per cell")
    if train_mask.all():
        raise ValueError("train_mask covers every cell: nothing is held out")
    if train_mask.sum() < 2:
        raise ValueError("train_mask selects fewer than two cells")
    return Standardizer().fit(x[train_mask])
```

- [ ] **Step 4: `python -m pytest tests/test_splits.py tests/test_preprocessing.py -q` → 9 passed**

### Task 9: Metriche distribuzionali e protocollo

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/evaluation/__init__.py`, `cfm_grn_v2/src/cfm_grn_v2/evaluation/metrics.py`
- Test: `cfm_grn_v2/tests/test_metrics.py`

**Interfaces:**
- Produces: `energy_distance(a, b) -> float`; `median_bandwidth(x, rng=None, n_max=500) -> float`; `mmd_rbf(a, b, bandwidth) -> float`; `w2_assignment(a, b, rng, n_max=500) -> float`; `mean_error(a, b) -> float`; `per_gene_w1(a, b, rng, n_max=2000) -> float`; `EvalProtocol(n_samples=1000, seed=0, bandwidth=1.0, w2_max=500)` con `.subsample(x, rng)` e `.evaluate(pred, target, seed_offset=0) -> dict` (chiavi `energy_distance, mmd, w2, mean_error, per_gene_w1`); `noise_floor(target, protocol, n_rep=5) -> dict`; `permutation_null(pred, target, rng, n_perm=50, q=0.95) -> float` (chiave `null_q95` in `evaluate`, stesse numerosità della valutazione).

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_metrics.py
import numpy as np

from cfm_grn_v2.evaluation.metrics import (EvalProtocol, energy_distance, mean_error, median_bandwidth, mmd_rbf,
                                           noise_floor, per_gene_w1, w2_assignment)


def test_energy_distance_orders_shifts():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 3))
    b = rng.normal(size=(500, 3))
    near = energy_distance(a, b)
    far = energy_distance(a, b + 2.0)
    assert abs(near) < 0.05 and far > near + 1.0


def test_energy_distance_is_unbiased_at_small_n():
    rng = np.random.default_rng(3)
    vals = [energy_distance(rng.normal(size=(40, 3)), rng.normal(size=(40, 3))) for _ in range(200)]
    assert abs(np.mean(vals)) < 0.02


def test_w2_of_pure_shift_equals_shift_norm():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 2))
    shift = np.array([3.0, 4.0])
    assert abs(w2_assignment(a, a + shift, np.random.default_rng(1), n_max=300) - 5.0) < 1e-6


def test_mmd_and_mean_error():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 2))
    b = rng.normal(size=(300, 2))
    bw = median_bandwidth(a)
    assert mmd_rbf(a, b, bw) < mmd_rbf(a, b + 3.0, bw)
    assert abs(mean_error(a, a + 1.0) - np.sqrt(2.0)) < 1e-6


def test_permuted_genes_worsen_metrics():
    rng = np.random.default_rng(0)
    scale = np.array([1.0, 3.0, 0.2])
    a = rng.normal(size=(400, 3)) * scale
    b = rng.normal(size=(400, 3)) * scale
    b_perm = b[:, [2, 0, 1]]
    assert per_gene_w1(a, b_perm, np.random.default_rng(1)) > per_gene_w1(a, b, np.random.default_rng(1))
    assert energy_distance(a, b_perm) > energy_distance(a, b)


def test_protocol_and_noise_floor():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(2000, 4))
    p = EvalProtocol(n_samples=500, seed=0, bandwidth=1.0)
    m = p.evaluate(t + 0.5, t)
    assert set(m) == {"energy_distance", "mmd", "w2", "mean_error", "per_gene_w1", "null_q95"}
    assert m["energy_distance"] > m["null_q95"]
    same = p.evaluate(t[:1000], t[1000:])
    assert same["energy_distance"] <= same["null_q95"]
    floor = noise_floor(t, p, n_rep=3)
    assert floor["energy_distance"] < m["energy_distance"]
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/evaluation/__init__.py
```

```python
# file: cfm_grn_v2/src/cfm_grn_v2/evaluation/metrics.py
"""Distributional metrics computed under a protocol fixed before the test is looked at."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _mean_offdiag(d: np.ndarray) -> float:
    n = d.shape[0]
    if n < 2:
        raise ValueError("need at least two samples")
    return float((d.sum() - np.trace(d)) / (n * (n - 1)))


def energy_distance(a, b) -> float:
    """2 E|X - Y| - E|X - X'| - E|Y - Y'| with unbiased within-sample terms (U-statistic):
    expectation 0 for identical distributions at any sample size (can be slightly negative)."""
    a, b = _f64(a), _f64(b)
    return float(2.0 * cdist(a, b).mean() - _mean_offdiag(cdist(a, a)) - _mean_offdiag(cdist(b, b)))


def median_bandwidth(x, rng: np.random.Generator | None = None, n_max: int = 500) -> float:
    """Median pairwise distance heuristic (computed on TRAINING cells, then frozen)."""
    x = _f64(x)
    rng = np.random.default_rng(0) if rng is None else rng
    if len(x) > n_max:
        x = x[rng.choice(len(x), n_max, replace=False)]
    d = cdist(x, x)[np.triu_indices(len(x), 1)]
    return float(max(np.median(d), 1e-6))


def mmd_rbf(a, b, bandwidth: float) -> float:
    """RBF MMD^2 with unbiased within-sample terms."""
    a, b = _f64(a), _f64(b)
    g = 1.0 / (2.0 * bandwidth ** 2)
    kaa = np.exp(-g * cdist(a, a, "sqeuclidean"))
    kbb = np.exp(-g * cdist(b, b, "sqeuclidean"))
    kab = np.exp(-g * cdist(a, b, "sqeuclidean"))
    return float(_mean_offdiag(kaa) + _mean_offdiag(kbb) - 2.0 * kab.mean())


def w2_assignment(a, b, rng: np.random.Generator, n_max: int = 500) -> float:
    """Exact 2-Wasserstein between balanced subsamples (Hungarian assignment)."""
    a, b = _f64(a), _f64(b)
    n = min(len(a), len(b), n_max)
    ia = rng.choice(len(a), n, replace=False)
    ib = rng.choice(len(b), n, replace=False)
    cost = cdist(a[ia], b[ib], "sqeuclidean")
    r, c = linear_sum_assignment(cost)
    return float(np.sqrt(cost[r, c].mean()))


def mean_error(a, b) -> float:
    return float(np.linalg.norm(_f64(a).mean(0) - _f64(b).mean(0)))


def per_gene_w1(a, b, rng: np.random.Generator, n_max: int = 2000) -> float:
    """Mean over genes of the 1-D Wasserstein distance (sorted-sample form)."""
    a, b = _f64(a), _f64(b)
    n = min(len(a), len(b), n_max)
    a = a[rng.choice(len(a), n, replace=False)]
    b = b[rng.choice(len(b), n, replace=False)]
    return float(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).mean())


def permutation_null(pred, target, rng: np.random.Generator, n_perm: int = 50, q: float = 0.95) -> float:
    """q-quantile of the energy distance between two random splits of the pooled samples: the null
    distribution of ED under 'pred and target share a distribution', at the SAME sample sizes as
    the evaluation. A prediction with ED below this value is indistinguishable from the target."""
    pred, target = _f64(pred), _f64(target)
    pooled = np.concatenate([pred, target])
    n = len(pred)
    vals = [energy_distance(pooled[perm[:n]], pooled[perm[n:]])
            for perm in (rng.permutation(len(pooled)) for _ in range(n_perm))]
    return float(np.quantile(vals, q))


@dataclass
class EvalProtocol:
    n_samples: int = 1000
    seed: int = 0
    bandwidth: float = 1.0
    w2_max: int = 500
    n_perm: int = 50

    def subsample(self, x, rng: np.random.Generator) -> np.ndarray:
        x = np.asarray(x)
        if len(x) <= self.n_samples:
            return x
        return x[rng.choice(len(x), self.n_samples, replace=False)]

    def evaluate(self, pred, target, seed_offset: int = 0) -> dict:
        rng = np.random.default_rng(self.seed + seed_offset)
        p = self.subsample(pred, rng)
        t = self.subsample(target, rng)
        return {
            "energy_distance": energy_distance(p, t),
            "mmd": mmd_rbf(p, t, self.bandwidth),
            "w2": w2_assignment(p, t, rng, self.w2_max),
            "mean_error": mean_error(p, t),
            "per_gene_w1": per_gene_w1(p, t, rng),
            "null_q95": permutation_null(p, t, rng, self.n_perm),
        }


def noise_floor(target, protocol: EvalProtocol, n_rep: int = 5) -> dict:
    """Metrics between two disjoint halves of the target: below this, models are indistinguishable."""
    target = np.asarray(target)
    acc: dict = {}
    for r in range(n_rep):
        rng = np.random.default_rng(protocol.seed + 1000 + r)
        perm = rng.permutation(len(target))
        h = len(target) // 2
        m = protocol.evaluate(target[perm[:h]], target[perm[h:2 * h]], seed_offset=r)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}
```

- [ ] **Step 4: `python -m pytest tests/test_metrics.py -q` → 6 passed**

### Task 10: Simulatore GRN con attività per arco variabile nel tempo

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/data/synthetic_grn.py`
- Test: `cfm_grn_v2/tests/test_synthetic_grn.py`

**Interfaces:**
- Produces: `EdgeSchedule(kind, t_switch, width)`; `TimeVaryingGRNSimulator(n_genes=30, n_regulators=8, density=0.2, frac_dynamic=0.5, frac_repressive=0.3, hill=2.0, decay=0.7, basal=0.3, noise=0.05, dt=0.01, t_max=6.0, seed=0)` con attributi `W [G,G]` (W[j,i] = effetto di j su i), `edges [E,2]`, `schedules`, `gene_names`, `regulators`, `t_max`; metodi `activity(t) -> [E]`, `effective_weights(t) -> [G,G]`, `drift(x, t, knockout=None)`, `simulate_cohort(n_cells, t_end, rng, knockout=None) -> [n, G]`, `snapshot_set(times, n_cells_per_cohort=400, replicates=("r1","r2"), knockouts=(), seed=0) -> (SnapshotSet, truth)` con `truth = {"W", "edges", "times", "activity" [K,E], "schedules", "regulators"}`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_synthetic_grn.py
import numpy as np

from cfm_grn_v2.data.synthetic_grn import TimeVaryingGRNSimulator
from cfm_grn_v2.evaluation.metrics import energy_distance


def test_activity_bounds_and_schedules():
    sim = TimeVaryingGRNSimulator(n_genes=12, n_regulators=4, seed=0)
    a = np.stack([sim.activity(t) for t in np.linspace(0, sim.t_max, 11)])
    assert a.shape == (11, len(sim.edges)) and a.min() >= 0 and a.max() <= 1
    const = [e for e, s in enumerate(sim.schedules) if s.kind == "constant"]
    dyn = [e for e, s in enumerate(sim.schedules) if s.kind != "constant"]
    assert np.allclose(a[:, const], 1.0)
    assert len(dyn) > 0 and (np.abs(a[0, dyn] - a[-1, dyn]) > 0.8).all()


def test_snapshot_set_is_reproducible_finite_nonnegative():
    sim = TimeVaryingGRNSimulator(n_genes=12, n_regulators=4, seed=0)
    s1, truth = sim.snapshot_set([0.0, 2.0, 4.0], n_cells_per_cohort=50, replicates=("r1",), seed=3)
    s2, _ = sim.snapshot_set([0.0, 2.0, 4.0], n_cells_per_cohort=50, replicates=("r1",), seed=3)
    assert np.array_equal(s1.x, s2.x) and np.isfinite(s1.x).all() and (s1.x >= 0).all()
    assert truth["activity"].shape == (3, len(sim.edges)) and truth["W"].shape == (12, 12)
    assert s1.summary()["n_cells"].tolist() == [50, 50, 50]


def test_time_changes_distribution_more_than_noise():
    sim = TimeVaryingGRNSimulator(n_genes=20, n_regulators=6, seed=1)
    s, _ = sim.snapshot_set([0.0, 6.0], n_cells_per_cohort=300, replicates=("r1",), seed=0)
    x0 = s.x[s.mask(times=[0.0])]
    x1 = s.x[s.mask(times=[6.0])]
    assert energy_distance(x0, x1) > 5 * energy_distance(x1[:150], x1[150:])


def test_knockout_gene_stays_silent():
    sim = TimeVaryingGRNSimulator(n_genes=12, n_regulators=4, seed=0)
    s, _ = sim.snapshot_set([0.0, 3.0], n_cells_per_cohort=40, replicates=("r1",), knockouts=(0,), seed=0)
    assert s.conditions() == ["ctrl", "ko_g0"]
    assert np.all(s.x[s.mask(conditions=["ko_g0"])][:, 0] == 0)
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/data/synthetic_grn.py
"""Synthetic gene-regulatory dynamics with KNOWN time-varying edge activity.

Ground truth exists for (i) the signed regulator -> target support W and (ii) the per-edge
activity m_e(t) in [0, 1] as a function of biological_time, so their recovery can be measured
exactly (Phase 3). Cells are sampled destructively: every (time, condition, replicate) cohort
is an independent set of trajectories, hence snapshots are unpaired, like real scRNA-seq.

dx_i = [ clip(basal + sum_j m_ij(t) W_ji h(x_j), 0) - decay x_i ] dt + noise sqrt(x_i) dW
with h(x) = x^n / (K^n + x^n). A knockout clamps x_k = 0 for the whole trajectory.
Switch times lie in [0.25, 0.7] t_max, so the last transitions (validation, test) are autonomous
relaxation after the last switch: forecastable in principle, never seen as a context in training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import SnapshotSet


@dataclass(frozen=True)
class EdgeSchedule:
    kind: str          # "constant" | "switch_on" | "switch_off"
    t_switch: float
    width: float


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class TimeVaryingGRNSimulator:
    def __init__(self, n_genes: int = 30, n_regulators: int = 8, density: float = 0.2, frac_dynamic: float = 0.5,
                 frac_repressive: float = 0.3, hill: float = 2.0, decay: float = 0.7, basal: float = 0.3,
                 noise: float = 0.05, dt: float = 0.01, t_max: float = 6.0, seed: int = 0):
        if n_regulators > n_genes:
            raise ValueError("n_regulators cannot exceed n_genes")
        rng = np.random.default_rng(seed)
        self.G, self.R = int(n_genes), int(n_regulators)
        self.hill, self.decay, self.basal, self.noise = float(hill), float(decay), float(basal), float(noise)
        self.dt, self.t_max, self.K = float(dt), float(t_max), 1.0
        W = np.zeros((self.G, self.G))
        for j in range(self.R):
            candidates = [i for i in range(self.G) if i != j]
            targets = [i for i in candidates if rng.random() < density] or [int(rng.choice(candidates))]
            for i in targets:
                sign = -1.0 if rng.random() < frac_repressive else 1.0
                W[j, i] = sign * rng.uniform(0.8, 2.0)
        self.W = W
        self.edges = np.argwhere(W != 0)          # rows (j, i): regulator j -> target i
        self.schedules = []
        for _ in range(len(self.edges)):
            if rng.random() < frac_dynamic:
                kind = "switch_on" if rng.random() < 0.5 else "switch_off"
                self.schedules.append(EdgeSchedule(kind, float(rng.uniform(0.25, 0.7) * self.t_max), 0.05 * self.t_max))
            else:
                self.schedules.append(EdgeSchedule("constant", 0.0, 1.0))
        self.gene_names = [f"g{i}" for i in range(self.G)]
        self.regulators = list(range(self.R))

    def activity(self, t: float) -> np.ndarray:
        out = np.ones(len(self.edges))
        for e, sch in enumerate(self.schedules):
            if sch.kind == "switch_on":
                out[e] = _sigmoid((t - sch.t_switch) / sch.width)
            elif sch.kind == "switch_off":
                out[e] = 1.0 - _sigmoid((t - sch.t_switch) / sch.width)
        return out

    def effective_weights(self, t: float) -> np.ndarray:
        w = np.zeros_like(self.W)
        j, i = self.edges[:, 0], self.edges[:, 1]
        w[j, i] = self.W[j, i] * self.activity(t)
        return w

    def drift(self, x: np.ndarray, t: float, knockout=None) -> np.ndarray:
        xc = np.clip(x, 0.0, None)
        h = xc ** self.hill / (self.K ** self.hill + xc ** self.hill)
        production = np.clip(self.basal + h @ self.effective_weights(t), 0.0, None)
        dx = production - self.decay * x
        if knockout is not None:
            dx[:, knockout] = 0.0
        return dx

    def simulate_cohort(self, n_cells: int, t_end: float, rng: np.random.Generator, knockout=None) -> np.ndarray:
        x = self.basal / self.decay + 0.1 * np.abs(rng.normal(size=(n_cells, self.G)))
        if knockout is not None:
            x[:, knockout] = 0.0
        t = 0.0
        for _ in range(int(round(t_end / self.dt))):
            dx = self.drift(x, t, knockout)
            x = x + dx * self.dt + self.noise * np.sqrt(self.dt) * np.sqrt(np.clip(x, 0.0, None)) * rng.normal(size=x.shape)
            x = np.clip(x, 0.0, None)
            if knockout is not None:
                x[:, knockout] = 0.0
            t += self.dt
        return x.astype(np.float32)

    def snapshot_set(self, times, n_cells_per_cohort: int = 400, replicates=("r1", "r2"), knockouts=(), seed: int = 0):
        times = [float(t) for t in times]
        if any(t < 0.0 or t > self.t_max for t in times):
            raise ValueError(f"times must lie in [0, {self.t_max}]")
        conditions = [("ctrl", None)] + [(f"ko_{self.gene_names[int(k)]}", int(k)) for k in knockouts]
        xs, ts, cs, rs = [], [], [], []
        for ri, rep in enumerate(replicates):
            for ci, (cname, ko) in enumerate(conditions):
                for ti, t in enumerate(times):
                    rng = np.random.default_rng([seed, ri, ci, ti])
                    x = self.simulate_cohort(n_cells_per_cohort, t, rng, ko)
                    xs.append(x)
                    ts.append(np.full(len(x), t))
                    cs.append(np.array([cname] * len(x)))
                    rs.append(np.array([rep] * len(x)))
        snaps = SnapshotSet(np.concatenate(xs), np.concatenate(ts), np.concatenate(cs), np.concatenate(rs),
                            self.gene_names, name="synthetic_grn")
        truth = {"W": self.W.copy(), "edges": self.edges.copy(), "times": np.array(times),
                 "activity": np.stack([self.activity(t) for t in times]),
                 "schedules": [s.__dict__ for s in self.schedules], "regulators": list(self.regulators)}
        return snaps, truth
```

- [ ] **Step 4: `python -m pytest tests/test_synthetic_grn.py -q` → 4 passed**

### Task 11: Controlli obbligatori

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/evaluation/controls.py`
- Test: `cfm_grn_v2/tests/test_controls.py`

**Interfaces:**
- Produces: `persistence(x0)`; `train_mean(x_train_t1, n)`; `mean_shift(x0, x_train_t0, x_train_t1)`; `ot_pairs(x0, x1, rng, max_size=1000) -> (x0p, x1p)`; `LinearMap(alpha=1.0).fit(x0p, x1p).predict(x0)`; `MLPMap(hidden=128, n_iters=2000, lr=1e-3, seed=0).fit(x0p, x1p).predict(x0)`; `freefm_replay(x0, x0p, x1p, k=1)`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_controls.py
import numpy as np

from cfm_grn_v2.evaluation.controls import (LinearMap, MLPMap, freefm_replay, mean_shift, ot_pairs, persistence,
                                            train_mean)


def _shift_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n, 3))
    x1 = rng.normal(size=(n, 3)) + np.array([2.0, 0.0, -1.0])
    return x0, x1


def test_persistence_and_train_mean():
    x0, x1 = _shift_data()
    assert np.array_equal(persistence(x0), x0)
    m = train_mean(x1, 5)
    assert m.shape == (5, 3) and np.allclose(m[0], x1.mean(0))


def test_mean_shift_moves_by_training_displacement():
    x0, x1 = _shift_data()
    assert np.allclose(mean_shift(x0, x0, x1).mean(0), x1.mean(0), atol=1e-6)


def test_linear_and_mlp_maps_learn_a_shift():
    x0, x1 = _shift_data()
    p0, p1 = ot_pairs(x0, x1, np.random.default_rng(0), max_size=400)
    lin = LinearMap(alpha=1.0).fit(p0, p1)
    mlp = MLPMap(hidden=32, n_iters=1500, lr=1e-2, seed=0).fit(p0, p1)
    xt, _ = _shift_data(seed=5)
    for pred in (lin.predict(xt), mlp.predict(xt)):
        assert np.allclose(pred.mean(0) - xt.mean(0), [2.0, 0.0, -1.0], atol=0.4)


def test_freefm_replay_reuses_observed_displacements():
    x0, x1 = _shift_data()
    p0, p1 = ot_pairs(x0, x1, np.random.default_rng(0), max_size=400)
    assert np.allclose(freefm_replay(p0, p0, p1, k=1), p1)
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/evaluation/controls.py
"""Controls every flow-matching result must be reported against.

persistence     : predict x1 = x0 (no change)
train_mean      : degenerate distribution at the mean of the last observed snapshot
mean_shift      : x0 + (mean of last observed target - mean of its source)
LinearMap       : ridge regression of the OT displacement on the source state
MLPMap          : deterministic MLP regression of the OT displacement (no generative noise)
freefm_replay   : replay of observed transitions (nearest training source's displacement), the
                  "trajectory replay" critique of flow matching turned into a baseline
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from torch import nn

from ..flows.coupling import ot_coupling


def persistence(x0) -> np.ndarray:
    return np.asarray(x0).copy()


def train_mean(x_train_t1, n: int) -> np.ndarray:
    return np.repeat(np.asarray(x_train_t1).mean(0, keepdims=True), n, axis=0)


def mean_shift(x0, x_train_t0, x_train_t1) -> np.ndarray:
    return np.asarray(x0) + (np.asarray(x_train_t1).mean(0) - np.asarray(x_train_t0).mean(0))


def ot_pairs(x0, x1, rng: np.random.Generator, max_size: int = 1000):
    """Balanced OT pairs used to fit deterministic controls (training data only)."""
    x0, x1 = np.asarray(x0), np.asarray(x1)
    i0, i1 = ot_coupling(x0, x1, rng, max_size)
    return x0[i0], x1[i1]


class LinearMap:
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, x0p, x1p) -> "LinearMap":
        self.model.fit(np.asarray(x0p), np.asarray(x1p) - np.asarray(x0p))
        return self

    def predict(self, x0) -> np.ndarray:
        x0 = np.asarray(x0)
        return x0 + self.model.predict(x0)


class MLPMap:
    def __init__(self, hidden: int = 128, n_iters: int = 2000, lr: float = 1e-3, seed: int = 0):
        self.hidden, self.n_iters, self.lr, self.seed = int(hidden), int(n_iters), float(lr), int(seed)
        self.net = None

    def fit(self, x0p, x1p) -> "MLPMap":
        torch.manual_seed(self.seed)
        x = torch.as_tensor(np.asarray(x0p), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(x1p) - np.asarray(x0p), dtype=torch.float32)
        d = x.shape[1]
        self.net = nn.Sequential(nn.Linear(d, self.hidden), nn.SiLU(), nn.Linear(self.hidden, self.hidden), nn.SiLU(),
                                 nn.Linear(self.hidden, d))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed)
        b = min(256, len(x))
        for _ in range(self.n_iters):
            idx = torch.randint(0, len(x), (b,), generator=gen)
            loss = ((self.net(x[idx]) - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def predict(self, x0) -> np.ndarray:
        x0 = np.asarray(x0)
        with torch.no_grad():
            return x0 + self.net(torch.as_tensor(x0, dtype=torch.float32)).numpy()


def freefm_replay(x0, x0p, x1p, k: int = 1) -> np.ndarray:
    """Move every cell by the mean displacement of its k nearest TRAINING sources."""
    x0p, x1p = np.asarray(x0p), np.asarray(x1p)
    _, idx = NearestNeighbors(n_neighbors=k).fit(x0p).kneighbors(np.asarray(x0))
    return np.asarray(x0) + (x1p - x0p)[idx].mean(1)
```

- [ ] **Step 4: `python -m pytest tests/test_controls.py -q` → 4 passed**

### Task 12: Configurazione e griglie

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/training/__init__.py`, `cfm_grn_v2/src/cfm_grn_v2/training/config.py`
- Test: `cfm_grn_v2/tests/test_config.py`

**Interfaces:**
- Produces: `TrainConfig` (frozen dataclass: `hidden=256, depth=3, lr=1e-3, weight_decay=0.0, batch_size=256, n_iters=2000, sigma_min=1e-3, coupling="ot", n_ode_steps=20, ode_method="rk4", time_embed_dim=32, cond_dim=8, time_conditioning="delta", eval_every=200, patience=5, seed=0, shuffle_biological_time=False, ema_decay=0.999, grad_clip=1.0, pair_mode="consecutive", clock="flow", path="linear", chain="ot"`) con `.to_dict()`, `.from_dict(d)` (chiavi ignote → `TypeError`), `.from_json(path)`, `.replace(**kw)`; `expand_grid(grid: dict[str, list]) -> list[dict]`; `random_subset(configs, n, seed=0) -> list[dict]`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_config.py
import json

import pytest

from cfm_grn_v2.training.config import TrainConfig, expand_grid, random_subset


def test_config_roundtrip_and_replace(tmp_path):
    cfg = TrainConfig(hidden=64, n_iters=10)
    d = cfg.to_dict()
    assert TrainConfig.from_dict(d) == cfg
    assert cfg.replace(lr=5e-4).lr == 5e-4 and cfg.lr == 1e-3
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    assert TrainConfig.from_json(p) == cfg


def test_unknown_key_raises():
    with pytest.raises(TypeError):
        TrainConfig.from_dict({"hidden": 64, "not_a_field": 1})


def test_v013_training_fields_have_defaults():
    cfg = TrainConfig()
    assert cfg.ema_decay == 0.999 and cfg.grad_clip == 1.0 and cfg.pair_mode == "consecutive" and cfg.clock == "flow" and cfg.path == "linear" and cfg.chain == "ot"
    assert TrainConfig.from_dict({"pair_mode": "all", "ema_decay": 0.0}).pair_mode == "all"


def test_expand_grid_and_subset():
    grid = expand_grid({"hidden": [64, 128], "lr": [1e-3, 1e-4], "coupling": ["ot"]})
    assert len(grid) == 4 and all(g["coupling"] == "ot" for g in grid)
    sub = random_subset(grid, 2, seed=0)
    assert len(sub) == 2 and all(s in grid for s in sub)
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/training/__init__.py
```

```python
# file: cfm_grn_v2/src/cfm_grn_v2/training/config.py
"""Versioned training configuration and hyper-parameter grids."""
from __future__ import annotations

import dataclasses
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 256
    depth: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    n_iters: int = 2000
    sigma_min: float = 1e-3
    coupling: str = "ot"              # "ot" | "independent"
    n_ode_steps: int = 20
    ode_method: str = "rk4"
    time_embed_dim: int = 32
    cond_dim: int = 8
    time_conditioning: str = "delta"      # "delta" | "absolute" | "none" (see ContextEncoder)
    eval_every: int = 200
    patience: int = 5
    seed: int = 0
    shuffle_biological_time: bool = False   # control: destroys temporal order in the training labels
    ema_decay: float = 0.999          # exponential moving average of the weights, used for validation and prediction; 0 = off
    grad_clip: float = 1.0            # max gradient norm per step; 0 = off
    pair_mode: str = "consecutive"    # "consecutive": (t_k, t_k+1) transitions | "all": every ordered pair (t_i, t_j), i < j
    clock: str = "flow"               # "flow": v(x, s, ctx) over s in [0, 1] | "duration": autonomous v(x, cond) over the biological horizon
    path: str = "linear"              # "linear": straight pair interpolants | "spline": cubic spline through OT-chained trajectories (needs clock="duration")
    chain: str = "ot"                 # spline path only: "ot" (chained optimal transport) | "random" (random pairing, ablation)
    spline_kind: str = "natural"      # spline boundary condition: "natural" | "not-a-knot" | "clamped"
    field_time: str = "none"          # duration clock: "none" = autonomous v(x, cond) | "bio" = v(x, t/time_scale, cond), t biological (v0.1.11)
    path_center: str = "spline"       # spline path centre: "spline" (cubic through the knot means) | "linear" (straight segments between them, v0.1.10)
    marginal_iters: int = 0           # v0.1.9 distributional fine-tune: marginal-matching steps after the path stage (0 = off)
    marginal_weight: float = 1.0      # weight of the two-sample divergence between the pushforward and the next snapshot
    marginal_prior: float = 1.0       # weight of the retained path loss during the fine-tune (0 = pure marginal matching)
    marginal_batch: int = 256         # cells per side of the divergence
    marginal_steps: int = 8           # ODE steps per unit of biological time inside the differentiable rollout
    marginal_metric: str = "energy"   # "energy" (U-statistic energy distance) | "sliced_w2"
    marginal_window: float = 0.0      # during the fine-tune, supervise the path only within this many time units of an observed time (0 = everywhere)
    marginal_lr: float = 3e-4
    marginal_eval_every: int = 25
    marginal_patience: int = 8

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise TypeError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def replace(self, **kw) -> "TrainConfig":
        return dataclasses.replace(self, **kw)


def expand_grid(grid: dict) -> list:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*[grid[k] for k in keys])]


def random_subset(configs: list, n: int, seed: int = 0) -> list:
    if n >= len(configs):
        return list(configs)
    idx = np.random.default_rng(seed).choice(len(configs), n, replace=False)
    return [configs[i] for i in sorted(idx)]
```

- [ ] **Step 4: `python -m pytest tests/test_config.py -q` → 3 passed**

### Task 13: Trainer, task di forecast, checkpoint, controllo a tempo rimescolato

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/training/trainer.py`
- Test: `cfm_grn_v2/tests/test_trainer.py`, `cfm_grn_v2/tests/test_leakage.py`

**Interfaces:**
- Consumes: `SnapshotSet`, `fit_on_train`, `Standardizer`, `ot_coupling`, `independent_coupling`, `cfm_loss`, `sample_flow_time`, `integrate`, `ConditionRegistry`, `ContextEncoder`, `MLPVectorField`, `EvalProtocol`, `energy_distance`, `TrainConfig`.
- Produces: `Transition(condition, replicate, bio_t0, bio_t1, x0, x1)`; `ForecastTask(source_mask, target_mask, bio_t0, bio_t1, name)`; `time_task(snapshots, t0, t1, replicates=None)`; `replicate_task(snapshots, replicate, t0, t1)`; `build_transitions(snapshots, times, x=None, min_cells=2, pair_mode="consecutive") -> list[Transition]`; `WeightEMA(modules: dict[str, nn.Module], decay)` con `.update()` e `.state() -> dict[name, state_dict]`; `build_networks(config, registry, n_genes, time_scale) -> (ContextEncoder, MLPVectorField)` (con `clock="duration"`: contesto `none`, campo `use_time=False`; `path="spline"` richiede `clock="duration"`); `spline_paths(fit, fit_times, x_fit, rng, max_cells=1000, chain="ot") -> list[(condition, SplinePath)]`; `task_groups(snapshots, task, min_cells=2) -> list[(cond, rep, m0, m1)]`; `FlowModel` con `.predict(x0_raw, condition, bio_t0, bio_t1)`, `.predict_standardized(...)`, `.n_params`, `.best_iter`, `.best_val`, `.train_seconds`, `.save(path)`, `FlowModel.load(path, expected_gene_names=None)`; `train_flow(snapshots, fit_mask, fit_times, config, val_task=None, protocol=None, verbose=False) -> (FlowModel, history)`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_trainer.py
import numpy as np
import pytest
import torch

from cfm_grn_v2.data.schema import SnapshotSet
from cfm_grn_v2.data.synthetic_toys import make_toy
from cfm_grn_v2.evaluation.metrics import EvalProtocol, energy_distance
from cfm_grn_v2.models.conditioning import UnknownConditionError
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.trainer import FlowModel, WeightEMA, build_transitions, replicate_task, train_flow


def _setup():
    s = make_toy("translation", n_per_group=400, seed=0)
    fit = s.mask(replicates=["r1"])
    val = replicate_task(s, "r2", 0.0, 1.0)
    test = replicate_task(s, "r3", 0.0, 1.0)
    return s, fit, val, test


def test_build_transitions_consecutive_pairs():
    s, fit, _, _ = _setup()
    tr = build_transitions(s.subset(fit), [0.0, 1.0])
    assert len(tr) == 1 and tr[0].bio_t0 == 0.0 and tr[0].bio_t1 == 1.0 and tr[0].condition == "ctrl"
    assert tr[0].x0.shape == (400, 2) and tr[0].x1.shape == (400, 2)


def test_build_transitions_all_pairs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2)).astype(np.float32)
    t = np.repeat([0.0, 1.0, 2.0, 3.0], 10)
    s = SnapshotSet(x, t, np.array(["ctrl"] * 40), np.array(["r1"] * 40), ["a", "b"])
    cons = build_transitions(s, [0.0, 1.0, 2.0, 3.0])
    allp = build_transitions(s, [0.0, 1.0, 2.0, 3.0], pair_mode="all")
    assert [(tr.bio_t0, tr.bio_t1) for tr in cons] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    assert sorted((tr.bio_t0, tr.bio_t1) for tr in allp) == [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (1.0, 2.0), (1.0, 3.0), (2.0, 3.0)]
    assert all(tr.bio_t0 < tr.bio_t1 and len(tr.x0) == 10 for tr in allp)
    with pytest.raises(ValueError):
        build_transitions(s, [0.0, 1.0], pair_mode="every")


def test_weight_ema_matches_manual_average():
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 2)
    ema = WeightEMA({"lin": lin}, decay=0.5)
    w0 = lin.weight.detach().clone()
    with torch.no_grad():
        lin.weight.fill_(1.0)
    ema.update()                                   # n=0: d = min(0.5, 1/10) = 0.1
    expected = 0.1 * w0 + 0.9 * torch.ones_like(w0)
    assert torch.allclose(ema.state()["lin"]["weight"], expected)
    with torch.no_grad():
        lin.weight.fill_(3.0)
    ema.update()                                   # n=1: d = min(0.5, 2/11)
    d = 2.0 / 11.0
    expected = d * expected + (1.0 - d) * 3.0 * torch.ones_like(w0)
    assert torch.allclose(ema.state()["lin"]["weight"], expected)
    assert not torch.allclose(lin.weight, ema.state()["lin"]["weight"])   # the shadow never aliases the module
    with pytest.raises(ValueError):
        WeightEMA({"lin": lin}, decay=1.0)


def test_train_flow_ema_and_grad_clip_options():
    s, fit, val, _ = _setup()
    for ema in (0.0, 0.9):
        cfg = TrainConfig(hidden=16, depth=1, n_iters=30, batch_size=64, eval_every=10, patience=10, seed=0,
                          ema_decay=ema, grad_clip=0.5)
        model, hist = train_flow(s, fit, [0.0, 1.0], cfg, val_task=val, protocol=EvalProtocol(n_samples=100, n_perm=5))
        pred = model.predict(s.x[val.source_mask][:20], "ctrl", 0.0, 1.0)
        assert np.isfinite(pred).all() and len(hist) == 3 and model.best_iter in (10, 20, 30)


def test_duration_clock_trains_an_autonomous_field(tmp_path):
    s, fit, val, test = _setup()
    cfg = TrainConfig(hidden=32, depth=2, lr=3e-3, n_iters=200, batch_size=128, eval_every=100, patience=5, seed=0,
                      clock="duration", ema_decay=0.0)
    model, hist = train_flow(s, fit, [0.0, 1.0], cfg, val_task=val, protocol=EvalProtocol(n_samples=200, n_perm=5))
    assert model.vector_field.use_time is False and model.context_encoder.time_conditioning == "none"
    x0, x1 = s.x[test.source_mask], s.x[test.target_mask]
    pred1 = model.predict(x0, "ctrl", 0.0, 1.0)
    assert energy_distance(pred1, x1) < energy_distance(x0, x1)
    pred2 = model.predict(x0, "ctrl", 0.0, 2.0)                    # longer horizon: integrates further
    assert np.isfinite(pred2).all() and not np.allclose(pred1, pred2)
    assert np.allclose(model.predict(x0[:10], "ctrl", 3.0, 4.0), pred1[:10], atol=1e-5)   # only the horizon matters
    with pytest.raises(ValueError):
        model.predict(x0[:5], "ctrl", 1.0, 1.0)
    p = tmp_path / "m.pt"
    model.save(p)
    assert np.allclose(FlowModel.load(p).predict(x0[:10], "ctrl", 0.0, 1.0), pred1[:10], atol=1e-5)
    with pytest.raises(ValueError):
        train_flow(s, fit, [0.0, 1.0], cfg.replace(clock="sundial"))


def test_spline_path_learns_curvature_between_snapshots():
    rng = np.random.default_rng(0)
    n, times = 150, [0.0, 1.0, 2.0, 3.0]
    base = rng.normal(size=(n, 2)) * 0.05
    xs, ts = [], []
    for t in times:
        xs.append(base + np.array([t, 0.5 * t * t]) + rng.normal(size=(n, 2)) * 0.02)
        ts.append(np.full(n, t))
    s = SnapshotSet(np.concatenate(xs).astype(np.float32), np.concatenate(ts), np.array(["ctrl"] * (4 * n)),
                    np.array(["r1"] * (4 * n)), ["a", "b"])
    cfg = TrainConfig(hidden=32, depth=2, lr=3e-3, n_iters=300, batch_size=128, eval_every=100, patience=5, seed=0,
                      clock="duration", path="spline", ema_decay=0.0)
    model, _ = train_flow(s, s.mask(times=[0.0, 1.0, 3.0]), [0.0, 1.0, 3.0], cfg)
    x1, x2 = s.x[s.mask(times=[1.0])], s.x[s.mask(times=[2.0])]
    pred = model.predict(x1, "ctrl", 1.0, 2.0)
    assert energy_distance(pred, x2) < energy_distance(x1, x2)
    with pytest.raises(ValueError):
        train_flow(s, s.mask(times=[0.0, 1.0]), [0.0, 1.0], cfg.replace(clock="flow"))
    with pytest.raises(ValueError):
        train_flow(s, s.mask(times=[0.0, 1.0]), [0.0, 1.0], cfg.replace(path="bezier"))


def test_train_flow_beats_persistence_on_translation(tmp_path):
    s, fit, val, test = _setup()
    cfg = TrainConfig(hidden=64, depth=2, lr=3e-3, n_iters=400, batch_size=128, eval_every=100, patience=3, seed=0)
    model, hist = train_flow(s, fit, [0.0, 1.0], cfg, val_task=val, protocol=EvalProtocol(n_samples=300))
    assert "val_energy_distance" in hist[0] and model.best_iter is not None
    x0 = s.x[test.source_mask]
    x1 = s.x[test.target_mask]
    pred = model.predict(x0, "ctrl", 0.0, 1.0)
    assert energy_distance(pred, x1) < energy_distance(x0, x1)
    path = tmp_path / "m.pt"
    model.save(path)
    loaded = FlowModel.load(path)
    assert np.allclose(loaded.predict(x0[:10], "ctrl", 0.0, 1.0), pred[:10], atol=1e-5)
    with pytest.raises(UnknownConditionError):
        model.predict(x0[:5], "never_seen", 0.0, 1.0)


def test_checkpoint_gene_mismatch_raises(tmp_path):
    s, fit, val, test = _setup()
    cfg = TrainConfig(hidden=16, depth=1, n_iters=20, batch_size=64, eval_every=10, seed=0)
    model, _ = train_flow(s, fit, [0.0, 1.0], cfg)
    p = tmp_path / "m.pt"
    model.save(p)
    with pytest.raises(ValueError):
        FlowModel.load(p, expected_gene_names=["x", "y"])


def test_validation_cells_in_fit_mask_raise():
    s, fit, val, test = _setup()
    cfg = TrainConfig(hidden=16, depth=1, n_iters=5, batch_size=32, eval_every=5)
    with pytest.raises(ValueError):
        train_flow(s, fit | val.target_mask, [0.0, 1.0], cfg, val_task=val)


def test_field_time_bio_learns_a_reversal_the_autonomous_field_cannot():
    """field_time="bio": the duration-clock field sees the biological time. Cells go out and come back
    (x = 0 -> 1 -> 0 at t = 0, 1, 2). At the turning point the path velocity is zero and afterwards it points
    back, so an autonomous field v(x) started from the cells at t = 1 has no way to know it must return; a
    field v(x, t) does. The model is also non-autonomous (the start time matters) and survives save/load."""
    rng = np.random.default_rng(0)
    n, times, pos = 150, [0.0, 1.0, 2.0], [0.0, 1.0, 0.0]
    all_t, all_p = times + [0.5], pos + [0.5]                  # t = 0.5 is held out (the fit mask must hide cells)
    xs = [np.column_stack([np.full(n, p), np.zeros(n)]) + rng.normal(size=(n, 2)) * 0.05 for p in all_p]
    s = SnapshotSet(np.concatenate(xs).astype(np.float32), np.repeat(all_t, n), np.array(["ctrl"] * (4 * n)),
                    np.array(["r1"] * (4 * n)), ["a", "b"])
    cfg = TrainConfig(hidden=32, depth=2, lr=3e-3, n_iters=600, batch_size=128, eval_every=200, patience=5, seed=0,
                      clock="duration", path="spline", ema_decay=0.0)
    x1, x2 = s.x[s.mask(times=[1.0])], s.x[s.mask(times=[2.0])]
    auto, _ = train_flow(s, s.mask(times=times), times, cfg)
    bio, _ = train_flow(s, s.mask(times=times), times, cfg.replace(field_time="bio"))
    e_auto = energy_distance(auto.predict(x1, "ctrl", 1.0, 2.0), x2)
    e_bio = energy_distance(bio.predict(x1, "ctrl", 1.0, 2.0), x2)
    assert e_bio < 0.5 * e_auto and e_bio < energy_distance(x1, x2)
    z = np.zeros((4, 2), dtype=np.float32)
    assert not np.allclose(bio.predict_standardized(z, "ctrl", 0.0, 1.0), bio.predict_standardized(z, "ctrl", 1.0, 2.0))
    assert np.allclose(auto.predict_standardized(z, "ctrl", 0.0, 1.0), auto.predict_standardized(z, "ctrl", 1.0, 2.0),
                       atol=1e-6)


def test_field_time_bio_save_load_and_validation(tmp_path):
    rng = np.random.default_rng(1)
    n, times = 60, [0.0, 1.0, 2.0]
    xs = [rng.normal(size=(n, 2)) * 0.1 + np.array([t, 0.0]) for t in times]
    s = SnapshotSet(np.concatenate(xs).astype(np.float32), np.repeat(times, n), np.array(["ctrl"] * (3 * n)),
                    np.array(["r1"] * (3 * n)), ["a", "b"])
    cfg = TrainConfig(hidden=16, depth=2, n_iters=40, batch_size=32, eval_every=20, seed=0, clock="duration",
                      path="spline", field_time="bio", marginal_iters=4, marginal_eval_every=2, marginal_batch=16,
                      marginal_steps=2)
    model, _ = train_flow(s, s.mask(times=[0.0, 2.0]), [0.0, 2.0], cfg, replicate_task(s, "r1", 0.0, 1.0),
                          EvalProtocol(n_samples=40))
    p = tmp_path / "bio.pt"
    model.save(p)
    z = s.x[:5]
    assert np.allclose(FlowModel.load(p).predict(z, "ctrl", 0.5, 1.5), model.predict(z, "ctrl", 0.5, 1.5), atol=1e-5)
    with pytest.raises(ValueError):
        train_flow(s, s.mask(times=[0.0, 1.0]), [0.0, 1.0], TrainConfig(n_iters=2, field_time="bio"))
    with pytest.raises(ValueError):
        train_flow(s, s.mask(times=[0.0, 1.0]), [0.0, 1.0], cfg.replace(field_time="clockwork"))
```

```python
# file: cfm_grn_v2/tests/test_leakage.py
import numpy as np
import torch

from cfm_grn_v2.data.schema import SnapshotSet
from cfm_grn_v2.data.synthetic_toys import make_toy
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.trainer import replicate_task, train_flow


def _params(model):
    return torch.cat([p.detach().flatten() for p in model.vector_field.parameters()]).numpy()


def test_test_cells_do_not_influence_training():
    s = make_toy("translation", n_per_group=200, seed=0)
    fit = s.mask(replicates=["r1"])
    val = replicate_task(s, "r2", 0.0, 1.0)
    cfg = TrainConfig(hidden=16, depth=1, n_iters=30, batch_size=64, eval_every=10, seed=0)
    m1, _ = train_flow(s, fit, [0.0, 1.0], cfg, val_task=val)
    x2 = s.x.copy()
    x2[s.mask(replicates=["r3"])] += 50.0
    s2 = SnapshotSet(x2, s.biological_time, s.condition, s.replicate, s.gene_names, name=s.name)
    m2, _ = train_flow(s2, fit, [0.0, 1.0], cfg, val_task=val)
    assert np.allclose(_params(m1), _params(m2))
    assert np.allclose(m1.standardizer.mean_, m2.standardizer.mean_)


def test_shuffled_time_control_trains():
    s = make_toy("translation", n_per_group=200, seed=0)
    fit = s.mask(replicates=["r1"])
    cfg = TrainConfig(hidden=16, depth=1, n_iters=20, batch_size=64, eval_every=10, seed=0, shuffle_biological_time=True)
    model, hist = train_flow(s, fit, [0.0, 1.0], cfg)
    assert model.config.shuffle_biological_time and len(hist) >= 1
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/training/trainer.py
"""Training loop of the conditional flow matching engine (no graph, no mask).

``train_flow`` fits on the cells in ``fit_mask`` over consecutive ``fit_times``; an optional
``val_task`` drives early stopping on the validation energy distance. Validation cells never
touch the standardizer or the loss. ``FlowModel.predict`` maps observed cells at bio_t0 to a
population at bio_t1 by integrating the learned field over flow_time in [0, 1] (clock="flow") or,
with clock="duration", by integrating an autonomous field v(x, condition) trained on the velocity
(x1 - x0)/dt over the biological horizon bio_t1 - bio_t0 (dt-independent, like a neural ODE).
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .. import __version__
from ..data.preprocessing import Standardizer, fit_on_train
from ..data.schema import SnapshotSet
from ..evaluation.metrics import EvalProtocol, energy_distance
from ..flows.coupling import chain_ot, independent_coupling, ot_coupling
from ..flows.marginal import marginal_loss
from ..flows.path import SplinePath, cfm_loss, sample_flow_time, spline_loss
from ..flows.solver import integrate
from ..models.conditioning import ConditionRegistry, ContextEncoder
from ..models.vector_field import MLPVectorField
from .config import TrainConfig


@dataclass
class Transition:
    condition: str
    replicate: str
    bio_t0: float
    bio_t1: float
    x0: np.ndarray
    x1: np.ndarray


@dataclass
class ForecastTask:
    """Predict the cells in target_mask (at bio_t1) from the cells in source_mask (at bio_t0)."""
    source_mask: np.ndarray
    target_mask: np.ndarray
    bio_t0: float
    bio_t1: float
    name: str = "task"


def time_task(snapshots: SnapshotSet, t0: float, t1: float, replicates=None) -> ForecastTask:
    return ForecastTask(snapshots.mask(times=[t0], replicates=replicates),
                        snapshots.mask(times=[t1], replicates=replicates), float(t0), float(t1), f"t{t0}->t{t1}")


def replicate_task(snapshots: SnapshotSet, replicate: str, t0: float, t1: float) -> ForecastTask:
    return ForecastTask(snapshots.mask(times=[t0], replicates=[replicate]),
                        snapshots.mask(times=[t1], replicates=[replicate]), float(t0), float(t1),
                        f"{replicate}:t{t0}->t{t1}")


def build_transitions(snapshots: SnapshotSet, times, x=None, min_cells: int = 2, pair_mode: str = "consecutive") -> list:
    """Transitions within each (condition, replicate) group: consecutive (t_k -> t_k+1) pairs, or every
    ordered pair (t_i -> t_j, i < j) with pair_mode="all" (the delta-t context then carries the horizon)."""
    x = snapshots.x if x is None else np.asarray(x)
    if len(x) != snapshots.n_cells:
        raise ValueError("x must be aligned with snapshots (one row per cell)")
    times = np.unique(np.asarray(list(times), dtype=np.float64))
    if pair_mode == "consecutive":
        pairs = list(zip(times[:-1], times[1:]))
    elif pair_mode == "all":
        pairs = [(t0, t1) for i, t0 in enumerate(times) for t1 in times[i + 1:]]
    else:
        raise ValueError(f"unknown pair_mode {pair_mode!r} (expected 'consecutive' or 'all')")
    out = []
    for cond in snapshots.conditions():
        for rep in snapshots.replicates():
            for t0, t1 in pairs:
                m0 = snapshots.mask(times=[t0], conditions=[cond], replicates=[rep])
                m1 = snapshots.mask(times=[t1], conditions=[cond], replicates=[rep])
                if m0.sum() >= min_cells and m1.sum() >= min_cells:
                    out.append(Transition(cond, rep, float(t0), float(t1), x[m0], x[m1]))
    if not out:
        raise ValueError("no transition has enough cells on both sides; check times, conditions and replicates")
    return out


def task_groups(snapshots: SnapshotSet, task: ForecastTask, min_cells: int = 2) -> list:
    """(condition, replicate, source_mask, target_mask) for every group present on both sides of the task."""
    out = []
    for cond in snapshots.conditions():
        for rep in snapshots.replicates():
            g = snapshots.mask(conditions=[cond], replicates=[rep])
            m0, m1 = task.source_mask & g, task.target_mask & g
            if m0.sum() >= min_cells and m1.sum() >= min_cells:
                out.append((cond, rep, m0, m1))
    return out


class WeightEMA:
    """Exponential moving average of module weights, with the usual warm-up d_n = min(decay, (1+n)/(10+n))
    so the average is not dominated by the initial weights. Used for validation and for the final model."""

    def __init__(self, modules: dict, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        self.modules, self.decay, self.n_updates = dict(modules), float(decay), 0
        self.shadow = {name: {k: v.detach().clone() for k, v in m.state_dict().items()}
                       for name, m in self.modules.items()}

    def update(self) -> None:
        d = min(self.decay, (1.0 + self.n_updates) / (10.0 + self.n_updates))
        with torch.no_grad():
            for name, m in self.modules.items():
                for k, v in m.state_dict().items():
                    if v.dtype.is_floating_point:
                        self.shadow[name][k].mul_(d).add_(v, alpha=1.0 - d)
                    else:
                        self.shadow[name][k].copy_(v)
        self.n_updates += 1

    def state(self) -> dict:
        return copy.deepcopy(self.shadow)


def _context_mode(config: TrainConfig) -> str:
    """clock="duration" trains an autonomous field: the context carries the condition only."""
    if config.clock == "flow":
        return config.time_conditioning
    if config.clock == "duration":
        return "none"
    raise ValueError(f"unknown clock {config.clock!r} (expected 'flow' or 'duration')")


def _bio_time(config: TrainConfig) -> bool:
    """field_time="bio": the duration-clock field receives the biological time (scaled by time_scale)."""
    if config.field_time not in ("none", "bio"):
        raise ValueError(f"unknown field_time {config.field_time!r} (expected 'none' or 'bio')")
    if config.field_time == "bio" and config.clock != "duration":
        raise ValueError("field_time='bio' is defined for the duration clock only")
    return config.field_time == "bio"


def build_networks(config: TrainConfig, registry: ConditionRegistry, n_genes: int, time_scale: float):
    if config.path not in ("linear", "spline"):
        raise ValueError(f"unknown path {config.path!r} (expected 'linear' or 'spline')")
    if config.path == "spline" and config.clock != "duration":
        raise ValueError("path='spline' regresses biological velocities: it needs clock='duration'")
    bio = _bio_time(config)
    ctx = ContextEncoder(registry, config.cond_dim, time_scale, _context_mode(config))
    vf = MLPVectorField(n_genes, ctx.out_dim, config.hidden, config.depth, config.time_embed_dim,
                        use_time=config.clock == "flow" or bio, time_embed="scalar" if bio else "sinusoidal")
    return ctx, vf


def spline_paths(fit: SnapshotSet, fit_times, x_fit, rng: np.random.Generator, max_cells: int = 1000,
                 chain: str = "ot", bc_type: str = "natural", center: str = "spline") -> list:
    """One (condition, SplinePath) per (condition, replicate) group, through OT-chained pseudo-trajectories
    over every fit timepoint with at least two cells in that group."""
    times = np.unique(np.asarray(list(fit_times), dtype=np.float64))
    out = []
    for cond in fit.conditions():
        for rep in fit.replicates():
            keep = [(t, m) for t in times if (m := fit.mask(times=[t], conditions=[cond], replicates=[rep])).sum() >= 2]
            if len(keep) < 2:
                continue
            traj = chain_ot([x_fit[m] for _, m in keep], rng, max_cells, mode=chain)
            out.append((cond, SplinePath([t for t, _ in keep], traj, bc_type=bc_type, center=center)))
    if not out:
        raise ValueError("no (condition, replicate) group has two timepoints with enough cells")
    return out


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class FlowModel:
    def __init__(self, registry, standardizer, context_encoder, vector_field, config, gene_names, time_scale):
        self.registry = registry
        self.standardizer = standardizer
        self.context_encoder = context_encoder
        self.vector_field = vector_field
        self.config = config
        self.gene_names = list(gene_names)
        self.time_scale = float(time_scale)
        self.best_iter = None
        self.best_val = None
        self.marginal_iter = None
        self.train_seconds = None

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in list(self.context_encoder.parameters()) + list(self.vector_field.parameters()))

    def field(self, bio_t0: float = 0.0):
        """What the ODE solver integrates. The solver's clock s starts at 0; with field_time="bio" it is turned
        into the scaled biological time (bio_t0 + s) / time_scale the field was trained on."""
        if self.config.clock == "duration" and self.config.field_time == "bio":
            vf, t0, sc = self.vector_field, float(bio_t0), self.time_scale
            return lambda x, s, ctx: vf(x, (t0 + s) / sc, ctx)
        return self.vector_field

    def path_time(self, u) -> torch.Tensor | None:
        """Scaled biological times of path samples for a non-autonomous field (None for the autonomous one)."""
        if self.config.clock == "duration" and self.config.field_time == "bio":
            return torch.as_tensor(np.asarray(u) / self.time_scale, dtype=torch.float32)
        return None

    def _context(self, condition: str, bio_t0: float, bio_t1: float, b: int) -> torch.Tensor:
        idx = self.registry.encode([condition] * b)
        return self.context_encoder(idx, torch.full((b,), float(bio_t0)), torch.full((b,), float(bio_t1)))

    def predict_standardized(self, x0, condition: str, bio_t0: float, bio_t1: float, n_steps=None, method=None) -> np.ndarray:
        self.vector_field.eval()
        self.context_encoder.eval()
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(x0), dtype=torch.float32)
            ctx = self._context(condition, bio_t0, bio_t1, x.shape[0])
            steps = int(n_steps or self.config.n_ode_steps)
            horizon = 1.0
            if self.config.clock == "duration":
                horizon = float(bio_t1) - float(bio_t0)
                if horizon <= 0:
                    raise ValueError("bio_t1 must exceed bio_t0 for the duration clock")
                steps = max(steps, int(math.ceil(steps * horizon)))     # keep the step size of a unit horizon
            out = integrate(self.field(bio_t0), x, ctx, steps, method or self.config.ode_method, t_end=horizon)
        self.vector_field.train()
        self.context_encoder.train()
        return out.numpy()

    def predict(self, x0_raw, condition: str, bio_t0: float, bio_t1: float, **kw) -> np.ndarray:
        z = self.standardizer.transform(x0_raw)
        return self.standardizer.inverse_transform(self.predict_standardized(z, condition, bio_t0, bio_t1, **kw))

    def snapshot_state(self) -> dict:
        return {"ctx": copy.deepcopy(self.context_encoder.state_dict()),
                "vf": copy.deepcopy(self.vector_field.state_dict())}

    def load_snapshot(self, state: dict) -> None:
        self.context_encoder.load_state_dict(state["ctx"])
        self.vector_field.load_state_dict(state["vf"])

    def save(self, path) -> None:
        torch.save({"version": __version__, "config": self.config.to_dict(), "gene_names": self.gene_names,
                    "standardizer": self.standardizer.to_dict(), "registry": self.registry.to_dict(),
                    "time_scale": self.time_scale, "state": self.snapshot_state(),
                    "best_iter": self.best_iter, "best_val": self.best_val}, path)

    @classmethod
    def load(cls, path, expected_gene_names=None) -> "FlowModel":
        ck = torch.load(path, weights_only=False)
        if ck.get("version") != __version__:
            raise ValueError(f"checkpoint version {ck.get('version')!r} differs from code version {__version__!r}")
        if expected_gene_names is not None and list(expected_gene_names) != list(ck["gene_names"]):
            raise ValueError("checkpoint gene names differ from the data gene names")
        config = TrainConfig.from_dict(ck["config"])
        registry = ConditionRegistry.from_dict(ck["registry"])
        ctx, vf = build_networks(config, registry, len(ck["gene_names"]), ck["time_scale"])
        model = cls(registry, Standardizer.from_dict(ck["standardizer"]), ctx, vf, config, ck["gene_names"],
                    ck["time_scale"])
        model.load_snapshot(ck["state"])
        model.best_iter, model.best_val = ck.get("best_iter"), ck.get("best_val")
        return model


def _validate(model: FlowModel, snapshots: SnapshotSet, x_std: np.ndarray, task: ForecastTask,
              protocol: EvalProtocol) -> float:
    vals = []
    for cond, rep, m0, m1 in task_groups(snapshots, task):
        if cond not in model.registry.names:
            continue
        rng = np.random.default_rng(protocol.seed)
        x0 = protocol.subsample(x_std[m0], rng)
        x1 = protocol.subsample(x_std[m1], rng)
        vals.append(energy_distance(model.predict_standardized(x0, cond, task.bio_t0, task.bio_t1), x1))
    if not vals:
        raise ValueError("validation task shares no (condition, replicate) group with the training conditions")
    return float(np.mean(vals))


def _finetune_marginal(model: FlowModel, snapshots: SnapshotSet, x_std: np.ndarray, transitions: list, paths,
                       config: TrainConfig, val_task: ForecastTask | None, protocol: EvalProtocol,
                       rng: np.random.Generator, history: list, verbose: bool = False) -> None:
    """Second stage (v0.1.9): match the OBSERVED marginals instead of a constructed path.

    A path loss can only be as good as its path. On the synthetic regulatory benchmark the spline family's own
    marginal at the held-out timepoint sits at energy distance 0.636 while the field trained on it reaches
    0.641: the construction, not the network, is the ceiling (scripts_dev/diag_ceiling.py). This stage asks the
    field to push each observed snapshot onto the next one, scored by a differentiable two-sample divergence,
    so the interior of the transition is left to the field's own smoothness and to consistency with the other
    observed transitions. The path loss is kept with weight ``marginal_prior`` as a regulariser; early stopping
    still uses the validation timepoint, and the stage can only replace the pre-trained weights by improving it.
    """
    params = list(model.context_encoder.parameters()) + list(model.vector_field.parameters())
    opt = torch.optim.Adam(params, lr=config.marginal_lr, weight_decay=config.weight_decay)
    ema = WeightEMA({"ctx": model.context_encoder, "vf": model.vector_field}, config.ema_decay) if config.ema_decay > 0 else None
    gen = torch.Generator().manual_seed(int(config.seed) + 9973)
    w = np.array([min(len(t.x0), len(t.x1)) for t in transitions], dtype=np.float64)
    w /= w.sum()
    pw = None
    if paths:
        pw = np.array([p.n for _, p in paths], dtype=np.float64)
        pw /= pw.sum()
    best_val = model.best_val if (val_task is not None and model.best_val is not None) else float("inf")
    best_state = model.snapshot_state() if val_task is not None else None
    best_iter, bad = 0, 0
    for it in range(1, int(config.marginal_iters) + 1):
        tr = transitions[rng.choice(len(transitions), p=w)]
        b = min(int(config.marginal_batch), len(tr.x0), len(tr.x1))
        if b < 2:
            continue
        x0 = torch.as_tensor(tr.x0[rng.choice(len(tr.x0), b, replace=False)], dtype=torch.float32)
        x1 = torch.as_tensor(tr.x1[rng.choice(len(tr.x1), b, replace=False)], dtype=torch.float32)
        horizon = (tr.bio_t1 - tr.bio_t0) if config.clock == "duration" else 1.0
        steps = max(int(config.marginal_steps), int(math.ceil(config.marginal_steps * horizon)))
        loss = config.marginal_weight * marginal_loss(model.field(tr.bio_t0), x0, x1,
                                                      model._context(tr.condition, tr.bio_t0, tr.bio_t1, b),
                                                      horizon, steps, config.ode_method, config.marginal_metric, gen)
        if paths and config.marginal_prior > 0:
            cond, path = paths[rng.choice(len(paths), p=pw)]
            pb = min(config.batch_size, path.n)
            xs, vs, us = path.sample(rng, pb, window=config.marginal_window)
            loss = loss + config.marginal_prior * spline_loss(
                model.vector_field, torch.as_tensor(xs, dtype=torch.float32),
                torch.as_tensor(vs, dtype=torch.float32), model._context(cond, 0.0, 1.0, pb), config.sigma_min,
                time=model.path_time(us))
        opt.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
        opt.step()
        if ema is not None:
            ema.update()
        if it % config.marginal_eval_every == 0 or it == config.marginal_iters:
            rec = {"iter": it, "stage": "marginal", "loss": float(loss.item())}
            if val_task is not None:
                raw_state = model.snapshot_state()
                if ema is not None:
                    model.load_snapshot(ema.state())          # validate, and keep, the averaged weights
                v = _validate(model, snapshots, x_std, val_task, protocol)
                rec["val_energy_distance"] = v
                if v < best_val - 1e-9:
                    best_val, best_iter, best_state, bad = v, it, model.snapshot_state(), 0
                else:
                    bad += 1
                model.load_snapshot(raw_state)
            history.append(rec)
            if verbose:
                print(rec, flush=True)
            if val_task is not None and bad >= config.marginal_patience:
                break
    if best_state is not None:
        model.load_snapshot(best_state)
        model.best_val = best_val
    elif ema is not None:
        model.load_snapshot(ema.state())
    model.marginal_iter = best_iter


def train_flow(snapshots: SnapshotSet, fit_mask, fit_times, config: TrainConfig, val_task: ForecastTask | None = None,
               protocol: EvalProtocol | None = None, verbose: bool = False):
    seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
    gen = torch.Generator().manual_seed(config.seed)
    protocol = EvalProtocol() if protocol is None else protocol
    fit_mask = np.asarray(fit_mask, dtype=bool)
    fit_times = sorted(float(t) for t in fit_times)
    if val_task is not None and (fit_mask & val_task.target_mask).any():
        raise ValueError("validation target cells overlap the fit cells")
    standardizer = fit_on_train(snapshots.x, fit_mask)
    x_std = standardizer.transform(snapshots.x)
    fit = snapshots.subset(fit_mask)
    x_fit = x_std[fit_mask]
    if config.shuffle_biological_time:          # control: destroy temporal order within each condition
        bt = fit.biological_time.copy()
        for cond in fit.conditions():
            idx = np.flatnonzero(fit.condition == cond)
            bt[idx] = rng.permutation(bt[idx])
        fit = SnapshotSet(fit.x, bt, fit.condition, fit.replicate, fit.gene_names, name=fit.name,
                          require_two_timepoints=False)
    transitions = build_transitions(fit, fit_times, x_fit, pair_mode=config.pair_mode)
    paths = (spline_paths(fit, fit_times, x_fit, rng, chain=config.chain, bc_type=config.spline_kind,
                          center=config.path_center)
             if config.path == "spline" else None)
    registry = ConditionRegistry(fit.conditions())
    t_all = list(fit_times) + ([val_task.bio_t1] if val_task is not None else [])
    time_scale = max(max(abs(t) for t in t_all), 1e-6)
    ctx, vf = build_networks(config, registry, snapshots.n_genes, time_scale)
    model = FlowModel(registry, standardizer, ctx, vf, config, snapshots.gene_names, time_scale)
    params = list(ctx.parameters()) + list(vf.parameters())
    opt = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    ema = WeightEMA({"ctx": ctx, "vf": vf}, config.ema_decay) if config.ema_decay > 0 else None
    weights = np.array([min(len(t.x0), len(t.x1)) for t in transitions], dtype=np.float64)
    weights /= weights.sum()
    if paths is not None:
        path_weights = np.array([p.n for _, p in paths], dtype=np.float64)
        path_weights /= path_weights.sum()
    history, best_state, best_val, best_iter, bad = [], None, float("inf"), config.n_iters, 0
    t_start = time.time()
    for it in range(1, config.n_iters + 1):
        if paths is not None:                        # curved multi-marginal paths, autonomous velocity target
            cond, path = paths[rng.choice(len(paths), p=path_weights)]
            b = min(config.batch_size, path.n)
            xs, vs, us = path.sample(rng, b)
            context = model._context(cond, 0.0, 1.0, b)
            loss = spline_loss(vf, torch.as_tensor(xs, dtype=torch.float32), torch.as_tensor(vs, dtype=torch.float32),
                               context, config.sigma_min, time=model.path_time(us))
        else:
            tr = transitions[rng.choice(len(transitions), p=weights)]
            b = min(config.batch_size, len(tr.x0), len(tr.x1))
            if config.coupling == "ot":
                i0, i1 = ot_coupling(tr.x0, tr.x1, rng, max_size=b)
            elif config.coupling == "independent":
                i0, i1 = independent_coupling(len(tr.x0), len(tr.x1), rng, size=b)
            else:
                raise ValueError(f"unknown coupling {config.coupling!r}")
            x0 = torch.as_tensor(tr.x0[i0], dtype=torch.float32)
            x1 = torch.as_tensor(tr.x1[i1], dtype=torch.float32)
            context = model._context(tr.condition, tr.bio_t0, tr.bio_t1, b)
            scale = 1.0 / (tr.bio_t1 - tr.bio_t0) if config.clock == "duration" else 1.0
            loss = cfm_loss(vf, x0, x1, context, sample_flow_time(b, gen), config.sigma_min, target_scale=scale)
        opt.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
        opt.step()
        if ema is not None:
            ema.update()
        if it % config.eval_every == 0 or it == config.n_iters:
            rec = {"iter": it, "loss": loss.item(), "seconds": time.time() - t_start}
            if val_task is not None:
                raw_state = model.snapshot_state()
                if ema is not None:
                    model.load_snapshot(ema.state())        # validate, and keep, the averaged weights
                v = _validate(model, snapshots, x_std, val_task, protocol)
                rec["val_energy_distance"] = v
                if v < best_val - 1e-9:
                    best_val, best_iter, best_state, bad = v, it, model.snapshot_state(), 0
                else:
                    bad += 1
                model.load_snapshot(raw_state)
            history.append(rec)
            if verbose:
                print(rec, flush=True)
            if val_task is not None and bad >= config.patience:
                break
    if best_state is not None:
        model.load_snapshot(best_state)
    elif ema is not None:
        model.load_snapshot(ema.state())
    model.best_iter = best_iter
    model.best_val = best_val if val_task is not None else None
    if config.marginal_iters > 0:
        _finetune_marginal(model, snapshots, x_std, transitions, paths, config, val_task, protocol, rng, history, verbose)
    model.train_seconds = time.time() - t_start
    return model, history
```

- [ ] **Step 4: `python -m pytest tests/test_trainer.py tests/test_leakage.py -q` → 6 passed**

### Task 14: Grid search annidata e benchmark di forecast con controlli

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/training/search.py`
- Test: `cfm_grn_v2/tests/test_search.py`

**Interfaces:**
- Consumes: `train_flow`, `ForecastTask`, `task_groups`, `build_transitions`, controlli del Task 11, `EvalProtocol`, `median_bandwidth`, `noise_floor`, `Standardizer`, `expand_grid`, `random_subset`.
- Produces: `nested_grid_search(snapshots, fit_mask, fit_times, val_task, base_config, grid: dict | list[dict], seeds, protocol, max_configs=None, seed=0, verbose=False) -> (TrainConfig, DataFrame)`; `run_forecast_benchmark(snapshots, fit_mask, fit_times, val_task, test_task, config, seeds, protocol, dataset_name, refit_on_all=False, shuffled_control=True, mlp_kwargs=None, linear_alpha=1.0, verbose=False) -> DataFrame` con colonne `dataset, model, seed, group, condition, replicate, energy_distance, mmd, w2, mean_error, per_gene_w1` (+ `train_seconds, n_params, best_iter` per il FM) e righe `model="noise_floor"`; `select_best(table, base_config) -> TrainConfig` (media su seed e fold); `interpolation_val_folds(times, test_time, n_folds=2) -> list[float]`; `interpolation_design(times, test_time, val_time=None) -> dict(times, test_time, t_prev, t_next, obs_times, val_time, v_prev, fit_times, horizon, frac)`; `run_interpolation_benchmark(snapshots, times, test_time, config, seeds, protocol, dataset_name, val_time=None, refit_on_all=False, shuffled_control=True, mlp_kwargs=None, linear_alpha=1.0, verbose=False) -> DataFrame` (stesse colonne; controlli `persistence, mean_interp, ot_interp, linear, mlp, freefm_replay`, senza `train_mean`).

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_search.py
import numpy as np
import pandas as pd
import pytest

from cfm_grn_v2.data.schema import SnapshotSet
from cfm_grn_v2.data.synthetic_toys import make_toy
from cfm_grn_v2.evaluation.metrics import EvalProtocol
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.search import (interpolation_design, interpolation_val_folds, nested_grid_search,
                                        run_forecast_benchmark, run_interpolation_benchmark, select_best)
from cfm_grn_v2.training.trainer import replicate_task


def _drift_set(n=120, seed=0):
    """Four timepoints of a constant-velocity drift: the McCann interpolant recovers the inner one exactly."""
    rng = np.random.default_rng(seed)
    v = np.array([1.0, -0.5])
    xs, ts = [], []
    for t in (0.0, 1.0, 2.0, 3.0):
        xs.append(rng.normal(size=(n, 2)) * 0.1 + t * v)
        ts.append(np.full(n, t))
    x = np.concatenate(xs).astype(np.float32)
    return SnapshotSet(x, np.concatenate(ts), np.array(["ctrl"] * (4 * n)), np.array(["r1"] * (4 * n)), ["a", "b"])


def test_val_folds_and_select_best_over_folds():
    assert interpolation_val_folds([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3.0) == [1.0, 5.0]
    assert interpolation_val_folds([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2.0) == [1.0, 5.0]
    assert interpolation_val_folds([0.0, 1.0, 2.0, 3.0], 2.0, n_folds=2) == [1.0]
    table = pd.DataFrame([{"config_id": 0, "hidden": 16, "seed": 0, "val_energy_distance": 0.1, "val_time": 1.0},
                          {"config_id": 0, "hidden": 16, "seed": 0, "val_energy_distance": 0.9, "val_time": 5.0},
                          {"config_id": 1, "hidden": 24, "seed": 0, "val_energy_distance": 0.4, "val_time": 1.0},
                          {"config_id": 1, "hidden": 24, "seed": 0, "val_energy_distance": 0.4, "val_time": 5.0}])
    best = select_best(table, TrainConfig())
    assert best.hidden == 24 and isinstance(best.hidden, int)


def test_interpolation_design_and_benchmark_rows():
    d = interpolation_design([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3.0)
    assert d["val_time"] == 5.0 and d["fit_times"] == [0.0, 1.0, 2.0, 4.0, 6.0] and d["v_prev"] == 4.0
    assert d["t_prev"] == 2.0 and d["t_next"] == 4.0 and d["horizon"] == 1.0 and d["frac"] == 0.5
    with pytest.raises(ValueError):
        interpolation_design([0.0, 1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        interpolation_design([0.0, 1.0, 2.0], 1.0)            # no second inner timepoint for validation
    s = _drift_set()
    cfg = TrainConfig(hidden=16, depth=1, n_iters=40, batch_size=64, eval_every=20, patience=2, clock="duration")
    df = run_interpolation_benchmark(s, [0.0, 1.0, 2.0, 3.0], 2.0, cfg, seeds=[0],
                                     protocol=EvalProtocol(n_samples=100, n_perm=5), dataset_name="drift",
                                     mlp_kwargs={"hidden": 16, "n_iters": 50})
    models = set(df["model"])
    for m in ("flow_matching", "flow_matching_shuffled_time", "persistence", "mean_interp", "ot_interp", "linear",
              "mlp", "freefm_replay", "noise_floor"):
        assert m in models
    assert "train_mean" not in models and df["energy_distance"].notna().all()
    ed = df.groupby("model")["energy_distance"].mean()
    assert ed["ot_interp"] < ed["persistence"] and ed["mean_interp"] < ed["persistence"]
    assert ed["freefm_replay"] < ed["persistence"]


def test_nested_search_and_benchmark_rows():
    s = make_toy("translation", n_per_group=200, seed=0)
    fit = s.mask(replicates=["r1"])
    val = replicate_task(s, "r2", 0.0, 1.0)
    test = replicate_task(s, "r3", 0.0, 1.0)
    base = TrainConfig(hidden=16, depth=1, n_iters=40, batch_size=64, eval_every=20, patience=2)
    proto = EvalProtocol(n_samples=150, seed=0)
    best, table = nested_grid_search(s, fit, [0.0, 1.0], val, base, {"hidden": [16, 24]}, seeds=[0], protocol=proto)
    assert best.hidden in (16, 24) and len(table) == 2 and "val_energy_distance" in table
    _, explicit = nested_grid_search(s, fit, [0.0, 1.0], val, base, [{"hidden": 16}], seeds=[0], protocol=proto)
    assert len(explicit) == 1
    df = run_forecast_benchmark(s, fit, [0.0, 1.0], val, test, best, seeds=[0, 1], protocol=proto,
                                dataset_name="toy", mlp_kwargs={"hidden": 16, "n_iters": 50})
    models = set(df["model"])
    for m in ("flow_matching", "flow_matching_shuffled_time", "persistence", "train_mean", "mean_shift", "linear",
              "mlp", "freefm_replay", "noise_floor"):
        assert m in models
    assert set(df["seed"]) == {0, 1} and df["energy_distance"].notna().all()
    assert (df[df["model"] == "flow_matching"]["best_iter"] >= 20).all()
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/training/search.py
"""Nested grid search (validation timepoint only), the forecast benchmark and the leave-one-inner-timepoint-out
interpolation benchmark, both with mandatory controls fitted on the same observed cells."""
from __future__ import annotations

import time
from itertools import zip_longest

import numpy as np
import pandas as pd

from ..data.preprocessing import Standardizer
from ..evaluation.controls import LinearMap, MLPMap, freefm_replay, mean_shift, ot_pairs, persistence, train_mean
from ..evaluation.metrics import EvalProtocol, median_bandwidth, noise_floor
from ..flows.coupling import ot_coupling
from .config import expand_grid, random_subset
from .trainer import build_transitions, task_groups, time_task, train_flow


def nested_grid_search(snapshots, fit_mask, fit_times, val_task, base_config, grid, seeds, protocol,
                       max_configs=None, seed=0, verbose=False):
    """Select the configuration with the lowest mean validation energy distance. The test is never touched."""
    configs = list(grid) if isinstance(grid, list) else expand_grid(grid)
    if max_configs is not None:
        configs = random_subset(configs, max_configs, seed)
    rows = []
    for k, over in enumerate(configs):
        for s in seeds:
            cfg = base_config.replace(**over, seed=s)
            t0 = time.time()
            model, _ = train_flow(snapshots, fit_mask, fit_times, cfg, val_task, protocol)
            rows.append({"config_id": k, **over, "seed": s, "val_energy_distance": model.best_val,
                         "best_iter": model.best_iter, "seconds": time.time() - t0})
            if verbose:
                print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    return select_best(table, base_config), table


_TABLE_COLS = {"config_id", "seed", "val_energy_distance", "best_iter", "seconds", "dataset", "val_time"}


def select_best(table: pd.DataFrame, base_config):
    """Configuration with the lowest mean validation energy distance over seeds (and, if the table stacks
    several validation folds, over folds)."""
    best_id = table.groupby("config_id")["val_energy_distance"].mean().idxmin()
    sub = table[table["config_id"] == best_id]
    over = {k: sub[k].iloc[0] for k in table.columns if k not in _TABLE_COLS}     # column-wise: dtypes preserved
    return base_config.replace(**{k: (v.item() if hasattr(v, "item") else v) for k, v in over.items()})


def interpolation_val_folds(times, test_time, n_folds: int = 2) -> list:
    """Inner observed timepoints usable for nested validation of an interpolation fold: farthest from
    test_time first, alternating sides when possible, at most n_folds."""
    times = sorted(float(t) for t in times)
    test_time = float(test_time)
    inner = [t for t in times if t != test_time][1:-1]
    before = sorted(t for t in inner if t < test_time)
    after = sorted((t for t in inner if t > test_time), reverse=True)
    picks = [t for pair in zip_longest(before, after) for t in pair if t is not None]
    return picks[:n_folds]


def interpolation_design(times, test_time, val_time=None) -> dict:
    """Leave-one-inner-timepoint-out design: the held-out timepoint is predicted from the previous observed
    one, and the transition that brackets the gap (t_prev -> t_next) is observed. Nested validation holds
    out a second inner timepoint (default: the observed inner timepoint farthest from test_time)."""
    times = sorted(float(t) for t in times)
    test_time = float(test_time)
    if test_time not in times[1:-1]:
        raise ValueError(f"test_time {test_time} must be an inner timepoint of {times}")
    k = times.index(test_time)
    t_prev, t_next = times[k - 1], times[k + 1]
    obs_times = [t for t in times if t != test_time]
    inner = obs_times[1:-1]
    if not inner:
        raise ValueError("interpolation with nested validation needs at least four timepoints")
    val_time = max(inner, key=lambda t: (abs(t - test_time), t)) if val_time is None else float(val_time)
    if val_time not in inner:
        raise ValueError(f"val_time {val_time} must be an inner observed timepoint, one of {inner}")
    fit_times = [t for t in obs_times if t != val_time]
    v_prev = max(t for t in fit_times if t < val_time)
    return {"times": times, "test_time": test_time, "t_prev": t_prev, "t_next": t_next, "obs_times": obs_times,
            "val_time": val_time, "v_prev": v_prev, "fit_times": fit_times, "horizon": test_time - t_prev,
            "frac": (test_time - t_prev) / (t_next - t_prev)}


def _horizon_pairs(snapshots, x_eval, all_mask, obs_times, cond, horizon, rng, max_pairs=1000):
    """Pooled OT pairs over the observed pairs of timepoints of one condition whose horizon equals the test
    horizon; if none exists, every consecutive transition with its displacement rescaled to that horizon."""
    sub = snapshots.subset(all_mask)
    x_sub = x_eval[all_mask]
    same = [tr for tr in build_transitions(sub, obs_times, x_sub, pair_mode="all")
            if tr.condition == cond and abs((tr.bio_t1 - tr.bio_t0) - horizon) < 1e-9]
    scaled = not same
    trs = same or [tr for tr in build_transitions(sub, obs_times, x_sub) if tr.condition == cond]
    p0, p1 = [], []
    for tr in trs:
        a, b = ot_pairs(tr.x0, tr.x1, rng, max_size=max_pairs)
        if scaled:
            b = a + (b - a) * (horizon / (tr.bio_t1 - tr.bio_t0))
        p0.append(a)
        p1.append(b)
    if not p0:
        raise ValueError(f"no observed transition for condition {cond!r}")
    return np.concatenate(p0), np.concatenate(p1)


def run_interpolation_benchmark(snapshots, times, test_time, config, seeds, protocol, dataset_name, val_time=None,
                                refit_on_all=False, shuffled_control=True, mlp_kwargs=None, linear_alpha=1.0,
                                verbose=False):
    """One row per (model, seed, condition|replicate group) for a held-out inner timepoint.

    The flow model is early-stopped on the nested validation timepoint and, if refit_on_all, retrained on
    every observed timepoint for the selected number of iterations; it predicts test_time from the cells at
    t_prev over the horizon test_time - t_prev. Controls (same observed cells, same evaluation space):
    persistence; mean_interp = x0 + frac * (mean x(t_next) - mean x(t_prev)); ot_interp = displacement
    (McCann) interpolation along the OT pairs of the bracketing transition; linear/mlp maps fitted on the
    observed pairs with the test horizon; freefm_replay = the velocity of the transition arriving at t_prev
    replayed at the nearest arrival position (the bracketing pairs when t_prev is the first timepoint).
    """
    d = interpolation_design(times, test_time, val_time)
    fit_mask = snapshots.mask(times=d["fit_times"])
    all_mask = snapshots.mask(times=d["obs_times"])
    val_task = time_task(snapshots, d["v_prev"], d["val_time"])
    test_task = time_task(snapshots, d["t_prev"], d["test_time"])
    if (all_mask & test_task.target_mask).any():
        raise ValueError("test target cells overlap the observed cells")
    eval_std = Standardizer().fit(snapshots.x[all_mask])
    x_eval = eval_std.transform(snapshots.x)
    protocol = EvalProtocol(n_samples=protocol.n_samples, seed=protocol.seed,
                            bandwidth=median_bandwidth(x_eval[all_mask]), w2_max=protocol.w2_max,
                            n_perm=protocol.n_perm)
    groups = task_groups(snapshots, test_task)
    if not groups:
        raise ValueError("test task has no (condition, replicate) group with cells on both sides")
    horizon, frac = d["horizon"], d["frac"]
    rows = []

    def add(model_name, seed, cond, rep, pred, target, extra=None):
        rows.append({"dataset": dataset_name, "model": model_name, "seed": seed, "group": f"{cond}|{rep}",
                     "condition": cond, "replicate": rep, **protocol.evaluate(pred, target, seed_offset=seed),
                     **(extra or {})})

    for seed in seeds:
        variants = [("flow_matching", config.replace(seed=seed))]
        if shuffled_control:
            variants.append(("flow_matching_shuffled_time", config.replace(seed=seed, shuffle_biological_time=True)))
        for name, cfg in variants:
            t0 = time.time()
            model, _ = train_flow(snapshots, fit_mask, d["fit_times"], cfg, val_task, protocol)
            if refit_on_all:
                n_iters = max(int(model.best_iter), cfg.eval_every)
                model, _ = train_flow(snapshots, all_mask, d["obs_times"], cfg.replace(n_iters=n_iters), None, protocol)
            secs = time.time() - t0
            for cond, rep, m0, m1 in groups:
                if cond not in model.registry.names:
                    continue
                pred = eval_std.transform(model.predict(snapshots.x[m0], cond, d["t_prev"], d["test_time"]))
                add(name, seed, cond, rep, pred, x_eval[m1],
                    {"train_seconds": secs, "n_params": model.n_params, "best_iter": model.best_iter})
            if verbose:
                print(f"[{dataset_name}] {name} seed={seed} done in {secs:.0f}s", flush=True)
        rng = np.random.default_rng(20_000 + seed)
        for cond, rep, m0, m1 in groups:
            x0, target = x_eval[m0], x_eval[m1]
            x_next = x_eval[all_mask & snapshots.mask(times=[d["t_next"]], conditions=[cond], replicates=[rep])]
            add("persistence", seed, cond, rep, persistence(x0), target)
            add("mean_interp", seed, cond, rep, x0 + frac * (x_next.mean(0) - x0.mean(0)), target)
            i0, i1 = ot_coupling(x0, x_next, rng, max_size=len(x0))
            add("ot_interp", seed, cond, rep, x0[i0] + frac * (x_next[i1] - x0[i0]), target)
            p0, p1 = _horizon_pairs(snapshots, x_eval, all_mask, d["obs_times"], cond, horizon, rng)
            add("linear", seed, cond, rep, LinearMap(alpha=linear_alpha).fit(p0, p1).predict(x0), target)
            add("mlp", seed, cond, rep, MLPMap(seed=seed, **(mlp_kwargs or {})).fit(p0, p1).predict(x0), target)
            before = [t for t in d["obs_times"] if t < d["t_prev"]]
            if before:
                t_pp = before[-1]
                x_pp = x_eval[all_mask & snapshots.mask(times=[t_pp], conditions=[cond], replicates=[rep])]
                q0, q1 = ot_pairs(x_pp, x0, rng, max_size=1000)
                arrival = q1 + (q1 - q0) * (horizon / (d["t_prev"] - t_pp))
                add("freefm_replay", seed, cond, rep, freefm_replay(x0, q1, arrival, k=1), target)
            else:
                q0, q1 = ot_pairs(x0, x_next, rng, max_size=1000)
                add("freefm_replay", seed, cond, rep, freefm_replay(x0, q0, q0 + frac * (q1 - q0), k=1), target)
            rows.append({"dataset": dataset_name, "model": "noise_floor", "seed": seed, "group": f"{cond}|{rep}",
                         "condition": cond, "replicate": rep, **noise_floor(target, protocol)})
    return pd.DataFrame(rows)


def _control_pairs(snapshots, x_eval, all_mask, all_times, cond, rng, max_pairs=1000):
    """Pooled OT pairs over every observed transition of one condition (fit + validation cells only)."""
    sub = snapshots.subset(all_mask)
    p0, p1 = [], []
    for tr in build_transitions(sub, all_times, x_eval[all_mask]):
        if tr.condition == cond:
            a, b = ot_pairs(tr.x0, tr.x1, rng, max_size=max_pairs)
            p0.append(a)
            p1.append(b)
    if not p0:
        raise ValueError(f"no observed transition for condition {cond!r}")
    return np.concatenate(p0), np.concatenate(p1)


def run_forecast_benchmark(snapshots, fit_mask, fit_times, val_task, test_task, config, seeds, protocol, dataset_name,
                           refit_on_all=False, shuffled_control=True, mlp_kwargs=None, linear_alpha=1.0, verbose=False):
    """One row per (model, seed, condition|replicate group) on test_task.

    Evaluation space: standardised on the observed cells (fit + validation). The flow model is
    early-stopped on the validation task and, if refit_on_all, retrained on fit + validation for the
    selected number of iterations. Controls are fitted on the same observed cells. Test cells are
    used only as the forecast source (bio_t0) and as the target to score.
    """
    fit_mask = np.asarray(fit_mask, dtype=bool)
    all_mask = fit_mask | val_task.source_mask | val_task.target_mask
    if (all_mask & test_task.target_mask).any():
        raise ValueError("test target cells overlap the observed (fit + validation) cells")
    all_times = sorted({float(t) for t in fit_times} | {val_task.bio_t0, val_task.bio_t1})
    eval_std = Standardizer().fit(snapshots.x[all_mask])
    x_eval = eval_std.transform(snapshots.x)
    protocol = EvalProtocol(n_samples=protocol.n_samples, seed=protocol.seed,
                            bandwidth=median_bandwidth(x_eval[all_mask]), w2_max=protocol.w2_max)
    groups = task_groups(snapshots, test_task)
    if not groups:
        raise ValueError("test task has no (condition, replicate) group with cells on both sides")
    rows = []

    def add(model_name, seed, cond, rep, pred, target, extra=None):
        rows.append({"dataset": dataset_name, "model": model_name, "seed": seed, "group": f"{cond}|{rep}",
                     "condition": cond, "replicate": rep, **protocol.evaluate(pred, target, seed_offset=seed),
                     **(extra or {})})

    for seed in seeds:
        variants = [("flow_matching", config.replace(seed=seed))]
        if shuffled_control:
            variants.append(("flow_matching_shuffled_time", config.replace(seed=seed, shuffle_biological_time=True)))
        for name, cfg in variants:
            t0 = time.time()
            model, _ = train_flow(snapshots, fit_mask, fit_times, cfg, val_task, protocol)
            if refit_on_all:
                n_iters = max(int(model.best_iter), cfg.eval_every)
                model, _ = train_flow(snapshots, all_mask, all_times, cfg.replace(n_iters=n_iters), None, protocol)
            secs = time.time() - t0
            for cond, rep, m0, m1 in groups:
                if cond not in model.registry.names:
                    continue
                pred = eval_std.transform(model.predict(snapshots.x[m0], cond, test_task.bio_t0, test_task.bio_t1))
                add(name, seed, cond, rep, pred, x_eval[m1],
                    {"train_seconds": secs, "n_params": model.n_params, "best_iter": model.best_iter})
            if verbose:
                print(f"[{dataset_name}] {name} seed={seed} done in {secs:.0f}s", flush=True)
        rng = np.random.default_rng(10_000 + seed)
        prev_t, last_t = all_times[-2], all_times[-1]
        for cond, rep, m0, m1 in groups:
            x0, target = x_eval[m0], x_eval[m1]
            obs_prev = x_eval[all_mask & snapshots.mask(times=[prev_t], conditions=[cond])]
            obs_last = x_eval[all_mask & snapshots.mask(times=[last_t], conditions=[cond])]
            add("persistence", seed, cond, rep, persistence(x0), target)
            add("train_mean", seed, cond, rep, train_mean(obs_last, len(x0)), target)
            add("mean_shift", seed, cond, rep, mean_shift(x0, obs_prev, obs_last), target)
            p0, p1 = _control_pairs(snapshots, x_eval, all_mask, all_times, cond, rng)
            add("linear", seed, cond, rep, LinearMap(alpha=linear_alpha).fit(p0, p1).predict(x0), target)
            add("mlp", seed, cond, rep, MLPMap(seed=seed, **(mlp_kwargs or {})).fit(p0, p1).predict(x0), target)
            add("freefm_replay", seed, cond, rep, freefm_replay(x0, p0, p1, k=1), target)
            rows.append({"dataset": dataset_name, "model": "noise_floor", "seed": seed, "group": f"{cond}|{rep}",
                         "condition": cond, "replicate": rep, **noise_floor(target, protocol)})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: `python -m pytest tests/test_search.py -q` → 1 passed**

### Task 15: Report, IC95, differenze appaiate, verdetto

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/evaluation/report.py`
- Test: `cfm_grn_v2/tests/test_report.py`

**Interfaces:**
- Produces: `METRICS`, `CONTROLS`; `per_seed(df)`; `summarize(df) -> DataFrame[dataset, model, metric, mean, std, ci95, n]`; `paired_difference(df, dataset, model, reference, metric="energy_distance") -> dict(mean, ci_low, ci_high, n, ...)`; `verdict(df, dataset, ..., mode="benchmark"|"sanity") -> dict(mode, pass, matches_target, null_q95, beats_all_controls, best_control, paired_vs_best_control, uses_time, paired_vs_shuffled_time, fm_mean, noise_floor, within_3x_noise_floor)`; `write_report(df, md_path, json_path, modes=None) -> dict[dataset, verdict]`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_report.py
import numpy as np
import pandas as pd

from cfm_grn_v2.evaluation.report import METRICS, paired_difference, summarize, verdict, write_report


def _fake_rows():
    rows = []
    rng = np.random.default_rng(0)
    levels = (("flow_matching", 0.1), ("flow_matching_shuffled_time", 0.9), ("persistence", 1.0), ("mean_shift", 0.6),
              ("linear", 0.5), ("mlp", 0.5), ("train_mean", 1.5), ("freefm_replay", 0.7), ("noise_floor", 0.05))
    for seed in range(3):
        for model, base in levels:
            rows.append({"dataset": "toy", "model": model, "seed": seed, "group": "ctrl|r3", "null_q95": 0.12,
                         **{m: base + 0.01 * rng.normal() for m in METRICS}})
    return pd.DataFrame(rows)


def test_summary_paired_and_verdict(tmp_path):
    df = _fake_rows()
    summ = summarize(df)
    assert {"dataset", "model", "metric", "mean", "std", "ci95", "n"} <= set(summ.columns)
    d = paired_difference(df, "toy", "flow_matching", "persistence")
    assert d["ci_high"] < 0 and d["n"] == 3
    v = verdict(df, "toy")
    assert v["pass"] and v["uses_time"] and v["best_control"] in ("linear", "mlp") and v["mode"] == "benchmark"
    assert verdict(df, "toy", mode="sanity")["pass"]
    worse = df.copy()
    worse.loc[worse["model"] == "flow_matching", "energy_distance"] = 0.3
    assert not verdict(worse, "toy", mode="sanity")["pass"]
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    out = write_report(df, md, js)
    assert md.exists() and js.exists() and "flow_matching" in md.read_text(encoding="utf-8") and out["toy"]["pass"]
```

- [ ] **Step 2: eseguire → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/evaluation/report.py
"""Aggregation over seeds, confidence intervals, paired differences and the gate verdict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["energy_distance", "mmd", "w2", "mean_error", "per_gene_w1"]
CONTROLS = ["persistence", "train_mean", "mean_shift", "mean_interp", "ot_interp", "linear", "mlp", "freefm_replay"]


def per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Average the groups of each (dataset, model, seed) first: seeds are the replication unit."""
    cols = [c for c in METRICS + ["null_q95"] if c in df.columns]
    return df.groupby(["dataset", "model", "seed"], as_index=False)[cols].mean()


def _ci95(values) -> float:
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, model), g in per_seed(df).groupby(["dataset", "model"]):
        for m in METRICS:
            rows.append({"dataset": ds, "model": model, "metric": m, "mean": float(g[m].mean()),
                         "std": float(g[m].std(ddof=1)) if len(g) > 1 else 0.0, "ci95": _ci95(g[m]), "n": int(len(g))})
    return pd.DataFrame(rows)


def paired_difference(df: pd.DataFrame, dataset: str, model: str, reference: str, metric: str = "energy_distance") -> dict:
    ps = per_seed(df)
    ps = ps[ps["dataset"] == dataset]
    a = ps[ps["model"] == model].set_index("seed")[metric]
    b = ps[ps["model"] == reference].set_index("seed")[metric]
    seeds = sorted(set(a.index) & set(b.index))
    if not seeds:
        raise ValueError(f"no common seed between {model!r} and {reference!r} on {dataset!r}")
    d = np.array([a[s] - b[s] for s in seeds], dtype=np.float64)
    ci = _ci95(d)
    return {"model": model, "reference": reference, "metric": metric, "mean": float(d.mean()),
            "ci_low": float(d.mean() - ci), "ci_high": float(d.mean() + ci), "n": int(len(d))}


def verdict(df: pd.DataFrame, dataset: str, fm: str = "flow_matching", shuffled: str = "flow_matching_shuffled_time",
            controls=CONTROLS, metric: str = "energy_distance", mode: str = "benchmark") -> dict:
    """mode="benchmark": FM must beat every control (paired CI95 < 0 against the best one) and use time.
    mode="sanity": FM must be indistinguishable from the target under the energy-distance permutation
    test (mean ED <= mean 95% null quantile) and use time (toys whose optimum is a linear map cannot be
    beaten, only matched)."""
    if mode not in ("benchmark", "sanity"):
        raise ValueError(f"unknown verdict mode {mode!r}")
    ps = per_seed(df)
    ps = ps[ps["dataset"] == dataset]
    means = ps.groupby("model")[metric].mean()
    present = [c for c in controls if c in means.index]
    if fm not in means.index or not present:
        raise ValueError(f"dataset {dataset!r} lacks {fm!r} or every control")
    best_control = min(present, key=lambda c: float(means[c]))
    d_best = paired_difference(df, dataset, fm, best_control, metric)
    beats_all = all(float(means[fm]) < float(means[c]) for c in present)
    ci_ok = d_best["ci_high"] < 0 if not np.isnan(d_best["ci_high"]) else d_best["mean"] < 0
    uses_time, d_shuf = None, None
    if shuffled in means.index:
        d_shuf = paired_difference(df, dataset, fm, shuffled, metric)
        uses_time = bool(d_shuf["mean"] < 0 and (np.isnan(d_shuf["ci_high"]) or d_shuf["ci_high"] < 0))
    floor = float(means["noise_floor"]) if "noise_floor" in means.index else float("nan")
    if "null_q95" in ps.columns:
        null_q95 = float(ps[ps["model"] == fm]["null_q95"].mean())
        matches_target = bool(float(means[fm]) <= null_q95)
    else:
        null_q95, matches_target = float("nan"), False
    core = (beats_all and ci_ok) if mode == "benchmark" else matches_target
    return {"dataset": dataset, "metric": metric, "mode": mode, "pass": bool(core and (uses_time is None or uses_time)),
            "beats_all_controls": bool(beats_all), "best_control": best_control, "paired_vs_best_control": d_best,
            "uses_time": uses_time, "paired_vs_shuffled_time": d_shuf, "fm_mean": float(means[fm]),
            "noise_floor": floor, "null_q95": null_q95, "matches_target": matches_target}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def write_report(df: pd.DataFrame, md_path, json_path, modes: dict | None = None) -> dict:
    """modes: dataset -> "benchmark" | "sanity" (default benchmark)."""
    modes = modes or {}
    summ = summarize(df)
    verdicts = {ds: verdict(df, ds, mode=modes.get(ds, "benchmark")) for ds in sorted(df["dataset"].unique())}
    lines = ["# Forecast benchmark report", "", "Metrics in the evaluation space standardised on observed cells; "
             "mean ± CI95 over seeds (groups averaged within seed). Lower is better.", ""]
    for ds in sorted(df["dataset"].unique()):
        s = summ[summ["dataset"] == ds]
        lines += [f"## {ds}", "", "| model | " + " | ".join(METRICS) + " | n seeds |", "|---|" + "---|" * (len(METRICS) + 1)]
        order = sorted(s["model"].unique(), key=lambda m: float(s[(s["model"] == m) & (s["metric"] == "energy_distance")]["mean"].iloc[0]))
        for m in order:
            cells = []
            for met in METRICS:
                r = s[(s["model"] == m) & (s["metric"] == met)].iloc[0]
                cells.append(f"{r['mean']:.3f}±{r['ci95']:.3f}" if not np.isnan(r["ci95"]) else f"{r['mean']:.3f}")
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {int(s[s['model'] == m]['n'].iloc[0])} |")
        v = verdicts[ds]
        pb = v["paired_vs_best_control"]
        lines += ["", f"Verdict ({v['mode']}): **{'PASS' if v['pass'] else 'FAIL'}** — best control `{v['best_control']}`, paired ΔED = "
                  f"{pb['mean']:.3f} [{pb['ci_low']:.3f}, {pb['ci_high']:.3f}] (n={pb['n']}), uses_time = {v['uses_time']}, "
                  f"noise floor = {v['noise_floor']:.3f}, null q95 = {v['null_q95']:.3f}, matches target = {v['matches_target']}", ""]
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(json_path).write_text(json.dumps(verdicts, indent=1, default=_json_default), encoding="utf-8")
    return verdicts
```

- [ ] **Step 4: `python -m pytest tests/test_report.py -q` → 1 passed**

### Task 16: Runner della Fase 1 e test di integrazione

**Files:**
- Create: `cfm_grn_v2/src/cfm_grn_v2/experiments/__init__.py`, `cfm_grn_v2/src/cfm_grn_v2/experiments/run_synthetic.py`
- Test: `cfm_grn_v2/tests/test_integration_toys.py` (marcato `slow`)

**Interfaces:**
- Produces: CLI `PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic --out outputs/phase1 [--quick] [--seeds 3] [--search-seeds 2] [--datasets toys,grn] [--max-configs N] [--seed 0]`; scrive `results.csv`, `search.csv`, `report.md`, `verdict.json`, `run_meta.json`, `grn_truth.npz`; `main(argv) -> dict[dataset, verdict]`.

- [ ] **Step 1: test (RED)**

```python
# file: cfm_grn_v2/tests/test_integration_toys.py
import pytest

from cfm_grn_v2.experiments.run_synthetic import main


@pytest.mark.slow
def test_quick_toys_run_produces_verdicts(tmp_path):
    v = main(["--out", str(tmp_path), "--quick", "--seeds", "2", "--search-seeds", "1", "--datasets", "toys",
              "--max-configs", "2"])
    assert set(v) == {"toy_translation", "toy_variance", "toy_bimodal", "toy_condition_shift"}
    assert (tmp_path / "report.md").exists() and (tmp_path / "results.csv").exists()
```

- [ ] **Step 2: eseguire `python -m pytest tests/test_integration_toys.py -q -m slow` → FAIL**

- [ ] **Step 3: implementazione**

```python
# file: cfm_grn_v2/src/cfm_grn_v2/experiments/__init__.py
```

```python
# file: cfm_grn_v2/src/cfm_grn_v2/experiments/run_synthetic.py
"""Phase 1 benchmark: toy transitions + synthetic time-varying GRN, nested search, controls, report.

    PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic --out outputs/phase1            # full run
    PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic --out outputs/quick --quick     # smoke run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..data.splits import forecast_split, inner_validation_split, replicate_split
from ..data.synthetic_grn import TimeVaryingGRNSimulator
from ..data.synthetic_toys import TOY_NAMES, make_toy
from ..evaluation.metrics import EvalProtocol
from ..evaluation.report import write_report
from ..training.config import TrainConfig, expand_grid, random_subset
from ..training.search import (interpolation_design, interpolation_val_folds, nested_grid_search,
                               run_forecast_benchmark, run_interpolation_benchmark, select_best)
from ..training.trainer import replicate_task, time_task

# Toys have a single transition: delta/absolute conditioning and consecutive/all pairs coincide there.
TOY_GRID = {"hidden": [128, 256], "lr": [1e-3, 3e-4], "sigma_min": [1e-3, 1e-2], "coupling": ["ot", "independent"]}
# GRN: delta conditioning only (absolute time cannot extrapolate to an unseen interval, decision v0.1.1);
# pair_mode is the multi-marginal question (consecutive snapshots vs every ordered pair, horizon in the context).
# path="spline" chains OT across all timepoints (coupling and pair_mode irrelevant) and needs the duration clock.


def _grn_grid(hidden, lr, sigma_min, coupling, pair_mode, clock, n_linear=None, seed=0) -> list:
    """Linear-path configurations (optionally a random subset of n_linear) plus every spline configuration."""
    linear = expand_grid({"hidden": hidden, "lr": lr, "sigma_min": sigma_min, "coupling": coupling,
                          "pair_mode": pair_mode, "clock": clock, "path": ["linear"]})
    if n_linear is not None:
        linear = random_subset(linear, n_linear, seed)
    spline = expand_grid({"hidden": hidden, "lr": lr, "sigma_min": sigma_min, "coupling": ["ot"],
                          "pair_mode": ["consecutive"], "clock": ["duration"], "path": ["spline"]})
    return linear + spline


GRN_GRID = _grn_grid([128, 256], [1e-3], [1e-3, 1e-2], ["ot", "independent"], ["consecutive", "all"],
                     ["flow", "duration"], n_linear=12)                            # 12 of 32 linear + 4 spline
QUICK_TOY_GRID = {"hidden": [64, 128], "lr": [1e-3], "sigma_min": [1e-3], "coupling": ["ot"]}
QUICK_GRN_GRID = _grn_grid([128], [1e-3], [1e-3], ["ot"], ["consecutive", "all"], ["flow", "duration"])   # 4 + 1
VERDICT_MODES = {"toys": "sanity", "grn": "benchmark", "grn_interp": "benchmark"}
GRN_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def toy_setup(name: str, n_per_group: int, seed: int):
    s = make_toy(name, n_per_group, seed=seed)
    sp = replicate_split(s.replicates(), seed=seed)
    fit = s.mask(replicates=sp["fit"])
    val = replicate_task(s, sp["val"][0], 0.0, 1.0)
    test = replicate_task(s, sp["test"][0], 0.0, 1.0)
    return s, fit, [0.0, 1.0], val, test


def grn_setup(n_cells: int, seed: int, n_genes: int, n_regulators: int):
    sim = TimeVaryingGRNSimulator(n_genes=n_genes, n_regulators=n_regulators, seed=seed)
    s, truth = sim.snapshot_set(GRN_TIMES, n_cells_per_cohort=n_cells, replicates=("r1", "r2"), seed=seed)
    sp = forecast_split(GRN_TIMES)
    inner = inner_validation_split(sp["train_times"])
    fit = s.mask(times=inner["fit_times"])
    val = time_task(s, inner["fit_times"][-1], inner["val_times"][0])
    test = time_task(s, sp["train_times"][-1], sp["test_times"][0])
    return s, fit, inner["fit_times"], val, test, truth


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/phase1")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--search-seeds", type=int, default=2)
    ap.add_argument("--datasets", default="toys,grn")
    ap.add_argument("--max-configs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", default="forecast,interpolation", help="GRN tasks: forecast and/or interpolation")
    ap.add_argument("--interp-times", default="3.0", help="comma-separated inner timepoints held out (one fold each)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quick = args.quick
    base = TrainConfig(n_iters=1500 if quick else 8000, eval_every=150 if quick else 250, patience=4 if quick else 8,
                       batch_size=128 if quick else 256, depth=2 if quick else 3)
    grids = {"toys": QUICK_TOY_GRID if quick else TOY_GRID, "grn": QUICK_GRN_GRID if quick else GRN_GRID}
    protocol = EvalProtocol(n_samples=300 if quick else 1000, seed=args.seed)
    seeds = list(range(args.seeds))
    search_seeds = list(range(args.search_seeds))
    mlp_kwargs = {"hidden": 64, "n_iters": 300} if quick else {"hidden": 128, "n_iters": 2000}
    meta = {"version": __version__, "args": vars(args), "base_config": base.to_dict(), "grids": grids,
            "protocol": protocol.__dict__, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "best_configs": {}}
    tables, searches, modes = [], [], {}
    datasets = args.datasets.split(",")

    def run(name, s, fit, fit_times, val, test, mode, grid):
        modes[name] = mode
        best, tab = nested_grid_search(s, fit, fit_times, val, base, grid, search_seeds, protocol,
                                       args.max_configs, args.seed)
        tab.insert(0, "dataset", name)
        searches.append(tab)
        meta["best_configs"][name] = best.to_dict()
        print(f"[{name}] best config: {best.to_dict()}", flush=True)
        tables.append(run_forecast_benchmark(s, fit, fit_times, val, test, best, seeds, protocol, name,
                                             mlp_kwargs=mlp_kwargs, verbose=True))

    if "toys" in datasets:
        for toy in TOY_NAMES:
            s, fit, fit_times, val, test = toy_setup(toy, 300 if quick else 1500, args.seed)
            run(s.name, s, fit, fit_times, val, test, VERDICT_MODES["toys"], grids["toys"])
    if "grn" in datasets:
        s, fit, fit_times, val, test, truth = grn_setup(150 if quick else 500, args.seed, 20 if quick else 40,
                                                        6 if quick else 10)
        np.savez(out / "grn_truth.npz", **{k: np.asarray(v) for k, v in truth.items() if k != "schedules"})
        (out / "grn_schedules.json").write_text(json.dumps(truth["schedules"], indent=1), encoding="utf-8")
        tasks = args.tasks.split(",")
        if "forecast" in tasks:
            run(s.name, s, fit, fit_times, val, test, VERDICT_MODES["grn"], grids["grn"])
        if "interpolation" in tasks:
            for t_hold in [float(t) for t in args.interp_times.split(",")]:
                name = f"{s.name}_interp_t{t_hold:g}"
                modes[name] = VERDICT_MODES["grn_interp"]
                fold_tables = []
                for v_time in interpolation_val_folds(GRN_TIMES, t_hold):      # selection averaged over two folds
                    d = interpolation_design(GRN_TIMES, t_hold, v_time)
                    fit_i = s.mask(times=d["fit_times"])
                    val_i = time_task(s, d["v_prev"], d["val_time"])
                    _, tab = nested_grid_search(s, fit_i, d["fit_times"], val_i, base, grids["grn"], search_seeds,
                                                protocol, args.max_configs, args.seed)
                    tab.insert(0, "dataset", name)
                    tab.insert(1, "val_time", v_time)
                    fold_tables.append(tab)
                tab = pd.concat(fold_tables, ignore_index=True)
                searches.append(tab)
                best = select_best(tab, base)
                meta["best_configs"][name] = best.to_dict()
                print(f"[{name}] best config: {best.to_dict()}", flush=True)
                tables.append(run_interpolation_benchmark(s, GRN_TIMES, t_hold, best, seeds, protocol, name,
                                                          mlp_kwargs=mlp_kwargs, verbose=True))
    df = pd.concat(tables, ignore_index=True)
    df.to_csv(out / "results.csv", index=False)
    pd.concat(searches, ignore_index=True).to_csv(out / "search.csv", index=False)
    verdicts = write_report(df, out / "report.md", out / "verdict.json", modes=modes)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for ds, v in verdicts.items():
        print(f"{ds} ({v['mode']}): {'PASS' if v['pass'] else 'FAIL'} fm={v['fm_mean']:.3f} best_control={v['best_control']} "
              f"uses_time={v['uses_time']} null_q95={v['null_q95']:.3f} matches={v['matches_target']}", flush=True)
    return verdicts


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: `python -m pytest tests -q` (tutti) → verde; poi `python -m pytest tests -q -m slow` → 1 passed; poi `PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic --out outputs/quick --quick` e lettura di `outputs/quick/report.md`**

- [x] **Step 5: run completo `PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic --out outputs/phase1_v1` (≤ 2 h CPU), verdetto G1 e voce in `docs/superpowers/decisions.md`**

  Esito (2026-09-18): G1 **PASS**. Toy `outputs/phase1_v1_toys` 3/4 sanity (translation 0.0031 vs null 0.0027); GRN interpolazione t=3 `outputs/phase1_v2_grn_interp` (v0.1.7, spline + clock duration): FM 0.641 vs interpolazione delle medie 1.102, IC95 appaiato [−0.463, −0.459], vs tempo mescolato [−3.56, −3.39]; forecast terminale `outputs/phase1_v1_grn` diagnostico (FAIL per costruzione, v0.1.5). Report unito: `outputs/phase1_report/report.md`.

---

## Parte D — Self-review del piano

1. **Copertura della spec**: path lineare + target analitico + MLP condizionato + solver (Task 2, 4, 5); coupling indipendente e OT entro gruppi con coerenza dei metadati (Task 3); due tempi separati con test che `ContextEncoder.forward` non riceve `flow_time` (Task 5); split per unità biologiche e preprocessing solo sul training con test anti-leakage (Task 8, 13); controlli persistenza / media / mean-shift / lineare / MLP / FreeFM / tempo rimescolato (Task 11, 14); metriche distribuzionali con protocollo fissato e noise floor (Task 9); tre trasformazioni sintetiche note + condizione non rappresentabile rifiutata (Task 7, 13); checkpoint con schema, geni, preprocessing, configurazione, versione e rifiuto se incompatibile (Task 13); seed multipli, IC95, differenze appaiate, verdetto (Task 15); grid search annidata senza toccare il test (Task 14). Non coperto per scelta: AnnData e dati reali (Fase 2), crescita/sbilanciamento (Fase 2), GRN prior e gate (Fase 3) — la verità `m(t)` del simulatore è già salvata (`grn_truth.npz`) per la Fase 3.
2. **Placeholder**: nessun TBD/TODO; ogni step con codice ha il codice.
3. **Coerenza dei tipi**: `ForecastTask` e `train_flow(snapshots, fit_mask, fit_times, config, val_task, protocol)` sono usati con la stessa firma in Task 13, 14, 16; `EvalProtocol.evaluate(pred, target, seed_offset)` in Task 9 e 14; `TrainConfig.replace` in Task 12, 14, 16; `SnapshotSet(..., require_two_timepoints=False)` in Task 6 e 13.

## Parte E — Esecuzione

Esecuzione inline in questa sessione (skill `executing-plans`), perché il committente ha già detto di iniziare: i blocchi `# file:` vengono materializzati con `scripts/materialize_plan.py`, poi `pytest`, poi il run rapido, poi il run completo. Ogni deviazione dal codice del piano viene riportata nel piano stesso (il piano resta la fonte).

### Task 17: distributional fine-tune (v0.1.9)

The diagnosis in `scripts_dev/diag_ceiling.py` showed that the field reproduces its training path to
within 0.005 energy-distance units, so the path is the ceiling. This task adds a second stage that
matches the OBSERVED marginals directly, plus a windowed path supervision that stops the spline from
dictating the interior of a long gap. Selected on the validation folds (`scripts_dev/combo_scan.py`).

```python
# file: cfm_grn_v2/src/cfm_grn_v2/flows/marginal.py
"""Distributional (marginal-matching) objective (v0.1.9).

Conditional flow matching regresses the field on the velocity of a CONSTRUCTED path — a straight segment
between coupled cells, or a cubic spline through chained pseudo-trajectories. The learned field can therefore
never be better than that construction: if the path family's own marginal at an unobserved time is wrong, the
field faithfully reproduces the error (measured: spline marginal 0.636, trained model 0.641 on the synthetic
regulatory benchmark).

The objective here references no path. It asks the field to transport an observed snapshot onto the NEXT
observed snapshot and scores the result with a differentiable two-sample divergence, so the interior of the
trajectory is left to the field's own smoothness and to consistency with the other observed transitions.
Related in spirit to TrajectoryNet (Tong et al. 2020) and MIOFlow (Huguet et al. 2022), which match observed
marginals through a neural ODE; the implementation here is ours and is used as a fine-tuning stage on top of
the flow-matching pre-training, with the path loss retained as a regulariser.
"""
from __future__ import annotations

import torch

from .solver import integrate


def energy_distance_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable U-statistic energy distance 2 E|X - Y| - E|X - X'| - E|Y - Y'|.

    Expectation zero for identical distributions at any sample size (the same estimator as the evaluation
    metric, so the training signal and the reported number are the same quantity on different data)."""
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"expected two 2-D tensors with the same width, got {tuple(a.shape)} and {tuple(b.shape)}")
    n, m = a.shape[0], b.shape[0]
    if n < 2 or m < 2:
        raise ValueError("the U-statistic energy distance needs at least two samples on each side")
    ab = torch.cdist(a, b).mean()
    daa, dbb = torch.cdist(a, a), torch.cdist(b, b)
    aa = (daa.sum() - daa.diagonal().sum()) / (n * (n - 1))
    bb = (dbb.sum() - dbb.diagonal().sum()) / (m * (m - 1))
    return 2.0 * ab - aa - bb


def sliced_w2(a: torch.Tensor, b: torch.Tensor, n_projections: int = 64,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """Sliced squared 2-Wasserstein distance: mean over random one-dimensional projections of the squared
    difference between the sorted projected samples (needs equal sample sizes)."""
    if a.shape != b.shape:
        raise ValueError(f"sliced_w2 needs equal shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
    dirs = torch.randn(a.shape[1], int(n_projections), generator=generator, dtype=a.dtype, device=a.device)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-12)
    pa = torch.sort(a @ dirs, dim=0).values
    pb = torch.sort(b @ dirs, dim=0).values
    return ((pa - pb) ** 2).mean()


def divergence(pred: torch.Tensor, target: torch.Tensor, metric: str = "energy",
               generator: torch.Generator | None = None) -> torch.Tensor:
    if metric == "energy":
        return energy_distance_torch(pred, target)
    if metric == "sliced_w2":
        return sliced_w2(pred, target, generator=generator)
    raise ValueError(f"unknown marginal metric {metric!r} (expected 'energy' or 'sliced_w2')")


def marginal_loss(vector_field, x0: torch.Tensor, x1: torch.Tensor, context, horizon: float, n_steps: int = 10,
                  method: str = "rk4", metric: str = "energy",
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Divergence between the field's pushforward of x0 over `horizon` and the independently sampled x1.

    x0 and x1 are unpaired samples of the two snapshots: no coupling is computed and none is implied."""
    pred = integrate(vector_field, x0, context, int(n_steps), method, t_end=float(horizon))
    return divergence(pred, x1, metric, generator)
```

```python
# file: cfm_grn_v2/tests/test_marginal.py
import numpy as np
import pytest
import torch

from cfm_grn_v2.data.synthetic_toys import make_toy
from cfm_grn_v2.evaluation.metrics import EvalProtocol, energy_distance
from cfm_grn_v2.flows.marginal import divergence, energy_distance_torch, marginal_loss, sliced_w2
from cfm_grn_v2.models.vector_field import MLPVectorField
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.trainer import replicate_task, train_flow


def test_energy_distance_matches_the_numpy_estimator():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(60, 3)), rng.normal(size=(50, 3)) + 1.0
    got = float(energy_distance_torch(torch.as_tensor(a), torch.as_tensor(b)))
    assert got == pytest.approx(energy_distance(a, b), rel=1e-6, abs=1e-8)


def test_energy_distance_is_zero_for_two_halves_of_one_sample():
    x = torch.as_tensor(np.random.default_rng(1).normal(size=(400, 4)))
    assert abs(float(energy_distance_torch(x[:200], x[200:]))) < 0.05
    assert float(energy_distance_torch(x[:200], x[200:] + 2.0)) > 1.0


def test_energy_distance_rejects_degenerate_input():
    with pytest.raises(ValueError):
        energy_distance_torch(torch.zeros(1, 2), torch.zeros(5, 2))
    with pytest.raises(ValueError):
        energy_distance_torch(torch.zeros(5, 2), torch.zeros(5, 3))


def test_sliced_w2_zero_for_identical_and_positive_for_shifted():
    x = torch.as_tensor(np.random.default_rng(2).normal(size=(128, 5)))
    assert float(sliced_w2(x, x.clone())) == pytest.approx(0.0, abs=1e-9)
    assert float(sliced_w2(x, x + 1.0)) > 0.5
    with pytest.raises(ValueError):
        sliced_w2(x, x[:10])


def test_divergence_rejects_an_unknown_metric():
    x = torch.zeros(4, 2)
    with pytest.raises(ValueError):
        divergence(x, x, metric="wasserstein")


def test_marginal_loss_is_differentiable_and_learns_a_translation():
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    x0 = torch.as_tensor(rng.normal(size=(96, 2)), dtype=torch.float32)
    x1 = torch.as_tensor(rng.normal(size=(96, 2)) + np.array([2.0, -1.0]), dtype=torch.float32)
    field = MLPVectorField(2, 0, hidden=32, depth=2, use_time=False)
    opt = torch.optim.Adam(field.parameters(), lr=1e-2)
    first, loss = None, None
    for _ in range(80):
        loss = marginal_loss(field, x0, x1, None, horizon=1.0, n_steps=4)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in field.parameters())
        opt.step()
    assert torch.isfinite(loss) and loss.item() < 0.3 * first


def test_finetune_stage_runs_and_never_worsens_the_validation_score():
    s = make_toy("translation", n_per_group=200, seed=0)
    fit = s.mask(replicates=["r1"])
    val = replicate_task(s, "r2", 0.0, 1.0)
    proto = EvalProtocol(n_samples=150, seed=0)
    base = TrainConfig(hidden=32, depth=2, n_iters=200, eval_every=100, patience=3, clock="duration",
                       batch_size=64, seed=0)
    pre, _ = train_flow(s, fit, [0.0, 1.0], base, val, proto)
    tuned, hist = train_flow(s, fit, [0.0, 1.0], base.replace(marginal_iters=40, marginal_eval_every=10,
                                                             marginal_batch=64, marginal_steps=4), val, proto)
    assert any(r.get("stage") == "marginal" for r in hist)
    assert tuned.marginal_iter is not None
    assert tuned.best_val <= pre.best_val + 1e-9          # the stage keeps the pre-trained weights unless it wins
    assert np.isfinite(tuned.predict_standardized(np.zeros((5, 2), dtype=np.float32), "ctrl", 0.0, 1.0)).all()
```

### Task 18: linear path centre and gene detection filter (v0.1.10)

The diagnosis of the embryoid-body result (`scripts_dev/eb_diagnose.py`) separated two failures: the spline path's
centre (a cubic spline through the window means lands further from a held-out window than a straight line between
its neighbours) and the fit of the field. `SplinePath(center="linear")` keeps each trajectory's spline deviation and
moves the centre onto straight segments between the knot means (`TrainConfig.path_center`); `hidden=512` is the
second candidate. Both are chosen on validation folds with the test stage removed (`scripts_dev/select_config.py`).
The gene detection filter (`select_genes(min_detect=0.01)`, SCENIC default) lives in the Phase-2 plan.
