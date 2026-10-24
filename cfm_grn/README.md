# CFM-GRN: Conditional Flow Matching con Prior Regolatorio e Maschera di Attività Contestuale

Questo repository implementa l'architettura di ricerca per il dottorato su:
> **"Learning Context-Specific Regulatory Activity for Generalizable Single-Cell Perturbation Prediction"**

---

## 1. Fondamenti Teorici

### Perché il Flow Matching per la genomica computazionale?
Nei modelli generativi tradizionali per perturbazioni a singola cellula:
1. **VAE / CVAE** (es. scGen, CPA): comprimono l'espressione in uno spazio latente e aggiungono vettori di perturbazione lineari. Limite: le dinamiche cellulari reali sono altamente non lineari e la varianza latente tende a collassare o a produrre campioni sfocati.
2. **Diffusion Models / DDPM**: campionano invertendo una catena markoviana stocastica lenta (centinaia di passi), e stimano lo score $\nabla_x \log p_t(x)$ che è instabile in presenza di sparsity elevata.
3. **Continuous Normalizing Flows (CNF)**: richiedono l'integrazione numerica dell'ODE durante l'addestramento, con costo computazionale proibitivo $O(\text{steps})$.

**Conditional Flow Matching (CFM)** risolve questi problemi:
- **Simulation-free**: durante il training non si integra alcuna ODE! Si fa una regressione analitica diretta su vettori velocità target.
- **Optimal Transport Coupling**: accoppia cellule di controllo $x_0$ e perturbate $x_1$ minimizzando la distanza di trasporto $\sum \|x_0 - x_1\|^2$, generando traiettorie rettilinee che si integrano in 10-20 step di Runge-Kutta.

### Le Equazioni Fondamentali

#### 1. Probability Path & Velocity Target
Dati $x_0 \sim p_{\text{ctrl}}$ e $x_1 \sim p_{\text{pert}}$, il percorso di interpolazione è:
$$x_t = (1 - t)x_0 + t x_1 + \sigma_{\min}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Il campo vettoriale condizionato obiettivo (target di regressione) è costante lungo la retta:
$$u_t(x | x_0, x_1) = \frac{d x_t}{dt} = x_1 - x_0$$

#### 2. Loss di Addestramento
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1), (x_0, x_1), \epsilon} \left\| v_\theta(x_t, t, c) - (x_1 - x_0) \right\|^2$$

#### 3. Vincolo Biologico: Grafo $A_{\text{prior}}$ e Maschera $M_\theta$
Per evitare che la rete neurale sia una "scatola nera" non interpretabile con connessioni dense arbitrarie, vincoliamo il passaggio di informazione alla rete di regolazione genica (GRN):
$$A_{\text{eff}}(x_t, t, c) = A_{\text{prior}} \odot M_\theta(x_t, t, c)$$

dove:
- $A_{\text{prior}} \in \{0, 1\}^{G \times G}$ è il prior noto (ChIP-seq, ENCODE, JASPAR).
- $M_\theta(x_t, t, c) \in [0, 1]^E$ è la maschera di attività appresa, che quantifica l'attivazione o repressione di ciascun asse regolatorio nel contesto corrente.
- La dinamica $v_\theta(x_t, t, c)$ è calcolata tramite strati di Message Passing orientati lungo $A_{\text{eff}}$.

---

## 2. Struttura del Codice

```
cfm_grn/
├── cfm/
│   ├── flow_matcher.py       # Engine matematico CFM e Optimal Transport mini-batch
│   └── solver.py             # Integratore ODE (Runge-Kutta 4 ed Eulero)
├── models/
│   ├── regulatory_mask.py    # Modulo della maschera di attività M_theta e A_prior sparso
│   ├── graph_vector_field.py # Architettura MPNN DirectedGeneConv per v_theta
│   └── baselines.py          # Modelli di controllo (MLP denso, Grafo statico, Persistenza)
├── data/
│   ├── synthetic_grn.py      # Simulatore biologico con cinetiche di Hill e Langevin SDE
│   └── dataset.py            # DataLoader con split per condizioni HELD-OUT
├── evaluate.py               # Energy Distance, Pearson correlation, Wasserstein-1
├── train.py                  # Script di addestramento e confronto scientifico
└── tests/
    ├── test_cfm.py           # Test unitari matematici su CFM e toy problem
    └── test_models.py        # Test unitari su GNN e maschera regolatoria
```

---

## 3. Come Eseguire i Test e gli Esperimenti

### Eseguire i test unitari:
```bash
python -m pytest tests/ -v
```

### Eseguire un esperimento di addestramento:
```bash
# Modello proposto (GNN guidato da GRN con Maschera di Attività Appresa)
python train.py --model graph_cfm --epochs 50

# Baseline 1: Grafo statico senza maschera appresa
python train.py --model static_graph --epochs 50

# Baseline 2: MLP denso senza grafo biologico
python train.py --model mlp --epochs 50
```
