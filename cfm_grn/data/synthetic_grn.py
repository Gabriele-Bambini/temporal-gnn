"""
Simulatore Sintetico di Dinamiche di Regolazione Genica (Hill Kinetics & ODE biologiche).

Questo modulo genera dataset sintetici controllati dove:
1. La topologia della rete di regolazione A_true e' nota a priori (ground truth).
2. L'espressione e' regolata da equazioni cinetiche non-lineari (Hill kinetics):
       dx_i / dt = alpha_i + sum_{j in Act(i)} w_{ji} * (x_j^n / (K^n + x_j^n)) - gamma_i * x_i
3. Le perturbazioni (es. knockout del gene k) silenziano l'espressione del target (CRISPRi).
4. Fornisce popolazioni di cellule di controllo e cellule perturbate campionando stati stazionari
   con rumore intrinseco biologico (Langevin SDE).
"""

from typing import Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset
import numpy as np


class SyntheticGRNSimulator:
    """
    Simulatore biochimico basato su cinetiche di Hill ed integrazione Runge-Kutta / Eulero-Maruyama.
    """

    def __init__(
        self,
        num_genes: int = 20,
        density: float = 0.2,
        hill_coefficient: float = 2.0,
        decay_rate: float = 1.0,
        basal_rate: float = 0.5,
        noise_level: float = 0.05,
        seed: int = 42,
        repression_frac: float = 0.0,
        context_frac: float = 0.0,
    ):
        """
        Args:
            num_genes: Numero di geni G.
            density: Densita' degli archi orientati (frazione di connessioni attive).
            hill_coefficient: Coefficiente n nella funzione di Hill (cooperativita').
            decay_rate: Costante di degradazione gamma_i.
            basal_rate: Tasso di trascrizione basale alpha_i.
            noise_level: Livello di rumore stocastico nella trascrizione.
            repression_frac: fraction of edges that repress instead of activate.
                A repressive edge j -> i contributes w * K^n / (K^n + x_j^n): large when
                the repressor is low, so knocking a repressor out RAISES its target.
            context_frac: fraction of edges gated by a third gene g: the edge only
                acts when g is expressed (contribution multiplied by hill(x_g)).
                This is edge activity that depends on cell state -- the only setting
                in which a contextual mask can beat learned static edge weights.
            Both default to 0, which reproduces the original simulator exactly
            (no extra random draws, same computation path).
        """
        self.num_genes = num_genes
        self.n = hill_coefficient
        self.decay_rate = decay_rate
        self.basal_rate = basal_rate
        self.noise_level = noise_level
        self.K = 1.0  # Costante di attivazione a meta' saturazione

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Genera matrice di adiacenza orientata casuale (DAG o grafo ciclico sparso)
        rand_matrix = np.random.rand(num_genes, num_genes)
        # Rimuove auto-anelli
        np.fill_diagonal(rand_matrix, 0.0)
        adj_binary = (rand_matrix < density).astype(np.float32)

        # Pesi di regolazione positivi (attivazione) tra 0.5 e 2.0
        weights = adj_binary * (0.5 + 1.5 * np.random.rand(num_genes, num_genes))

        self.W = torch.tensor(weights, dtype=torch.float32)
        self.edge_index = torch.nonzero(self.W > 0, as_tuple=False).t().contiguous()
        self.num_edges = self.edge_index.shape[1]

        # Optional regulatory realism. Random draws happen ONLY when enabled, so the
        # default configuration keeps the original RNG stream and data bit-for-bit.
        E = self.num_edges
        src, dst = self.edge_index[0], self.edge_index[1]
        self.edge_w = self.W[src, dst]
        self.edge_sign = torch.ones(E)
        self.edge_gate = torch.full((E,), -1, dtype=torch.long)
        self._edge_path = repression_frac > 0 or context_frac > 0
        if repression_frac > 0:
            rep = torch.as_tensor(np.random.rand(E) < repression_frac)
            self.edge_sign[rep] = -1.0
        if context_frac > 0:
            ctx = np.random.rand(E) < context_frac
            for e in np.where(ctx)[0]:
                s, d = int(src[e]), int(dst[e])
                choices = [g for g in range(num_genes) if g not in (s, d)]
                self.edge_gate[e] = int(np.random.choice(choices))

    def true_edge_signs(self) -> dict:
        """{(src, dst): +1 or -1} for every true edge."""
        return {
            (int(s), int(d)): float(sg)
            for s, d, sg in zip(self.edge_index[0], self.edge_index[1], self.edge_sign)
        }

    def signed_prior(self, fp_rate: float = 0.5, fn_rate: float = 0.2, seed: int = 0,
                     noisy: bool = True, sign_error: float = 0.0):
        """
        (edge_index, edge_sign) for a signed prior. True edges carry their true sign;
        spurious edges get a random sign (a wrong edge in a curated database comes with
        a sign too). The edge SET is identical to noisy_prior(fp_rate, fn_rate, seed),
        because signs are drawn only after the edges are chosen -- so signed and
        unsigned runs can be compared pair by pair.

        sign_error: fraction of the TRUE edges in the prior whose sign is flipped
            (curated databases get some modes of regulation wrong). 0.5 = signs of true
            edges are coin flips, i.e. no sign information left. Drawn from its own RNG,
            so sign_error=0 reproduces the previous output exactly.
        """
        ei = self.noisy_prior(fp_rate, fn_rate, seed) if noisy else self.edge_index
        truth = self.true_edge_signs()
        rng = np.random.RandomState(10_000 + seed)
        signs = [truth.get((int(s), int(d)), float(rng.choice([-1.0, 1.0])))
                 for s, d in zip(ei[0], ei[1])]
        if sign_error > 0:
            flip_rng = np.random.RandomState(20_000 + seed)
            for i, (s, d) in enumerate(zip(ei[0], ei[1])):
                if (int(s), int(d)) in truth and flip_rng.rand() < sign_error:
                    signs[i] = -signs[i]
        return ei, torch.tensor(signs, dtype=torch.float32)

    def simulate_transients(
        self,
        num_cells: int,
        knockout_gene: Optional[int] = None,
        record_steps=(5, 10, 20, 40, 100),
        num_steps: int = 100,
        dt: float = 0.05,
    ):
        """
        Cells observed WHILE relaxing from a random state, with the true drift at each
        recorded state: the synthetic analogue of RNA velocity on a population that is
        not at equilibrium. Returns (states [N, G], drift [N, G]).

        Consumes the RNG: call it after generate_benchmark_dataset so the evaluation
        populations stay identical to the snapshot-only runs.
        """
        x = torch.rand(num_cells, self.num_genes) * 2.0
        states, drifts = [], []
        for step in range(1, num_steps + 1):
            x = x + dt * self.drift(x, knockout_gene=knockout_gene) \
                + self.noise_level * np.sqrt(dt) * torch.randn_like(x)
            x = torch.clamp(x, min=0.0)
            if step in record_steps:
                states.append(x.clone())
                drifts.append(self.drift(x, knockout_gene=knockout_gene).clone())
        return torch.cat(states), torch.cat(drifts)

    def noisy_prior(self, fp_rate: float = 0.5, fn_rate: float = 0.2, seed: int = 0) -> torch.Tensor:
        """
        A deliberately imperfect prior [2, E'] for the models: each true edge is dropped
        with probability fn_rate, and round(fp_rate * E) spurious edges (not in the true
        GRN, no self-loops) are added. Signs and gates are not revealed -- a real prior
        (ChIP-seq / motif) is also just a binary graph.

        With the exact true graph as prior, a learned mask can only switch real edges
        off, i.e. it can only hurt; this is what gives it something to correct.
        """
        rng = np.random.RandomState(seed)
        true_edges = sorted(map(tuple, self.edge_index.t().tolist()))
        true_set = set(true_edges)
        kept = [e for e in true_edges if rng.rand() >= fn_rate]
        n_fp = int(round(fp_rate * len(true_edges)))
        max_fp = self.num_genes * (self.num_genes - 1) - len(true_set)
        if n_fp > max_fp:
            raise ValueError(f"fp_rate asks for {n_fp} spurious edges, only {max_fp} possible")
        fp = set()
        while len(fp) < n_fp:
            s, d = int(rng.randint(self.num_genes)), int(rng.randint(self.num_genes))
            if s != d and (s, d) not in true_set:
                fp.add((s, d))
        edges = kept + sorted(fp)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def drift(self, x: torch.Tensor, knockout_gene: Optional[int] = None) -> torch.Tensor:
        """
        Calcola il campo vettoriale biologico dx/dt (derivate temporali delle concentrazioni).

        Args:
            x: Espressioni geniche [B, G].
            knockout_gene: Indice del gene perturbato (se None, cellula di controllo).
        """
        x_clamped = torch.clamp(x, min=0.0)
        synthesis = self.synthesis(x, knockout_gene)

        # Degradazione lineare: - gamma * x
        degradation = self.decay_rate * x_clamped

        dxdt = synthesis - degradation

        if knockout_gene is not None:
            # Forza la derivata a spingere il gene verso zero rapidamente
            dxdt[:, knockout_gene] = - 5.0 * x_clamped[:, knockout_gene]

        return dxdt

    def synthesis(self, x: torch.Tensor, knockout_gene: Optional[int] = None) -> torch.Tensor:
        """Transcription rate alpha(x) = basal + regulatory input [B, G]; zero for a knocked-out gene."""
        # Funzione di Hill per l'attivazione: f(x) = x^n / (K^n + x^n)
        x_clamped = torch.clamp(x, min=0.0)
        Kn = self.K ** self.n

        if not self._edge_path:
            hill_act = (x_clamped ** self.n) / (Kn + x_clamped ** self.n + 1e-8)
            # Trascrizione regolata: W^T * hill_act -> [B, G]
            transcription = torch.matmul(hill_act, self.W)
        else:
            src, dst = self.edge_index[0], self.edge_index[1]
            xs_n = x_clamped[:, src] ** self.n                      # [B, E]
            act = xs_n / (Kn + xs_n + 1e-8)
            rep = Kn / (Kn + xs_n)
            f = torch.where(self.edge_sign > 0, act, rep)
            gated = self.edge_gate >= 0
            if bool(gated.any()):
                xg_n = x_clamped[:, self.edge_gate[gated]] ** self.n
                gate = torch.ones_like(f)
                gate[:, gated] = xg_n / (Kn + xg_n + 1e-8)
                f = f * gate
            contrib = f * self.edge_w                               # [B, E]
            transcription = torch.zeros_like(x_clamped).index_add_(1, dst, contrib)

        # Sintesi totale = basale + trascrizione
        synthesis = self.basal_rate + transcription

        # Se il gene k e' in knockout (CRISPRi), la sua sintesi e' azzerata
        if knockout_gene is not None:
            synthesis[:, knockout_gene] = 0.0

        return synthesis

    def simulate_transients_us(
        self,
        num_cells: int,
        splicing_rate: float = 2.0,
        record_steps=(5, 10, 20, 40, 100),
        num_steps: int = 100,
        dt: float = 0.05,
    ):
        """
        Two-species (unspliced u, spliced s) version of the same regulatory system, for
        methods that read velocity from u/s (RegVelo, scVelo):
            du/dt = alpha(s) - beta * u,   ds/dt = beta * u - gamma * s
        with alpha(.) = self.synthesis(.) (identical regulation), gamma = decay_rate.
        Its fixed point is s* = alpha(s*) / gamma, i.e. the same steady state as the
        one-species model, so predictions can be scored against the same evaluation
        populations. Cells are recorded while relaxing from a random state (control only).
        Returns (u [N, G], s [N, G]).
        """
        s = torch.rand(num_cells, self.num_genes) * 2.0
        u = torch.rand(num_cells, self.num_genes) * 2.0 / splicing_rate
        us, ss = [], []
        sq = self.noise_level * np.sqrt(dt)
        for step in range(1, num_steps + 1):
            a = self.synthesis(s)
            du = a - splicing_rate * u
            ds = splicing_rate * u - self.decay_rate * s
            u = torch.clamp(u + dt * du + sq * torch.randn_like(u), min=0.0)
            s = torch.clamp(s + dt * ds + sq * torch.randn_like(s), min=0.0)
            if step in record_steps:
                us.append(u.clone())
                ss.append(s.clone())
        return torch.cat(us), torch.cat(ss)

    def simulate_steady_state(
        self,
        num_cells: int = 500,
        knockout_gene: Optional[int] = None,
        num_steps: int = 100,
        dt: float = 0.05,
    ) -> torch.Tensor:
        """
        Simula una popolazione di cellule all'equilibrio stazionario tramite Eulero-Maruyama SDE.
        """
        # Stato iniziale casuale
        x = torch.rand(num_cells, self.num_genes) * 2.0

        for _ in range(num_steps):
            drift_term = self.drift(x, knockout_gene=knockout_gene)
            diffusion = self.noise_level * np.sqrt(dt) * torch.randn_like(x)
            x = x + dt * drift_term + diffusion
            x = torch.clamp(x, min=0.0)  # Le concentrazioni biologiche sono >= 0

        return x.detach()

    def generate_benchmark_dataset(
        self,
        cells_per_condition: int = 300,
        perturbed_genes: Optional[list] = None,
        num_perturbations: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """
        Genera un dataset completo con cellule di controllo e perturbazioni su geni selezionati.

        num_perturbations: how many top out-degree regulators to knock out when
            perturbed_genes is not given (default 3 = original behaviour).
        """
        if perturbed_genes is None:
            # Sceglie come target i geni con grado uscente piu' alto (TF principali)
            out_degrees = torch.sum(self.W > 0, dim=1).numpy()
            perturbed_genes = [int(g) for g in np.argsort(-out_degrees)[:num_perturbations]]

        data_dict = {}

        # 1. Popolazione di controllo (non perturbata)
        ctrl_cells = self.simulate_steady_state(num_cells=cells_per_condition, knockout_gene=None)
        data_dict["control"] = ctrl_cells
        data_dict["perturbed"] = {}

        # 2. Popolazioni perturbate per ciascun gene target
        for gene_idx in perturbed_genes:
            pert_cells = self.simulate_steady_state(
                num_cells=cells_per_condition,
                knockout_gene=gene_idx,
            )
            data_dict["perturbed"][gene_idx] = pert_cells

        data_dict["edge_index"] = self.edge_index
        data_dict["W_true"] = self.W
        data_dict["num_genes"] = self.num_genes
        data_dict["perturbed_genes"] = perturbed_genes

        return data_dict


class SyntheticDataset(Dataset):
    """
    PyTorch Dataset per coppie (controllo x_0, perturbato x_1, indicatore di perturbazione c).
    """

    def __init__(self, benchmark_data: Dict[str, torch.Tensor]):
        super().__init__()
        self.ctrl_cells = benchmark_data["control"]
        self.num_genes = benchmark_data["num_genes"]
        self.samples = []

        num_ctrl = self.ctrl_cells.shape[0]

        for gene_idx, pert_cells in benchmark_data["perturbed"].items():
            num_pert = pert_cells.shape[0]
            # Indicatore one-hot della perturbazione [G]
            pert_vec = torch.zeros(self.num_genes)
            pert_vec[gene_idx] = 1.0

            # Crea coppie appaiando o campionando
            for i in range(num_pert):
                ctrl_idx = i % num_ctrl
                x_0 = self.ctrl_cells[ctrl_idx]
                x_1 = pert_cells[i]
                self.samples.append((x_0, x_1, pert_vec, gene_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.samples[idx]
