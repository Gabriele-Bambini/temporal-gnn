r"""
Modulo di Prior Regolatorio e Maschera di Attivita' Contestuale (GRN Mask).

Biologia & Matematica:
La matrice di adiacenza del grafo regolatorio A_prior in {0, 1}^{G x G}
rappresenta interazioni fisiche note (es. TF j si lega al promotore del gene i).
Tuttavia, in un dato contesto cellulare x_t e sotto una specifica perturbazione P:
non tutti i legami sono funzionalmente attivi contemporaneamente.

Definiamo l'adiacenza efficace (Effective Adjacency):
    A_eff(x_t, t, P) = A_prior \odot M_theta(x_t, t, P)

dove M_theta in [0, 1]^E assegna un peso continuo di attivita' o inibizione a ciascun
arco noto in A_prior, condizionato dallo stato della cellula e dalla perturbazione.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class SparseAdjacencyPrior:
    """
    Contenitore e gestore dell'adiacenza biologica a priori A_prior in formato sparso (edge_index).
    """

    def __init__(self, num_genes: int, edge_index: torch.Tensor, gene_names: Optional[list] = None):
        """
        Args:
            num_genes: Numero totale di geni G.
            edge_index: Tensore di indici degli archi [2, E], dove edge_index[0] = sorgente (TF),
                        edge_index[1] = bersaglio (target).
            gene_names: Lista opzionale dei nomi dei geni (es. ['TP53', 'MYC', ...]).
        """
        self.num_genes = num_genes
        self.edge_index = edge_index
        self.gene_names = gene_names or [f"Gene_{i}" for i in range(num_genes)]
        self.num_edges = edge_index.shape[1]

    @classmethod
    def from_dense_adjacency(cls, adj: torch.Tensor, gene_names: Optional[list] = None) -> "SparseAdjacencyPrior":
        """Crea il prior sparso a partire da una matrice quadrata densa [G, G]."""
        num_genes = adj.shape[0]
        # Estrae le coordinate degli elementi non nulli
        edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous()
        return cls(num_genes=num_genes, edge_index=edge_index, gene_names=gene_names)

    def to(self, device: torch.device) -> "SparseAdjacencyPrior":
        """Sposta gli indici degli archi sul device specificato."""
        self.edge_index = self.edge_index.to(device)
        return self


class RegulatoryActivityMask(nn.Module):
    """
    Rete neurale che predice la maschera di attivita' M_theta in [0, 1]^E per ciascun arco biologico.
    
    Architettura:
    1. Encoder cellulare: comprime lo stato di espressione x_t [B, G], il tempo t [B, 1]
       e la perturbazione c [B, C] in un embedding di contesto cellulare h_context [B, d_ctx].
    2. Edge Scoring Network: combina l'embedding del TF sorgente, del gene bersaglio e il contesto
       per calcolare un logit di attivita' s_{j -> i} per ciascun arco in edge_index.
    3. Attivazione Sigmoid (o clampata) per vincolare il peso in [0, 1].
    """

    def __init__(
        self,
        num_genes: int,
        edge_index: torch.Tensor,
        perturbation_dim: int = 0,
        gene_embedding_dim: int = 32,
        context_dim: int = 64,
        hidden_dim: int = 64,
        pert_mode: str = "per_edge",
    ):
        """
        Args:
            num_genes: Numero di geni G.
            edge_index: Tensore degli indici degli archi [2, E].
            perturbation_dim: Dimensione del vettore di condizione/perturbazione (0 se nessuna).
            gene_embedding_dim: Dimensione degli embedding latenti per ciascun gene.
            context_dim: Dimensione dell'embedding globale della cellula.
            hidden_dim: Dimensione degli strati nascosti dell'MLP.
            pert_mode: How a per-gene perturbation vector reaches the edge scorer.
                "per_edge" (default): each edge sees two scalars, pert[src] and pert[dst],
                through weights shared by all edges -- so a gene never perturbed in
                training is handled by the same weights as the ones that were.
                "dense": the whole [G] vector goes through a Linear layer (original
                behaviour). The column for a held-out gene never receives a non-zero
                input in training, so it stays at random init and injects noise into
                every edge weight at test time. Kept for ablation only.
                Falls back to "dense" when perturbation_dim != num_genes.
        """
        super().__init__()
        self.num_genes = num_genes
        # Registra edge_index come buffer persistente (non trainable)
        self.register_buffer("edge_index", edge_index)
        self.num_edges = edge_index.shape[1]
        self.per_edge_pert = (
            pert_mode == "per_edge" and perturbation_dim == num_genes and perturbation_dim > 0
        )

        # Embedding specifici per i singoli geni
        self.gene_embeddings = nn.Embedding(num_genes, gene_embedding_dim)

        # Encoder dello stato cellulare: [x (G) + t (1) (+ pert (C) only in dense mode)]
        in_dim_ctx = num_genes + 1 + (0 if self.per_edge_pert else perturbation_dim)
        self.cell_encoder = nn.Sequential(
            nn.Linear(in_dim_ctx, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
            nn.SiLU(),
        )

        # Rete di scoring per arco: [h_context, emb_src, emb_dst (, pert_src, pert_dst)] -> 1
        edge_in_dim = context_dim + 2 * gene_embedding_dim + (2 if self.per_edge_pert else 0)
        self.edge_scorer = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        perturbation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcola i pesi della maschera di attivita' M_theta per ogni cellula nel batch e per ogni arco.
        
        Args:
            x: Espressione genica corrente [B, G].
            t: Tempo normalizzato [B, 1].
            perturbation: Vettore di condizione/perturbazione [B, C] (opzionale).
            
        Returns:
            mask: Pesi degli archi in [0, 1], con forma [B, E].
        """
        batch_size = x.shape[0]

        # 1. Costruzione input contesto cellulare
        ctx_inputs = [x, t]
        if perturbation is not None and not self.per_edge_pert:
            ctx_inputs.append(perturbation)
        ctx_cat = torch.cat(ctx_inputs, dim=-1)

        # Embedding di contesto globale per cellula: [B, context_dim]
        h_cell = self.cell_encoder(ctx_cat)

        # 2. Recupero embedding dei geni sorgente e bersaglio per tutti gli archi
        src_indices = self.edge_index[0]  # [E]
        dst_indices = self.edge_index[1]  # [E]

        src_emb = self.gene_embeddings(src_indices)  # [E, d_gene]
        dst_emb = self.gene_embeddings(dst_indices)  # [E, d_gene]
        edge_static_feat = torch.cat([src_emb, dst_emb], dim=-1)  # [E, 2 * d_gene]

        # Espansione per il batch
        # h_cell: [B, 1, context_dim] espanso a [B, E, context_dim]
        h_cell_expanded = h_cell.unsqueeze(1).expand(batch_size, self.num_edges, -1)
        # edge_static: [1, E, 2 * d_gene] espanso a [B, E, 2 * d_gene]
        edge_static_expanded = edge_static_feat.unsqueeze(0).expand(batch_size, self.num_edges, -1)

        # Concatenazione: [B, E, context_dim + 2 * d_gene]
        edge_parts = [h_cell_expanded, edge_static_expanded]
        if self.per_edge_pert:
            if perturbation is None:
                perturbation = torch.zeros(batch_size, self.num_genes, device=x.device, dtype=x.dtype)
            pert_src = perturbation[:, src_indices].unsqueeze(-1)  # [B, E, 1]
            pert_dst = perturbation[:, dst_indices].unsqueeze(-1)  # [B, E, 1]
            edge_parts += [pert_src, pert_dst]
        edge_inputs = torch.cat(edge_parts, dim=-1)

        # 3. Calcolo dei pesi continui tramite Sigmoid: [B, E]
        logits = self.edge_scorer(edge_inputs).squeeze(-1)
        mask = torch.sigmoid(logits)

        return mask

    def compute_regularization_loss(self, mask: torch.Tensor, l1_weight: float = 1e-3) -> torch.Tensor:
        """
        Calcola la perdita di regolarizzazione di sparsita' L1 sulla maschera appresa.
        Incoraggia il modello a mantenere attivi solo gli archi strettamente necessari.
        """
        return l1_weight * torch.mean(mask)
