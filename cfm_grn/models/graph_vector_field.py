"""
Graph Vector Field: Rete Neurale MPNN per la predizione del campo vettoriale v_theta(x, t, c).

Biologia & Matematica:
Un gene i varia la sua espressione secondo la dinamica di sintesi e degradazione:
    dx_i / dt = v_theta,i(x_t, t, c)
Nel nostro modello, la sintesi di i dipende SOLO dai suoi regolatori biologici j in N_in(i),
pesati dall'attivita' contestuale appresa M_theta(j -> i):
    m_i = sum_{j in N_in(i)} A_eff(j -> i) * W_msg * h_j
    h_i^{(l+1)} = MLP([h_i^{(l)}, m_i])
    v_i = Readout(h_i)
"""

from typing import Optional
import torch
import torch.nn as nn
from .regulatory_mask import RegulatoryActivityMask, SparseAdjacencyPrior


class DirectedGeneConv(nn.Module):
    """
    Strato di convoluzione su grafo diretto ottimizzato per batch di cellule.
    Esegue il message passing aggregando i messaggi dai TF sorgente ai geni bersaglio,
    pesati dalla matrice di adiacenza efficace A_eff = A_prior * M_theta.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Trasformazione lineare per i messaggi uscenti dai TF
        self.msg_linear = nn.Linear(in_channels, out_channels, bias=False)
        # MLP di aggiornamento dello stato del nodo: combina stato precedente e messaggi ricevuti
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: Embedding dei nodi genici [B, G, in_channels].
            edge_index: Indici degli archi [2, E].
            edge_weight: Pesi contestuali degli archi [B, E] (calcolati da RegulatoryActivityMask).
            
        Returns:
            h_next: Nuovi embedding per ciascun gene [B, G, out_channels].
        """
        B, G, _ = h.shape
        E = edge_index.shape[1]

        src_idx = edge_index[0]  # [E]
        dst_idx = edge_index[1]  # [E]

        # 1. Recupero feature dei nodi sorgente (TF): [B, E, in_channels]
        h_src = h[:, src_idx, :]

        # 2. Calcolo dei messaggi: [B, E, out_channels]
        raw_msg = self.msg_linear(h_src)

        # 3. Ponderazione dei messaggi con la maschera di attivita' regolatoria M_theta
        # edge_weight: [B, E, 1]
        weighted_msg = raw_msg * edge_weight.unsqueeze(-1)

        # 4. Aggregazione sparsa per ciascun gene bersaglio (dst_idx)
        # Inizializziamo il tensore aggregato: [B, G, out_channels]
        aggregated_msg = torch.zeros(B, G, self.out_channels, device=h.device, dtype=h.dtype)
        
        # index_add_ esegue la somma atomica di tutti i messaggi incidenti su dst_idx
        # Espandiamo dst_idx per il batch
        dst_expanded = dst_idx.unsqueeze(0).unsqueeze(-1).expand(B, E, self.out_channels)
        aggregated_msg.scatter_add_(dim=1, index=dst_expanded, src=weighted_msg)

        # 5. Aggiornamento dello stato: concatenazione [h_vecchio, msg_aggregato] -> MLP
        cat_feat = torch.cat([h, aggregated_msg], dim=-1)
        h_next = self.update_mlp(cat_feat)

        return h_next


class GraphVectorField(nn.Module):
    """
    Rete Neurale completa basata su Grafo Regolatorio per approssimare v_theta(x, t, condition).
    
    Flusso di calcolo:
    1. Calcola la maschera di attivita' contestuale M_theta tramite RegulatoryActivityMask.
    2. Proietta l'espressione scalare x_i, il tempo globale t e l'indicatore di perturbazione
       in uno spazio latente node-level [B, G, hidden_dim].
    3. Esegue L strati di DirectedGeneConv lungo il grafo biologico orientato.
    4. Proietta lo stato finale in uno scalare di velocita' per ciascun gene: v_i in R [B, G].
    """

    def __init__(
        self,
        num_genes: int,
        edge_index: torch.Tensor,
        perturbation_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_mask: bool = True,
        mask_pert_mode: str = "per_edge",
        edge_sign: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            num_genes: Numero totale di geni G.
            edge_index: Adiacenza biologica A_prior [2, E].
            perturbation_dim: Dimensione del vettore di perturbazione (es. G se one-hot CRISPR, o C).
            hidden_dim: Canali interni degli embedding e dei messaggi.
            num_layers: Numero di strati di message passing (profondita' di regolazione).
            use_mask: Se True, apprende e applica M_theta. Se False, usa pesi fissi 1.0 (baseline grafo statico).
            mask_pert_mode: forwarded to RegulatoryActivityMask ("per_edge" or "dense").
            edge_sign: optional [E] tensor of +1 (activation) / -1 (repression) for a
                signed prior. Multiplies every edge weight, so messages along repressive
                edges enter the target with flipped sign. None = unsigned prior.
        """
        super().__init__()
        self.num_genes = num_genes
        self.register_buffer("edge_index", edge_index)
        self.register_buffer(
            "edge_sign", edge_sign.float() if edge_sign is not None else None
        )
        self.perturbation_dim = perturbation_dim
        self.hidden_dim = hidden_dim
        self.use_mask = use_mask

        # Modulo per la maschera di attivita'
        if use_mask:
            self.mask_module = RegulatoryActivityMask(
                num_genes=num_genes,
                edge_index=edge_index,
                perturbation_dim=perturbation_dim,
                hidden_dim=hidden_dim,
                pert_mode=mask_pert_mode,
            )
        else:
            self.mask_module = None

        # Embedding d'ingresso per ogni gene: riceve [x_i (1), t (1), pert_i (se per-gene)]
        # Per consentire specificita', ogni gene ha anche un proprio ID embedding
        self.gene_id_emb = nn.Embedding(num_genes, hidden_dim)

        # Proiezione scalare di espressione e tempo
        self.input_proj = nn.Sequential(
            nn.Linear(2 + (1 if perturbation_dim > 0 else 0), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Strati di convoluzione genica su grafo orientato
        self.conv_layers = nn.ModuleList([
            DirectedGeneConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Readout finale: converte l'embedding di ciascun gene nella sua velocita' scalare dx_i / dt
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        Args:
            x: Espressione corrente [B, G].
            t: Tempo scalare [B, 1].
            perturbation: Vettore di condizione [B, C] (es. [B, G] con 1 sul gene perturbato).
            
        Returns:
            v: Vettore velocita' dx/dt predetto per ogni cellula e per ogni gene [B, G].
        """
        B, G = x.shape
        device = x.device

        # 1. Calcolo dei pesi degli archi (maschera di attivita' M_theta)
        edge_weight = self.compute_edge_weight(x, t, perturbation)  # [B, E]
        if self.edge_sign is not None:
            edge_weight = edge_weight * self.edge_sign.unsqueeze(0)

        # 2. Costruzione delle feature d'ingresso per ciascun gene [B, G, in_features]
        # x_feat: [B, G, 1]
        x_expanded = x.unsqueeze(-1)
        # t_feat: [B, G, 1] espanso per tutti i geni
        t_expanded = t.unsqueeze(1).expand(B, G, 1)

        feat_list = [x_expanded, t_expanded]
        if perturbation is not None:
            if perturbation.shape[-1] == G:
                # Se la perturbazione e' specificata per ciascun gene (es. one-hot target)
                pert_expanded = perturbation.unsqueeze(-1)
            else:
                # Se e' un vettore di condizione compresso, lo proiettiamo su G
                pert_expanded = torch.zeros(B, G, 1, device=device)
            feat_list.append(pert_expanded)

        node_inputs = torch.cat(feat_list, dim=-1)  # [B, G, in_dim]

        # Proiezione e somma con l'embedding di identita' del gene
        h = self.input_proj(node_inputs)  # [B, G, hidden_dim]
        gene_ids = torch.arange(G, device=device)
        h = h + self.gene_id_emb(gene_ids).unsqueeze(0)  # Somma broadcast sul batch

        # 3. Propagazione dei messaggi lungo la GRN modulata
        for conv in self.conv_layers:
            h = h + conv(h, self.edge_index, edge_weight)  # Connessione residuale

        # 4. Readout: predice la velocita' istantanea per ciascun gene
        v = self.readout(h).squeeze(-1)  # [B, G]

        return v

    def compute_edge_weight(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        perturbation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-cell edge weights [B, E]: contextual mask if enabled, else all ones."""
        if self.use_mask:
            return self.mask_module(x, t, perturbation)
        return torch.ones(x.shape[0], self.edge_index.shape[1], device=x.device)

    def get_latest_mask(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        perturbation: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Restituisce i pesi della maschera M_theta per ispezione biologica."""
        if self.use_mask:
            return self.mask_module(x, t, perturbation)
        return None
