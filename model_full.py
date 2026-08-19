"""SpatialRL model components for multimodal spatial omics representation learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def knn_graph(pos, k, loop=False):
    """Build a k-nearest-neighbor graph from spatial coordinates."""
    n_nodes = pos.shape[0]
    dist = torch.cdist(pos, pos, p=2)
    if not loop:
        dist.fill_diagonal_(float('inf'))
    _, indices = torch.topk(dist, k, dim=1, largest=False)
    src = torch.arange(n_nodes, device=pos.device).unsqueeze(1).expand(-1, k)
    dst = indices
    edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)
    return edge_index


class ModalityEncoder(nn.Module):
    """Encode one molecular modality into the shared latent space."""
    def __init__(self, input_dim, d_model=128, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model)
        )
    
    def forward(self, x):
        return self.encoder(x)


class SpatialPositionEncoder(nn.Module):
    """Encode two-dimensional coordinates with sinusoidal features."""
    def __init__(self, d_pos=32, d_model=128):
        super().__init__()
        self.d_pos = d_pos
        self.projection = nn.Linear(d_pos * 2, d_model)
    
    def forward(self, pos):
        pos_norm = (pos - pos.min(0)[0]) / (pos.max(0)[0] - pos.min(0)[0] + 1e-8)
        pe_list = []
        for coord in [pos_norm[:, 0], pos_norm[:, 1]]:
            pe = torch.zeros(coord.shape[0], self.d_pos, device=coord.device)
            position = coord.unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_pos, 2, device=coord.device).float() * 
                                (-np.log(10000.0) / self.d_pos))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe_list.append(pe)
        pe_combined = torch.cat(pe_list, dim=1)
        return self.projection(pe_combined)



class GatedSpatialAttention(nn.Module):
    """Apply gated multi-head attention over spatial graph edges."""
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Learn how strongly each source node uses the spatial bias.
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, h, edge_index, spatial_weights):
        n_cells = h.shape[0]
        residual = h
        
        lambda_i = self.gate_net(h)
        
        Q = self.W_q(h).view(n_cells, self.n_heads, self.d_head)
        K = self.W_k(h).view(n_cells, self.n_heads, self.d_head)
        V = self.W_v(h).view(n_cells, self.n_heads, self.d_head)
        
        src, dst = edge_index[0], edge_index[1]
        q_i, k_j = Q[src], K[dst]
        
        content_scores = (q_i * k_j).sum(dim=-1) / np.sqrt(self.d_head)
        
        # Add a distance-based spatial bias modulated by the source-node gate.
        spatial_bias = torch.log(spatial_weights + 1e-8).unsqueeze(1)
        lambda_src = lambda_i[src]
        attn_scores = content_scores + lambda_src * spatial_bias
        
        attn_weights = self._edge_softmax(attn_scores, src, n_cells)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate messages by source node.
        v_j = V[dst]
        messages = attn_weights.unsqueeze(-1) * v_j
        z = torch.zeros(n_cells, self.n_heads, self.d_head, device=h.device)
        z.index_add_(0, src, messages)
        z = z.view(n_cells, self.d_model)
        z = self.W_o(z)
        
        return self.layer_norm(z + residual)
    
    def _edge_softmax(self, scores, src, n_nodes):
        max_scores = torch.full((n_nodes, scores.shape[1]), float('-inf'), device=scores.device)
        max_scores.index_reduce_(0, src, scores, 'amax', include_self=False)
        max_scores = max_scores[src]
        exp_scores = torch.exp(scores - max_scores)
        sum_exp = torch.zeros(n_nodes, scores.shape[1], device=scores.device)
        sum_exp.index_add_(0, src, exp_scores)
        sum_exp = sum_exp[src]
        return exp_scores / (sum_exp + 1e-8)



class MultiScaleSpatialModule(nn.Module):
    """Model spatial context at multiple neighborhood scales."""
    def __init__(self, d_model=128, n_heads=4, scales=[10, 20, 30], dropout=0.1):
        super().__init__()
        self.scales = scales
        self.attention_layers = nn.ModuleList([
            GatedSpatialAttention(d_model, n_heads, dropout) for _ in scales
        ])
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model, len(scales)),
            nn.Softmax(dim=1)
        )
    
    def forward(self, h, pos):
        n_cells = h.shape[0]
        scale_features = []
        
        for k, attn_layer in zip(self.scales, self.attention_layers):
            edge_index = knn_graph(pos, k=min(k, n_cells-1), loop=False)
            src, dst = edge_index[0], edge_index[1]
            dist = torch.norm(pos[src] - pos[dst], dim=1)
            sigma = torch.median(dist) * 1.5
            spatial_weights = torch.exp(-dist ** 2 / (2 * sigma ** 2))
            z_scale = attn_layer(h, edge_index, spatial_weights)
            scale_features.append(z_scale)
        
        # Adaptively combine the representations from all neighborhood scales.
        scale_weights = self.scale_fusion(h)
        z = torch.zeros_like(h)
        for i, z_scale in enumerate(scale_features):
            z += scale_weights[:, i:i+1] * z_scale
        
        return z



class CrossModalFusion(nn.Module):
    """Fuse two modality-specific representations with cross-attention."""
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()
        self.W_q_rna = nn.Linear(d_model, d_model)
        self.W_k_protein = nn.Linear(d_model, d_model)
        self.W_v_protein = nn.Linear(d_model, d_model)
        
        self.W_q_protein = nn.Linear(d_model, d_model)
        self.W_k_rna = nn.Linear(d_model, d_model)
        self.W_v_rna = nn.Linear(d_model, d_model)
        
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, z_rna, z_protein):
        # RNA queries protein features.
        Q_rna = self.W_q_rna(z_rna)
        K_protein = self.W_k_protein(z_protein)
        V_protein = self.W_v_protein(z_protein)
        
        attn_rp = torch.softmax(torch.mm(Q_rna, K_protein.t()) / np.sqrt(z_rna.shape[1]), dim=1)
        z_rna_enhanced = z_rna + self.dropout(torch.mm(attn_rp, V_protein))
        
        # Protein queries RNA features.
        Q_protein = self.W_q_protein(z_protein)
        K_rna = self.W_k_rna(z_rna)
        V_rna = self.W_v_rna(z_rna)
        
        attn_pr = torch.softmax(torch.mm(Q_protein, K_rna.t()) / np.sqrt(z_protein.shape[1]), dim=1)
        z_protein_enhanced = z_protein + self.dropout(torch.mm(attn_pr, V_rna))
        
        gate = self.gate_net(torch.cat([z_rna_enhanced, z_protein_enhanced], dim=1))
        z_fused = gate * z_rna_enhanced + (1 - gate) * z_protein_enhanced
        
        return self.layer_norm(z_fused), gate.squeeze(1)  # Shape: [N]; weight assigned to RNA.



class PrototypeClustering(nn.Module):
    """Perform soft clustering with learnable prototypes."""
    def __init__(self, d_model=128, n_clusters=7):
        super().__init__()
        self.n_clusters = n_clusters
        self.prototypes = nn.Parameter(torch.randn(n_clusters, d_model))
        nn.init.xavier_uniform_(self.prototypes)
    
    def forward(self, z, alpha=1.0):
        dist = torch.cdist(z, self.prototypes, p=2)
        q = torch.pow((1 + dist ** 2 / alpha), -(alpha + 1) / 2)
        q = q / q.sum(dim=1, keepdim=True)
        return q
    
    def target_distribution(self, q):
        p = q ** 2 / q.sum(dim=0, keepdim=True)
        p = p / p.sum(dim=1, keepdim=True)
        return p
    
    def reinitialize_dead_prototypes(self, z, q, min_count=20):
        """Reinitialize prototypes whose assignments fall below ``min_count``."""
        with torch.no_grad():
            cluster_assignments = q.argmax(dim=1)
            cluster_counts = torch.bincount(cluster_assignments, minlength=self.n_clusters)
            
            dead_prototypes = (cluster_counts < min_count).nonzero(as_tuple=True)[0]
            
            if len(dead_prototypes) > 0:
                # Reinitialize from a random latent sample with small perturbation.
                n_samples = z.shape[0]
                for dead_idx in dead_prototypes:
                    random_idx = torch.randint(0, n_samples, (1,), device=z.device)
                    noise = torch.randn_like(self.prototypes[dead_idx]) * 0.1
                    self.prototypes.data[dead_idx] = z[random_idx] + noise
                
                return len(dead_prototypes)
        return 0


class SpatialRL(nn.Module):
    """SpatialRL multimodal spatial transformer."""
    def __init__(
        self,
        n_genes,
        n_proteins,
        d_model=128,
        n_heads=4,
        n_clusters=8,
        scales=[5, 10, 15],
        dropout=0.1,
        use_protein=True
    ):
        super().__init__()
        self.use_protein = use_protein
        
        self.rna_encoder = ModalityEncoder(n_genes, d_model, dropout)
        if use_protein:
            self.protein_encoder = ModalityEncoder(n_proteins, d_model, dropout)
        self.spatial_encoder = SpatialPositionEncoder(d_pos=32, d_model=d_model)
        
        self.rna_spatial = MultiScaleSpatialModule(d_model, n_heads, scales, dropout)
        if use_protein:
            self.protein_spatial = MultiScaleSpatialModule(d_model, n_heads, scales, dropout)
            self.cross_modal_fusion = CrossModalFusion(d_model, dropout)
        
        self.clustering = PrototypeClustering(d_model, n_clusters)
    
    def forward(self, x_rna, pos, x_protein=None):
        h_rna = self.rna_encoder(x_rna)
        if self.use_protein and x_protein is not None:
            h_protein = self.protein_encoder(x_protein)
        
        s = self.spatial_encoder(pos)
        
        z_rna = self.rna_spatial(h_rna + s, pos)
        
        if self.use_protein and x_protein is not None:
            z_protein = self.protein_spatial(h_protein + s, pos)
            z, gate = self.cross_modal_fusion(z_rna, z_protein)
        else:
            z = z_rna
            gate = None
        
        q = self.clustering(z)
        
        return z, q, gate
