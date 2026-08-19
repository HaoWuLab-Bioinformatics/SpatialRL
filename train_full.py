"""Training utilities for SpatialRL pretraining and clustering finetuning."""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from model_full import SpatialRL, knn_graph
from data_loader import load_multimodal_data, load_rna_atac_data
from utils import set_seed


def contrastive_loss(z, edge_index, temperature=0.1):
    """Compute the graph-based contrastive loss."""
    src, dst = edge_index[0], edge_index[1]
    n_cells = z.shape[0]
    
    z_norm = F.normalize(z, p=2, dim=1)
    sim_matrix = torch.mm(z_norm, z_norm.t()) / temperature
    
    pos_mask = torch.zeros(n_cells, n_cells, device=z.device, dtype=torch.bool)
    pos_mask[src, dst] = True
    pos_mask[dst, src] = True
    
    diag_mask = torch.eye(n_cells, device=z.device, dtype=torch.bool)
    
    exp_sim = torch.exp(sim_matrix)
    pos_sim = (exp_sim * pos_mask).sum(dim=1)
    all_sim = (exp_sim * ~diag_mask).sum(dim=1)
    
    loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8).mean()
    return loss


def boundary_loss(z, edge_index, h_rna, margin=1.0):
    """Encourage latent separation across high-contrast RNA neighborhoods."""
    src, dst = edge_index[0], edge_index[1]
    
    feat_diff = torch.norm(h_rna[src] - h_rna[dst], dim=1)

    # Assign larger weights to neighbors with stronger feature contrast.
    weights = torch.sigmoid(feat_diff - feat_diff.median())

    emb_diff = torch.norm(z[src] - z[dst], dim=1)
    loss = (weights * torch.relu(margin - emb_diff)).mean()
    
    return loss


def kl_divergence(p, q):
    """Compute the mean KL divergence between two distributions."""
    return (p * torch.log(p / (q + 1e-8) + 1e-8)).sum(dim=1).mean()



def pretrain_stage(model, x_rna, x_protein, pos, n_epochs=100, lr=1e-3, device='cuda'):
    """Pretrain the model with graph-based contrastive and boundary losses."""
    print("\n" + "="*70)
    print("Contrastive Learning Training")
    print("="*70)
    
    model = model.to(device)
    x_rna = x_rna.to(device)
    if x_protein is not None:
        x_protein = x_protein.to(device)
    pos = pos.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    
    edge_index = knn_graph(pos, k=15, loop=False).to(device)
    
    losses = []
    iterator = tqdm(range(n_epochs), desc="Training")
    
    for epoch in iterator:
        model.train()
        
        z, q, _ = model(x_rna, pos, x_protein)

        loss_contrastive = contrastive_loss(z, edge_index, temperature=0.1)

        with torch.no_grad():
            h_rna = model.rna_encoder(x_rna)
        loss_boundary = boundary_loss(z, edge_index, h_rna, margin=1.0)
        
        loss = loss_contrastive + 0.001 * loss_boundary

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            iterator.set_postfix({
                'loss': f'{loss.item():.4f}',
                'contrastive': f'{loss_contrastive.item():.4f}',
                'boundary': f'{loss_boundary.item():.4f}'
            })
    
    print(f"Training completed. Final loss: {losses[-1]:.4f}")
    return losses


def finetune_stage(model, x_rna, x_protein, pos, n_epochs=200, lr=5e-4, 
                   beta_boundary=0.01, device='cuda'):
    """Finetune clustering assignments with boundary regularization."""
    print("\n" + "="*70)
    print("Stage 2: Clustering Finetuning with Boundary Constraint")
    print("="*70)
    
    model = model.to(device)
    x_rna = x_rna.to(device)
    if x_protein is not None:
        x_protein = x_protein.to(device)
    pos = pos.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    
    edge_index = knn_graph(pos, k=10, loop=False).to(device)
    
    with torch.no_grad():
        h_rna = model.rna_encoder(x_rna)
    
    losses = []
    iterator = tqdm(range(n_epochs), desc="Finetuning")
    
    for epoch in iterator:
        model.train()
        
        z, q, _ = model(x_rna, pos, x_protein)

        # Periodically revive prototypes with too few assignments.
        if epoch % 10 == 0:
            n_reinitialized = model.clustering.reinitialize_dead_prototypes(z, q, min_count=20)
            if n_reinitialized > 0:
                print(f"\n  Epoch {epoch}: Reinitialized {n_reinitialized} dead prototypes")
        
        with torch.no_grad():
            p = model.clustering.target_distribution(q)
        
        loss_cluster = kl_divergence(p, q)
        loss_boundary = boundary_loss(z, edge_index, h_rna, margin=1.0)

        cluster_probs = q.mean(dim=0)
        entropy = -(cluster_probs * torch.log(cluster_probs + 1e-8)).sum()
        max_entropy = np.log(model.clustering.n_clusters)
        loss_entropy = -entropy / max_entropy

        loss = loss_cluster + beta_boundary * loss_boundary + 1.0 * loss_entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                cluster_assignments = q.argmax(dim=1)
                n_active_clusters = len(torch.unique(cluster_assignments))
            
            iterator.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cluster': f'{loss_cluster.item():.4f}',
                'boundary': f'{loss_boundary.item():.4f}',
                'entropy': f'{loss_entropy.item():.4f}',
                'active': n_active_clusters
            })
    
    print(f"Finetuning completed. Final loss: {losses[-1]:.4f}")
    return losses



def train_full_model(
    model,
    x_rna,
    x_protein,
    pos,
    pretrain_epochs=50,
    finetune_epochs=200,
    pretrain_lr=1e-3,
    finetune_lr=5e-4,
    beta_boundary=0.001,
    device='cuda',
    seed=None
):
    """Run pretraining followed by clustering finetuning."""
    if seed is not None:
        set_seed(seed)

    pretrain_losses = pretrain_stage(
        model, x_rna, x_protein, pos,
        n_epochs=pretrain_epochs,
        lr=pretrain_lr,
        device=device
    )
    
    finetune_losses = finetune_stage(
        model, x_rna, x_protein, pos,
        n_epochs=finetune_epochs,
        lr=finetune_lr,
        beta_boundary=beta_boundary,
        device=device
    )
    
    return {
        'pretrain_losses': pretrain_losses,
        'finetune_losses': finetune_losses
    }


def predict(model, x_rna, x_protein, pos, device='cuda'):
    """Return latent embeddings, hard cluster assignments, and probabilities."""
    model.eval()
    model = model.to(device)
    x_rna = x_rna.to(device)
    if x_protein is not None:
        x_protein = x_protein.to(device)
    pos = pos.to(device)
    
    with torch.no_grad():
        z, q, _ = model(x_rna, pos, x_protein)
        cluster_assignments = q.argmax(dim=1)
    
    return z.cpu().numpy(), cluster_assignments.cpu().numpy(), q.cpu().numpy()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the SpatialRL model')
    parser.add_argument('--data_path', type=str, default='data/GSE198353_mmtv',
                        help='Path to data directory')
    parser.add_argument('--mode', type=str, default='rna_protein',
                        choices=['rna_protein', 'rna_atac'],
                        help='Data modality mode: rna_protein (default) or rna_atac (Mouse_Brain)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--pretrain_epochs', type=int, default=50,
                        help='Number of pretraining epochs')
    parser.add_argument('--finetune_epochs', type=int, default=200,
                        help='Number of finetuning epochs')
    parser.add_argument('--n_clusters', type=int, default=8,
                        help='Number of clusters')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("="*70)
    print("SpatialRL Training")
    print("="*70)
    print(f"Random seed: {args.seed}")
    print(f"Data path: {args.data_path}")
    
    print("\nLoading multimodal data...")
    if args.mode == 'rna_atac':
        data = load_rna_atac_data(args.data_path)
        print("Mode: RNA + ATAC (Mouse_Brain)")
    else:
        data = load_multimodal_data(args.data_path)
        print("Mode: RNA + Protein")
    
    x_rna = data['x_rna']
    x_protein = data['x_protein']
    pos = data['pos']
    
    n_genes = x_rna.shape[1]
    n_proteins = x_protein.shape[1] if x_protein is not None else 0
    
    modality2_name = "ATAC" if args.mode == 'rna_atac' else "Protein"

    print(f"\nData info:")
    print(f"  RNA shape: {x_rna.shape}")
    print(f"  {modality2_name} shape: {x_protein.shape if x_protein is not None else 'None'}")
    print(f"  Spatial coords: {pos.shape}")
    print(f"  Number of clusters: {args.n_clusters}")
    
    print(f"\nData diagnostics:")
    print(f"  RNA - min: {x_rna.min():.4f}, max: {x_rna.max():.4f}, mean: {x_rna.mean():.4f}")
    print(f"  RNA - NaN count: {torch.isnan(x_rna).sum().item()}")
    print(f"  RNA - Inf count: {torch.isinf(x_rna).sum().item()}")
    if x_protein is not None:
        print(f"  {modality2_name} - min: {x_protein.min():.4f}, max: {x_protein.max():.4f}, mean: {x_protein.mean():.4f}")
        print(f"  Protein - NaN count: {torch.isnan(x_protein).sum().item()}")
        print(f"  Protein - Inf count: {torch.isinf(x_protein).sum().item()}")
    print(f"  Spatial - min: {pos.min():.4f}, max: {pos.max():.4f}")
    print(f"  Spatial - range: {(pos.max() - pos.min()):.4f}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    model = SpatialRL(
        n_genes=n_genes,
        n_proteins=n_proteins,
        d_model=128,
        n_heads=4,
        n_clusters=args.n_clusters,  
        scales=[5, 10, 15],
        dropout=0.1,
        use_protein=(x_protein is not None)
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_results = train_full_model(
        model,
        x_rna,
        x_protein,
        pos,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        device=device,
        seed=args.seed
    )
    
    print("\nPredicting...")
    embeddings, cluster_assignments, probs = predict(model, x_rna, x_protein, pos, device)
    
    from pathlib import Path
    import anndata as ad

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    adata_result = ad.AnnData(X=data['adata_gex'].X)
    adata_result.obs_names = data['adata_gex'].obs_names.copy()
    adata_result.var_names = data['adata_gex'].var_names.copy()
    adata_result.obsm['X_spatialrl'] = embeddings
    adata_result.obs['spatialrl_cluster'] = cluster_assignments.astype(str)
    adata_result.obsm['spatialrl_probs'] = probs
    adata_result.obsm['spatial'] = pos.cpu().numpy()
    
    for key in data['adata_gex'].obs.keys():
        adata_result.obs[key] = data['adata_gex'].obs[key].values
    
    adata_result.var = data['adata_gex'].var
    adata_result.uns['spatialrl_params'] = {
        'n_genes': n_genes,
        'n_proteins': n_proteins,
        'd_model': 128,
        'n_heads': 4,
        'n_clusters': args.n_clusters,
        'scales': [5, 10, 15],
        'use_protein': (x_protein is not None),
        'mode': args.mode
    }
    
    output_file = output_dir / f'spatialrl_result_seed{args.seed}.h5ad'
    adata_result.write_h5ad(output_file)
    print(f"Results saved to {output_file}")
