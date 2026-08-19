"""General utilities for reproducibility, data loading, and result saving."""

import torch
import numpy as np
import random
import scanpy as sc
from pathlib import Path


def set_seed(seed):
    """
    Set random seeds for reproducible experiments.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def load_data(data_path, n_top_genes=2000):
    """
    Load and preprocess a spatial transcriptomics dataset.

    Args:
        data_path: Directory containing one or more h5ad dataset files.
        n_top_genes: Number of highly variable genes to retain.

    Returns:
        dict: RNA matrix, spatial coordinates, and the AnnData object.
    """
    data_path = Path(data_path)
    
    h5ad_files = list(data_path.glob("*_prepared.h5ad"))
    if not h5ad_files:
        h5ad_files = list(data_path.glob("*.h5ad"))
    
    if not h5ad_files:
        raise FileNotFoundError(f"No h5ad file found in {data_path}")
    
    adata = sc.read_h5ad(h5ad_files[0])
    print(f"Loaded {h5ad_files[0].name}")
    print(f"Original shape: {adata.shape}")
    
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    if adata.n_vars > n_top_genes:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3')
        except ImportError:
            # Fall back to the Seurat flavor when scikit-misc is unavailable.
            print("  Using 'seurat' flavor for HVG selection (seurat_v3 requires scikit-misc)")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
        
        adata = adata[:, adata.var['highly_variable']].copy()
        print(f"Selected {n_top_genes} highly variable genes")
    
    if hasattr(adata.X, 'toarray'):
        x_rna = adata.X.toarray()
    else:
        x_rna = adata.X
    
    x_rna = torch.FloatTensor(x_rna)
    
    if 'spatial' in adata.obsm:
        pos = adata.obsm['spatial']
    elif 'X_spatial' in adata.obsm:
        pos = adata.obsm['X_spatial']
    else:
        raise KeyError("No spatial coordinates found in adata.obsm")
    
    pos = torch.FloatTensor(pos[:, :2])
    
    return {
        'x_rna': x_rna,
        'pos': pos,
        'adata': adata
    }


def save_results(adata, embeddings, cluster_assignments, probs, output_path):
    """
    Save SpatialRL outputs to an AnnData object.

    Args:
        adata: Original AnnData object.
        embeddings: Cell embeddings.
        cluster_assignments: Predicted cluster IDs.
        probs: Cluster probabilities.
        output_path: Output path.
    """
    adata.obsm['X_spatialrl'] = embeddings
    adata.obs['spatialrl_cluster'] = cluster_assignments.astype(str)
    adata.obsm['spatialrl_probs'] = probs
    
    adata.write_h5ad(output_path)
    print(f"Results saved to {output_path}")
