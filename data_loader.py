"""Data loading utilities for multimodal spatial omics datasets."""

import torch
import numpy as np
import scanpy as sc
from pathlib import Path


def load_multimodal_data(data_path):
    """
    Load multimodal spatial transcriptomics data without preprocessing.

    Args:
        data_path: Dataset directory.

    Returns:
        dict: RNA, protein, spatial coordinates, labels, and AnnData objects.
    """
    data_path = Path(data_path)
    
    # Support both GEX-specific and generic h5ad filenames.
    gex_files = list(data_path.glob("*GEX*.h5ad"))
    if not gex_files:
        # Fall back to h5ad files that are not labeled as protein or ADT data.
        all_h5ad = list(data_path.glob("*.h5ad"))
        gex_files = [f for f in all_h5ad if "Protein" not in f.name and "ADT" not in f.name]
    
    if not gex_files:
        raise FileNotFoundError(f"No RNA/GEX data file found in {data_path}")
    
    gex_file = gex_files[0]
    adata_gex = sc.read_h5ad(gex_file)
    print(f"Loaded GEX: {gex_file.name}, shape: {adata_gex.shape}")
    
    print(f"  Using all genes: {adata_gex.shape}")
    
    if hasattr(adata_gex.X, 'toarray'):
        x_rna = torch.FloatTensor(adata_gex.X.toarray())
    else:
        x_rna = torch.FloatTensor(np.array(adata_gex.X))
    
    adt_files = list(data_path.glob("*ADT*.h5ad")) + list(data_path.glob("*Protein*.h5ad"))
    if adt_files:
        adata_adt = sc.read_h5ad(adt_files[0])
        print(f"Loaded ADT: {adt_files[0].name}, shape: {adata_adt.shape}")
        
        if hasattr(adata_adt.X, 'toarray'):
            x_protein = adata_adt.X.toarray()
        else:
            x_protein = adata_adt.X
        
        if np.isnan(x_protein).any() or np.isinf(x_protein).any():
            print(f"  WARNING: Protein data contains NaN or Inf, skipping protein modality")
            x_protein = None
        else:
            x_protein = torch.FloatTensor(x_protein)
    else:
        x_protein = None
        print("No ADT/Protein data found")
    
    if 'spatial' in adata_gex.obsm:
        pos = adata_gex.obsm['spatial']
    elif 'X_spatial' in adata_gex.obsm:
        pos = adata_gex.obsm['X_spatial']
    else:
        raise KeyError("No spatial coordinates found")
    
    pos = torch.FloatTensor(pos[:, :2])
    
    labels = None
    for label_key in ['cluster', 'celltype', 'annotation', 'leiden']:
        if label_key in adata_gex.obs:
            labels = adata_gex.obs[label_key].values
            if labels.dtype == object or isinstance(labels[0], str):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                labels = le.fit_transform(labels)
            print(f"Found labels in adata_gex.obs['{label_key}']")
            break
    
    if labels is None:
        print("No ground truth labels found")
        labels = np.zeros(adata_gex.n_obs)
    
    return {
        'x_rna': x_rna,
        'x_protein': x_protein,
        'pos': pos,
        'labels': labels,
        'adata_gex': adata_gex,
        'adata_adt': adata_adt if adt_files else None
    }


def load_rna_atac_data(data_path):
    """
    Load RNA + ATAC data, such as the Mouse_Brain dataset.

    The function reads the prepared matrices directly, matching the behavior
    of :func:`load_multimodal_data`.

    Returns:
        dict:
            x_rna    : (N, n_genes) RNA expression tensor.
            x_protein: (N, n_peaks) ATAC tensor stored in the protein slot.
            pos      : (N, 2) Spatial coordinate tensor.
            labels   : (N,) Reference labels, or zeros when unavailable.
            adata_gex: RNA AnnData object.
            adata_adt: ATAC AnnData object.
    """
    data_path = Path(data_path)

    rna_files = (list(data_path.glob("*RNA_prepared.h5ad")) or
                 list(data_path.glob("*RNA*.h5ad")))
    rna_files = [f for f in rna_files
                 if not any(k in f.name for k in ['ATAC', 'peaks', 'atac'])]
    if not rna_files:
        raise FileNotFoundError(f"No RNA file found in {data_path}")

    adata_rna = sc.read_h5ad(rna_files[0])
    print(f"Loaded RNA : {rna_files[0].name}, shape: {adata_rna.shape}")

    atac_files = (list(data_path.glob("*ATAC_prepared.h5ad")) or
                  list(data_path.glob("*peaks*.h5ad")) or
                  list(data_path.glob("*ATAC*.h5ad")))
    if not atac_files:
        raise FileNotFoundError(f"No ATAC file found in {data_path}")

    adata_atac = sc.read_h5ad(atac_files[0])
    print(f"Loaded ATAC: {atac_files[0].name}, shape: {adata_atac.shape}")

    print(f"  RNA  using all genes: {adata_rna.shape}")
    
    raw_rna = adata_rna.X
    x_rna = torch.FloatTensor(raw_rna.toarray() if hasattr(raw_rna, 'toarray') else np.array(raw_rna))
    print(f"  RNA  raw .X: {x_rna.shape}")

    if 'X_lsi' in adata_atac.obsm:
        lsi = adata_atac.obsm['X_lsi']
        n_lsi = lsi.shape[1]
        n_take = min(50, n_lsi - 1)   # Skip the first LSI component and use at most 50.
        x_atac = torch.FloatTensor(lsi[:, 1:n_take + 1])
        print(f"  ATAC X_lsi dims 1~{n_take}: {x_atac.shape}")
    else:
        raise KeyError("ATAC data does not contain obsm['X_lsi']; preprocess it first")

    for key in ['spatial', 'X_spatial']:
        if key in adata_rna.obsm:
            pos = torch.FloatTensor(adata_rna.obsm[key][:, :2])
            break
    else:
        raise KeyError("RNA data does not contain spatial coordinates")

    labels = None
    for label_key in ['celltype', 'annotation', 'layer', 'region', 'domain']:
        if label_key in adata_rna.obs:
            raw_labels = adata_rna.obs[label_key].values
            if raw_labels.dtype == object or str(raw_labels.dtype) == 'category':
                from sklearn.preprocessing import LabelEncoder
                labels = LabelEncoder().fit_transform(raw_labels)
            else:
                labels = raw_labels.astype(int)
            print(f"Found labels in adata_rna.obs['{label_key}'], n_classes={len(np.unique(labels))}")
            break

    if labels is None:
        print("No ground truth labels found (RNA_clusters/ATAC_clusters skipped as non-authoritative)")
        labels = np.zeros(adata_rna.n_obs, dtype=int)

    return {
        'x_rna'    : x_rna,
        'x_protein': x_atac,
        'pos'      : pos,
        'labels'   : labels,
        'adata_gex': adata_rna,
        'adata_adt': adata_atac,
    }
