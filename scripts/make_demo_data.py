"""Create a compact, reproducible SpatialRL demo dataset.

The demo is a deterministic spatially distributed subset of the public
GSE198353 mouse spleen RNA + Protein data already included in this release.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "GSE198353_mmtv"
OUTPUT_DIR = ROOT / "data" / "demo" / "GSE198353_mmtv_demo"
N_CELLS = 256
N_GENES = 1000


def _spatially_distributed_indices(coords: np.ndarray, n_cells: int) -> np.ndarray:
    """Select deterministic points spread over the full tissue extent."""
    coords = np.asarray(coords, dtype=float)[:, :2]
    scaled = (coords - coords.min(axis=0)) / (
        coords.max(axis=0) - coords.min(axis=0) + 1e-12
    )
    side = int(np.ceil(np.sqrt(n_cells)))
    targets = np.linspace(0.0, 1.0, side + 2, dtype=float)[1:-1]
    chosen: list[int] = []
    available = np.ones(len(coords), dtype=bool)

    for tx in targets:
        for ty in targets:
            if len(chosen) >= n_cells:
                break
            distances = np.sum((scaled - np.array([tx, ty])) ** 2, axis=1)
            distances[~available] = np.inf
            index = int(np.argmin(distances))
            if np.isfinite(distances[index]):
                chosen.append(index)
                available[index] = False

    if len(chosen) < n_cells:
        remaining = np.flatnonzero(available)[: n_cells - len(chosen)]
        chosen.extend(int(i) for i in remaining)

    return np.asarray(chosen[:n_cells], dtype=int)


def _select_genes(adata: ad.AnnData, n_genes: int) -> np.ndarray:
    """Prefer precomputed highly variable genes, then fill deterministically."""
    n_genes = min(n_genes, adata.n_vars)
    if "highly_variable" in adata.var:
        hvgs = np.flatnonzero(np.asarray(adata.var["highly_variable"], dtype=bool))
        selected = hvgs[:n_genes]
        if len(selected) < n_genes:
            remaining = np.setdiff1d(np.arange(adata.n_vars), selected, assume_unique=False)
            selected = np.concatenate([selected, remaining[: n_genes - len(selected)]])
        return selected
    return np.arange(n_genes, dtype=int)


def _strip_unused_payload(adata: ad.AnnData) -> ad.AnnData:
    """Keep the fields consumed by SpatialRL and remove bulky raw payloads/images."""
    adata = adata.copy()
    adata.raw = None
    adata.layers.clear()
    adata.uns.clear()
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(np.asarray(adata.X, dtype=np.float32))
    return adata


def main() -> None:
    source_gex = SOURCE_DIR / "GSE198353_mmtv_GEX_prepared.h5ad"
    source_adt = SOURCE_DIR / "GSE198353_mmtv_ADT_prepared.h5ad"
    output_gex = OUTPUT_DIR / "GSE198353_mmtv_demo_GEX.h5ad"
    output_adt = OUTPUT_DIR / "GSE198353_mmtv_demo_ADT.h5ad"

    print(f"Reading RNA: {source_gex}")
    gex = sc.read_h5ad(source_gex)
    print(f"Reading Protein: {source_adt}")
    adt = sc.read_h5ad(source_adt)

    common = gex.obs_names.intersection(adt.obs_names)
    if len(common) < N_CELLS:
        raise ValueError(f"Only {len(common)} shared cells are available")
    gex = gex[common].copy()
    adt = adt[common].copy()

    if "spatial" not in gex.obsm and "X_spatial" not in gex.obsm:
        raise KeyError("The source RNA data has no spatial coordinates")
    spatial_key = "spatial" if "spatial" in gex.obsm else "X_spatial"
    cell_indices = _spatially_distributed_indices(gex.obsm[spatial_key], N_CELLS)
    gene_indices = _select_genes(gex, N_GENES)

    selected_names = gex.obs_names[cell_indices]
    gex_demo = gex[selected_names, gene_indices].copy()
    adt_demo = adt[selected_names].copy()
    if spatial_key != "spatial":
        gex_demo.obsm["spatial"] = np.asarray(gex_demo.obsm[spatial_key])

    gex_demo = _strip_unused_payload(gex_demo)
    adt_demo = _strip_unused_payload(adt_demo)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gex_demo.write_h5ad(output_gex, compression="gzip", compression_opts=4)
    adt_demo.write_h5ad(output_adt, compression="gzip", compression_opts=4)

    print(f"Wrote RNA: {output_gex} ({gex_demo.n_obs} cells x {gex_demo.n_vars} genes)")
    print(f"Wrote Protein: {output_adt} ({adt_demo.n_obs} cells x {adt_demo.n_vars} features)")


if __name__ == "__main__":
    main()
