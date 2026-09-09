[README.md](https://github.com/user-attachments/files/32018673/README.md)
# GSE198353 demo subset

This directory contains a compact RNA + Protein demo derived from the public SPOTS mouse spleen dataset, GEO accession `GSE198353`.

## Contents

- `GSE198353_mmtv_demo_GEX.h5ad`: 256 spatial locations x 1,000 RNA features
- `GSE198353_mmtv_demo_ADT.h5ad`: the same 256 locations x 32 Protein features

The observation barcodes are aligned between the two files, and the RNA file contains spatial coordinates in `obsm['spatial']`. Bulky raw matrices, images, and unused payloads were removed so that the demo can be uploaded through the GitHub web interface.

This is a software demonstration subset, not a replacement for the complete source dataset and not intended for standalone biological conclusions. See the repository README and `scripts/make_demo_data.py` for the command and deterministic preparation recipe.
