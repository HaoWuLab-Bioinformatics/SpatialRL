# SpatialRL

SpatialRL is a spatial multimodal representation-learning framework for tissue domain discovery. It jointly models molecular heterogeneity and spatial organization across RNA + Protein or RNA + ATAC modalities.

## What is included

This repository contains the model, training entry point, data loaders, and a compact demo dataset suitable for a quick installation and smoke test. The full source datasets under `data/` are retained for research use; the files under `data/demo/` are the small files intended for GitHub web upload.

```text
SpatialRL/
├── model_full.py                 # SpatialRL model architecture
├── train_full.py                 # Training and prediction entry point
├── data_loader.py                # RNA/Protein and RNA/ATAC data loading
├── utils.py                      # Reproducibility and result utilities
├── scripts/make_demo_data.py     # Reproducible demo-data preparation
├── requirements.txt              # Python dependencies
└── data/
    ├── demo/GSE198353_mmtv_demo/ # 256-spot RNA + Protein demo
    ├── GSE198353_mmtv/
    ├── GSE263617_A1_LN/
    ├── GSE263617_A1_TNSL/
    ├── GSE263617_D1_LN/
    ├── GSE263617_D1_TNSL/
    └── Mouse_Brain/
```

## System requirements

- Python 3.10 or newer
- Linux, macOS, or Windows
- CPU is supported; a CUDA-capable GPU is recommended for larger datasets
- At least 4 GB of RAM for the compact demo; larger source datasets require substantially more memory
- No non-standard hardware is required for the demo

The code uses CUDA when available and otherwise falls back to the CPU. CUDA users should install a PyTorch build compatible with their local CUDA version.

The verified demo environment was Python 3.10.16, PyTorch 2.5.1, Scanpy 1.11.4, AnnData 0.11.4, NumPy 2.0.1, and SciPy 1.15.1 on Windows. Other compatible versions may also work.

## Installation

Create and activate a clean Python environment, then install PyTorch followed by the remaining dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

Typical installation time is a few minutes on a normal desktop computer, excluding the download time for a CUDA-specific PyTorch build.

## Quick demo: RNA + Protein

The demo is a deterministic subset of the public SPOTS mouse spleen dataset (GEO accession `GSE198353`). It contains 256 spatial locations, 1,000 RNA features, and 32 Protein features. Each demo H5AD file is less than 1 MiB and can be uploaded through the GitHub web interface.

Run a short CPU/GPU smoke test from the repository root:

```bash
python train_full.py \
  --data_path data/demo/GSE198353_mmtv_demo \
  --mode rna_protein \
  --n_clusters 8 \
  --pretrain_epochs 5 \
  --finetune_epochs 10 \
  --seed 42 \
  --output_dir results/demo
```

Expected output:

```text
results/demo/spatialrl_result_seed42.h5ad
```

The result file contains the SpatialRL embedding in `obsm['X_spatialrl']`, cluster probabilities in `obsm['spatialrl_probs']`, cluster labels in `obs['spatialrl_cluster']`, and spatial coordinates in `obsm['spatial']`.

In the verified environment, this command took approximately 6.5 seconds with CUDA and 7.5 seconds with CUDA disabled. The exact runtime depends on the processor and whether CUDA is available. The command above is intentionally shorter than the default training schedule and is intended as an installation/demo check, not as a production analysis.

## Train on the bundled RNA + Protein datasets

For the SPOTS mouse spleen dataset:

```bash
python train_full.py \
  --data_path data/GSE198353_mmtv \
  --mode rna_protein \
  --n_clusters 8 \
  --output_dir results/GSE198353_mmtv
```

For a Stereo-CITE-seq dataset, replace `--data_path` with one of the `data/GSE263617_*` directories. The directory must contain one RNA/GEX H5AD file and one ADT/Protein H5AD file with matching observations and spatial coordinates.

## Train on the bundled RNA + ATAC dataset

```bash
python train_full.py \
  --data_path data/Mouse_Brain \
  --mode rna_atac \
  --n_clusters 8 \
  --output_dir results/Mouse_Brain
```

The RNA + ATAC loader expects the RNA H5AD to contain spatial coordinates and the ATAC H5AD to contain `obsm['X_lsi']`. Training results are written to the requested output directory and do not modify the dataset directory.

## Preparing a new demo subset

The checked-in demo can be regenerated from the prepared GSE198353 files with:

```bash
python scripts/make_demo_data.py
```

The preparation is deterministic: it selects 256 spatially distributed locations, prefers precomputed highly variable genes, retains 1,000 RNA features and all Protein features, aligns observations by barcode, removes bulky raw/image payloads, and writes compressed H5AD files. The demo is for software validation and is not intended to replace the complete source dataset for scientific analysis.

## Reproducibility

Use `--seed 42` to reproduce the demo initialization and training random state. For a manuscript-level reproduction, record the Python/PyTorch/Scanpy versions, operating system, device, full input dataset, training epochs, cluster count, and output directory. The demo preparation recipe is in `scripts/make_demo_data.py`.

## Data sources and attribution

- SPOTS mouse spleen: GEO accession `GSE198353`
- Stereo-CITE-seq human lymph node and tonsil: GEO accession `GSE263617`
- Spatial RNA-ATAC mouse brain: AtlasXplore

The demo is a derived subset of `GSE198353`; users should retain the original accession attribution and check the source repository's data-use terms before redistributing the full source data.

## License and repository

The SpatialRL source code in this repository is released under the MIT License; see [LICENSE](LICENSE). The public source datasets remain subject to their original terms and are not relicensed by this repository.

The open-source repository is this GitHub repository: [SpatialRL](.).

## Citation and method description

When using SpatialRL, cite the associated manuscript or preprint supplied by the authors. The implementation details are documented in the docstrings in `model_full.py`, `data_loader.py`, and `train_full.py`; the training pipeline is implemented in `train_full.py`.

## Software-submission checklist

This release provides source code, a small demo dataset, installation instructions, system and dependency requirements, demo instructions, expected output, reproducibility guidance, data provenance, and licensing information. Before formal submission, ask a colleague unfamiliar with the software to install it from a clean environment and run the demo from this README.
