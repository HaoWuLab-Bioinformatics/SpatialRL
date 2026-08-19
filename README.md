# SpatialRL

SpatialRL, a spatial multi-modal representation learning framework for tissue domain discovery, which jointly models molecular heterogeneity and spatial organization across modalities.

## Project structure

```text
github_release/
├── model_full.py       # SpatialRL model architecture
├── train_full.py       # Training and prediction entry point
├── data_loader.py      # RNA/Protein and RNA/ATAC data loading
├── utils.py            # Data loading and result utilities
├── requirements.txt    # Python dependencies
└── data/
    ├── GSE198353_mmtv/
    ├── GSE263617_A1_LN/
    ├── GSE263617_A1_TNSL/
    ├── GSE263617_D1_LN/
    ├── GSE263617_D1_TNSL/
    └── Mouse_Brain/
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU execution, install a PyTorch build compatible with the local CUDA version. The code uses CUDA when available and otherwise falls back to the CPU.

## Train the RNA + Protein model

For example, using the SPOTS mouse spleen dataset:

```bash
python train_full.py \
  --data_path data/GSE198353_mmtv \
  --mode rna_protein \
  --n_clusters 8 \
  --output_dir results/GSE198353_mmtv
```

For Stereo-CITE-seq datasets, replace `--data_path` with the relevant `data/GSE263617_*` directory.

## Train the RNA + ATAC model

```bash
python train_full.py \
  --data_path data/Mouse_Brain \
  --mode rna_atac \
  --n_clusters 8 \
  --output_dir results/Mouse_Brain
```

Training results are written to the specified `output_dir` and do not modify the dataset directory.

## Dataset sources

The datasets in this release are from public resources:

- SPOTS mouse spleen: GEO accession `GSE198353`
- Stereo-CITE-seq human lymph node and tonsil: GEO accession `GSE263617`
- Spatial RNA-ATAC mouse brain: AtlasXplore
