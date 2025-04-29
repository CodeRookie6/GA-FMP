# MFTP_MFFP

MFTP_MFFP( multi-functional therapeutic peptide of multi-feature fusion prediction ) is a deep learning model designed for multifunctional therapeutic peptide prediction.

## Features

- Multi-feature fusion: sequence features + biological features + GNN-based graph features.
- Fuzzy representation using Gaussian membership.
- BiLSTM feature extraction.
- Hyperparameter optimization using Genetic Algorithm.
- Marginal focal dice loss function

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python main.py
```

Evaluate the model:

```bash
python test.py
```

## Requirements

- Python >= 3.7
- PyTorch
- torch_geometric
- pandas
- scikit-learn


