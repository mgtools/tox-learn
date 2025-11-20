# toxlearn

toxlearn is a unified toolkit for cross-chemical and cross-species
toxicity prediction. It includes preprocessing utilities, 2D and 3D
model training, and a full benchmarking framework for fingerprints,
species features, data-splitting strategies, and machine learning
models.

## Installation

Clone the repository and create the Conda environment: 

conda env create -f mordred_env.yml conda activate toxlearn

Project Structure

    toxlearn/
      ├─ mordred_env.yml      # Conda environment (RDKit, Mordred, PyTorch, etc.)
      ├─ data_prep/           # Data preprocessing, standardization, species features, split generation
      ├─ 3DmolTox/            # 3D model (3DMol-Tox) and training code
      ├─ config/              # YAML configs for 3d models setup
      ├─ benchmarking/        # Benchmarking code for fingerprints, species encodings, ML models, split types
      └─ README.md



Sample dataset with mordred fingerprint, using default taxonomy for species representation and split by CAS_ID group
      [Download dataset from Google Drive](https://drive.google.com/drive/folders/1D-paglmLlnHGQOCLe94TT2F7JtjYhmaP?usp=sharing)
