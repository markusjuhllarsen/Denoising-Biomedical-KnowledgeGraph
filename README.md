# Biomedical Knowledge Graph Denoising Pipeline

## Overview
This project implements a pipeline for denoising biomedical knowledge graphs using Graph Convolutional Networks. The goal is to identify and remove noisy or incorrect relations while preserving valid biomedical interactions critical for applications like drug discovery and toxicology. 


## Key Features
- **Multi-Type Relation Detection**:
  - Supports network, causal, genomic, and pathway relations with adaptive thresholds for biomedical accuracy.
- **GCN-Powered Denoising**:
  - Uses PyTorch Geometric for anomaly detection and real-time triplet scoring.
- **Validation Layer**:
  - Integrates DrugBank DTI pairs for robust validation of drug-target interactions.
- **Full-Stack Pipeline**:
  - Modular Python scripts for data preparation, training, denoising, and validation, adaptable to other biomedical KGs.

## Tools & Technologies
| Category          | Technologies                          |
|-------------------|---------------------------------------|
| Simulation        | PyTorch, PyTorch Geometric            |
| AI/ML             | Scikit-Learn, NumPy, Pandas           |
| Languages         | Python 3.10.17                        |
| Optional Tools    | Tqdm (progress bars)                  |

## Installation
### Prerequisites
- **Python Version**: 3.10.17
- **Numpy Version**: 2.0.1
- **Pandas version**: 2.2.3
- **Torch version**: 2.5.1
- **sklearn version**: 1.6.1
- **torch_geometric version**: 2.6.1
- **Hardware**: GPU recommended for GCN training (CUDA support).

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/TarikRH/Denoising-Biomedical-KnowledgeGraph.git
   cd Denoising-Biomedical-KnowledgeGraph
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install numpy pandas torch torch-geometric scikit-learn tqdm
   ```


## How to Execute the Pipeline
Run the following scripts in order : 

1. **Prepare Structured Data**:
   ```bash
   python prepare_structured_training.py
   ```
   - **Input**: `data/llm_data/edges.json`, `data/llm_data/descriptions.json`
   - **Output**: `data/structured_training_data/` (`adjacency_matrix.npy`, `mappings.json`, `processed_triplets.json`, `kg_structured_relationships.json`, `relation_matrices/`)
   - **Description**: Generates structured data for GCN training.

2. **Retrain GCN Model**:
   ```bash
   python retrain_gcn_structured.py
   ```
   - **Input**: `data/structured_training_data/` (`adjacency_matrix.npy`, `mappings.json`, `processed_triplets.json`), `data/drugbank_dti/train.tsv`
   - **Output**: `data/structured_training_data/` (`best_structured_gcn.pth`, `kg_structured_scores.npy`, `kg_gcn_scores_structured.npy`, `training_metadata.json`)
   - **Description**: Trains the GCN model to produce scores that can be used in denoising.
  - **Further Details**: The node features used in the GCN model are one-hot (relation) types + [degree, mean_ascii, is_drug, base, name_len, random], resulting in a feature vector of length 187 for each node.  
  Source-target pairs are given a positive or negative label pased on presence in DrugBank DTI data.  
  GCN model is trained using these node features and graph structure to output a 64-diemnsional embedding of each node. The training is done using the DrugBank DTI labels with Binary Cross Entropy loss.  
  Based on these 64-dimensional node embeddings, similary scores between sources and target are calculated.

3. **Apply Adaptive Denoising**:
   ```bash
   python test_structured_denoising.py
   ```
   - **Input**: `warith/kg_llm_relationships_all.json`, `data/results/gcn/triplet_scores.npy`, `data/drugbank_dti/train.tsv`
   - **Output**: `kg_adaptive_removed_triplets_structured.json`
   - **Description**: Removes noisy triplets using relation-specific thresholds.

4. **Validate Removed Triplets**:
   ```bash
   python final_validation_corrected.py
   ```
   - **Input**: `kg_adaptive_removed_triplets_structured.json`, ground truth
   - **Output**: `final_validation_report.json`
   - **Description**: Validates removed triplets against the ground truth to assess accuracy.



## Model Details

The GCN model used in this pipeline is implemented in `retrain_gcn_structured.py` and is based on PyTorch Geometric's `GCNConv` layers. Below are the main architectural details:

- **Architecture:**
  - 3-layer Graph Convolutional Network (GCN)
  - Each layer: `GCNConv` (standard aggregation: sum/mean)
  - Hidden dimensions: configurable (default: 128)
  - Output dimension: configurable (default: 64)
  - Activation: ReLU after each layer except the last
  - Dropout: applied after each hidden layer (default: 0.3)

- **Forward Pass:**
  1. Input node features → GCNConv → ReLU → Dropout
  2. Hidden → GCNConv → ReLU → Dropout
  3. Hidden → GCNConv → Output

- **Loss Function:**
  - Binary cross-entropy (BCEWithLogitsLoss) for link prediction

- **Optimizer:**
  - Adam

- **Training:**
  - Supervised with positive/negative pairs 
  - Early stopping based on validation AUC 

**Reproducibility:**
- All model code is in `retrain_gcn_structured.py` (see class `StructuredGCN`).
- You can extract the model into a separate `model.py` for modularity if desired.

## Configuration and Tuning
- **In `retrain_gcn_structured.py`**:
  - `epochs` (default: 300)
  - `hidden_dim` (default: 128)
  - `dropout` (default: 0.3)



## Project Structure
```
Denoising-Biomedical-KnowledgeGraph/
├── data/
│   ├── llm_data/
│       ├── edges.json
│       ├── source.json
│       ├── target.json
│       └── descriptions.json
│   ├── drugbank_dti
│       ├── drug_feat_info.json
│       ├── target_feat_info.json
│       ├── test.tsv
│       ├── train.tsv
│       └── valid.tsv
│   └── structured_training_data/
│       ├── adjacency_matrix.npy
│       ├── mappings.json
│       ├── processed_triplets.json
│       ├── kg_structured_relationships.json
│       └── relation_matrices/
├── prepare_structured_training.py
├── retrain_gcn_structured.py
├── test_structured_denoising.py
├── final_validation_corrected.py
├── requirements.txt
└── README.md
```
