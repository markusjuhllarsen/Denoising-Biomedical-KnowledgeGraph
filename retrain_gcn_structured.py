try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
"""
RE-TRAINING GCN with structured data
Uses matrices created by prepare_structured_training.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)

# --- DrugBank Integration ---
def load_drugbank_dti_pairs(drugbank_path, mappings):
    """Load DrugBank DTI pairs and map to KG indices. Raises clear error if file is missing."""
    logging.info("LOADING DRUGBANK DTI PAIRS")
    if not os.path.isfile(drugbank_path):
        raise FileNotFoundError(f"DrugBank DTI file not found: {drugbank_path}.")
    dti_pairs = set()
    id_to_idx = {str(k): v for k, v in mappings['id_to_idx'].items()}
    id_to_name = mappings.get('id_to_name', {})
    name_to_id = {v.lower(): k for k, v in id_to_name.items()}
    # DrugBank file: tab-separated, columns: drug_id, target_id, [label]
    found_by_id = 0
    found_by_name = 0
    with open(drugbank_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            drug_id, target_id = parts[0], parts[1]
            # Try direct ID match (as string)
            if drug_id in id_to_idx and target_id in id_to_idx:
                dti_pairs.add((id_to_idx[drug_id], id_to_idx[target_id]))
                found_by_id += 1
                continue
            # Fallback: try name-based matching if available
            drug_name = id_to_name.get(drug_id) or id_to_name.get(str(drug_id))
            target_name = id_to_name.get(target_id) or id_to_name.get(str(target_id))
            if drug_name and target_name:
                drug_name_l = drug_name.lower()
                target_name_l = target_name.lower()
                kg_drug_id = name_to_id.get(drug_name_l)
                kg_target_id = name_to_id.get(target_name_l)
                if kg_drug_id in id_to_idx and kg_target_id in id_to_idx:
                    dti_pairs.add((id_to_idx[kg_drug_id], id_to_idx[kg_target_id]))
                    found_by_name += 1
    logging.info(f"DrugBank DTI pairs found in KG: {len(dti_pairs):,} (by ID: {found_by_id}, by name: {found_by_name})")
    return dti_pairs

class StructuredGCN(nn.Module):
    """GCN model adapted to structured data"""
    
    def __init__(self, num_features, hidden_dim=64, output_dim=32, dropout=0.5):
        super(StructuredGCN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.dropout = dropout
        self.num_features = num_features
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        
        return x

def load_structured_data():
    """Load prepared structured data"""
    logging.info("LOADING STRUCTURED DATA")
    data_dir = os.path.join(os.path.dirname(__file__), 'data/structured_training_data')
    # Load adjacency matrix
    adj_matrix = np.load(os.path.join(data_dir, 'adjacency_matrix.npy'))
    logging.info(f"Adjacency matrix: {adj_matrix.shape}")
    # Load mappings
    with open(os.path.join(data_dir, 'mappings.json'), 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    logging.info(f"Mappings: {mappings['num_entities']} entities")
    # Load triplets
    with open(os.path.join(data_dir, 'processed_triplets.json'), 'r', encoding='utf-8') as f:
        triplets = json.load(f)
    logging.info(f"Triplets: {len(triplets)} relations")
    return adj_matrix, mappings, triplets

def create_node_features(num_entities, mappings):
    """Create node features based on metadata"""
    logging.info(f"CREATING NODE FEATURES")
    # Enriched features: one-hot type, degree, text embedding, is_drug, base, name length, random
    id_to_info = mappings['id_to_info']
    id_to_idx = {int(k): v for k, v in mappings['id_to_idx'].items()}
    # Collect entity types
    entity_types = set()
    for entity_id, info in id_to_info.items():
        entity_types.add(info['label'])
    entity_types = sorted(list(entity_types))
    type_to_idx = {etype: idx for idx, etype in enumerate(entity_types)}
    logging.info(f"Entity types: {len(entity_types)}")

    # Compute node degrees if adjacency matrix is available in mappings
    adj = None
    if 'adjacency_matrix' in mappings:
        adj = np.array(mappings['adjacency_matrix'])
    else:
        # Try to load from file if present
        adj_path = os.path.join(os.path.dirname(__file__), 'structured_training_data', 'adjacency_matrix.npy')
        if os.path.exists(adj_path):
            adj = np.load(adj_path)
    if adj is not None:
        degrees = adj.sum(axis=1)
    else:
        degrees = np.zeros(num_entities)

    # Feature dim: one-hot types + [degree, mean_ascii, is_drug, base, name_len, random]
    feature_dim = len(entity_types) + 6
    node_features = np.zeros((num_entities, feature_dim), dtype=np.float32)
    for entity_id, info in id_to_info.items():
        entity_idx = id_to_idx[int(entity_id)]
        entity_type = info['label']
        # One-hot encoding of type
        type_idx = type_to_idx[entity_type]
        node_features[entity_idx, type_idx] = 1.0
        # Degree (normalized)
        node_features[entity_idx, len(entity_types)] = degrees[entity_idx] / max(1, degrees.max())
        # Mean ASCII of name (simple text embedding)
        name = info.get('name', '')
        if name:
            mean_ascii = np.mean([ord(c) for c in name]) / 128.0
        else:
            mean_ascii = 0.0
        node_features[entity_idx, len(entity_types) + 1] = mean_ascii
        # is_drug indicator (1 if type contains 'drug' or 'medicament')
        is_drug = 1.0 if ('drug' in entity_type.lower() or 'medicament' in entity_type.lower()) else 0.0
        node_features[entity_idx, len(entity_types) + 2] = is_drug
        # Base feature
        node_features[entity_idx, len(entity_types) + 3] = 1.0
        # Normalized name length
        node_features[entity_idx, len(entity_types) + 4] = float(len(name)) / 100.0
        # Random feature
        node_features[entity_idx, len(entity_types) + 5] = np.random.normal(0, 0.1)
    logging.info(f"Node features created: {node_features.shape}")
    return node_features

def adjacency_to_edge_index(adj_matrix):
    """Convert adjacency matrix to edge_index for PyTorch Geometric"""
    logging.info(f"CONVERTING TO EDGE_INDEX")
    # Convert to sparse matrix and extract indices
    sparse_adj = sp.coo_matrix(adj_matrix)
    # Edge index: [2, num_edges]
    edge_index = np.vstack([sparse_adj.row, sparse_adj.col])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    logging.info(f"Edge index: {edge_index.shape}")
    return edge_index

def create_training_data_with_drugbank(triplets, num_entities, mappings, drugbank_path, test_ratio=0.2):
    """Create training and test data using DrugBank for positive/negative labels"""
    logging.info(f"CREATING TRAINING DATA (DrugBank labels)")
    # Load DrugBank DTI pairs as positives
    dti_pairs = load_drugbank_dti_pairs(drugbank_path, mappings)
    if not dti_pairs:
        raise ValueError("No DrugBank DTI pairs found in KG. Check DrugBank file and mappings.")
    positive_pairs = np.array(list(dti_pairs))
    positive_labels = np.ones(len(positive_pairs))
    # All possible pairs in KG (for negative sampling)
    all_possible_pairs = set()
    for triplet in triplets:
        src, tgt = triplet['source_idx'], triplet['target_idx']
        all_possible_pairs.add((src, tgt))
        all_possible_pairs.add((tgt, src))
    # Negative pairs: sample from KG pairs not in DrugBank
    negative_pairs = []
    positive_set = set(dti_pairs)
    num_neg = len(positive_pairs)
    logging.info(f"Sampling {num_neg:,} negative pairs...")
    if TQDM_AVAILABLE:
        pbar = tqdm(total=num_neg, desc="Negative pairs")
    last_print = 0
    while len(negative_pairs) < num_neg:
        src = np.random.randint(0, num_entities)
        tgt = np.random.randint(0, num_entities)
        if src != tgt and (src, tgt) in all_possible_pairs and (src, tgt) not in positive_set:
            negative_pairs.append([src, tgt])
            if TQDM_AVAILABLE:
                pbar.update(1)
            elif len(negative_pairs) % 1000 == 0 and len(negative_pairs) != last_print:
                logging.info(f"{len(negative_pairs):,} / {num_neg:,} negatives sampled...")
                last_print = len(negative_pairs)
    if TQDM_AVAILABLE:
        pbar.close()
    negative_pairs = np.array(negative_pairs)
    negative_labels = np.zeros(len(negative_pairs))
    # Combine positives and negatives
    all_pairs = np.vstack([positive_pairs, negative_pairs])
    all_labels = np.concatenate([positive_labels, negative_labels])
    # Split train/test
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        all_pairs, all_labels, test_size=test_ratio, random_state=42, stratify=all_labels
    )
    logging.info(f"Positive pairs (DrugBank): {len(positive_pairs):,}")
    logging.info(f"Negative pairs: {len(negative_pairs):,}")
    logging.info(f"Training: {len(train_pairs):,}")
    logging.info(f"Test: {len(test_pairs):,}")
    return (train_pairs, train_labels), (test_pairs, test_labels)

def train_gcn_model(node_features, edge_index, train_data, test_data, epochs=200):
    """Train the GCN model"""
    logging.info(f"TRAINING GCN MODEL")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")
    # Prepare data
    x = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index = edge_index.to(device)
    train_pairs, train_labels = train_data
    test_pairs, test_labels = test_data
    train_pairs = torch.tensor(train_pairs, dtype=torch.long).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.float).to(device)
    test_pairs = torch.tensor(test_pairs, dtype=torch.long).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.float).to(device)
    # Initialize model
    model = StructuredGCN(
        num_features=node_features.shape[1],
        hidden_dim=128,
        output_dim=64,
        dropout=0.3
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    logging.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    # Training
    model.train()
    best_auc = 0
    best_model_state = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        embeddings = model(x, edge_index)
        # Compute scores for training pairs
        src_embeddings = embeddings[train_pairs[:, 0]]
        tgt_embeddings = embeddings[train_pairs[:, 1]]
        # Similarity score (dot product)
        scores = torch.sum(src_embeddings * tgt_embeddings, dim=1)
        # Loss
        loss = criterion(scores, train_labels)
        loss.backward()
        optimizer.step()
        # Evaluate every 20 epochs
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_embeddings = model(x, edge_index)
                test_src_embeddings = test_embeddings[test_pairs[:, 0]]
                test_tgt_embeddings = test_embeddings[test_pairs[:, 1]]
                test_scores = torch.sum(test_src_embeddings * test_tgt_embeddings, dim=1)
                # Metrics
                test_probs = torch.sigmoid(test_scores).cpu().numpy()
                test_labels_np = test_labels.cpu().numpy()
                auc = roc_auc_score(test_labels_np, test_probs)
                ap = average_precision_score(test_labels_np, test_probs)
                logging.info(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | AUC: {auc:.4f} | AP: {ap:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_model_state = model.state_dict().copy()
            model.train()
    # Load best model
    model.load_state_dict(best_model_state)
    logging.info(f"TRAINING COMPLETE")
    logging.info(f"Best AUC: {best_auc:.4f}")
    return model

def generate_triplet_scores(model, node_features, edge_index, triplets, mappings):
    """Generate scores for all triplets"""
    logging.info(f"GENERATING SCORES FOR ALL TRIPLETS")
    device = next(model.parameters()).device
    x = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index = edge_index.to(device)
    model.eval()
    scores = []
    with torch.no_grad():
        # Generate embeddings
        embeddings = model(x, edge_index)
        # Compute scores for each triplet
        for i, triplet in enumerate(triplets):
            if i % 1000 == 0:
                logging.info(f"Progress: {i}/{len(triplets)} triplets processed...")
            source_idx = triplet['source_idx']
            target_idx = triplet['target_idx']
            src_embedding = embeddings[source_idx]
            tgt_embedding = embeddings[target_idx]
            # Similarity score
            score = torch.sum(src_embedding * tgt_embedding).item()
            # Normalize with sigmoid
            score = torch.sigmoid(torch.tensor(score)).item()
            scores.append(score)
    scores = np.array(scores)
    logging.info(f"Scores generated: {len(scores):,}")
    logging.info(f"Distribution: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    return scores

def save_retrained_results(model, scores, mappings):
    """Save retraining results"""
    logging.info(f"SAVING RESULTS")
    output_dir = os.path.join(os.path.dirname(__file__), 'data/results/gcn')
    # Save model
    model_path = os.path.join(output_dir, 'best_gcn.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved: {model_path}")
    # Save scores
    scores_path = os.path.join(output_dir, 'triplet_scores.npy')
    np.save(scores_path, scores)
    logging.info(f"Scores saved: {scores_path}")
    # Metadata
    metadata = {
        'model_type': 'StructuredGCN',
        'num_entities': mappings['num_entities'],
        'num_triplets': mappings['num_triplets'],
        'score_range': {
            'min': float(scores.min()),
            'max': float(scores.max()),
            'mean': float(scores.mean()),
            'std': float(scores.std())
        },
        'training_complete': True
    }
    with open(os.path.join(output_dir, 'training_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logging.info(f"Metadata saved")
    return model_path, scores_path


def main():
    """Main retraining pipeline with DrugBank label integration"""
    logging.info(f"RE-TRAINING GCN WITH STRUCTURED DATA (DrugBank labels)")
    logging.info("Using matrices created by prepare_structured_training.py")
    logging.info("DrugBank DTI pairs used for positive/negative labels")
    # Load structured data
    adj_matrix, mappings, triplets = load_structured_data()
    # Create node features
    num_entities = mappings['num_entities']
    node_features = create_node_features(num_entities, mappings)
    # Convert to edge_index
    edge_index = adjacency_to_edge_index(adj_matrix)
    drugbank_path = os.path.join(os.path.dirname(__file__), 'data/drugbank_dti/train.tsv')
    # Create training data using DrugBank
    train_data, test_data = create_training_data_with_drugbank(triplets, num_entities, mappings, drugbank_path)
    # Train model
    model = train_gcn_model(node_features, edge_index, train_data, test_data, epochs=300)
    # Generate scores for all triplets
    scores = generate_triplet_scores(model, node_features, edge_index, triplets, mappings)
    # Save results
    model_path, scores_path = save_retrained_results(model, scores, mappings)
    logging.info(f"RE-TRAINING COMPLETE!")
    logging.info(f"Model: {model_path}")
    logging.info(f"Scores: {scores_path}")
    logging.info(f"Entities: {num_entities:,}")
    logging.info(f"Triplets: {len(triplets):,}")


if __name__ == "__main__":
    main()
