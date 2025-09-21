import os
import json
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)

def load_structured_data(datapath):
    """Load structured data"""

    # Load edges.json (structured triplets)
    edges_path = os.path.join(os.path.dirname(__file__), datapath, 'edges.json')
    with open(edges_path, 'r', encoding='utf-8') as f:
        edges_data = json.load(f)

    # Load descriptions.json (ID -> name mapping)
    descriptions_path = os.path.join(os.path.dirname(__file__), datapath, 'descriptions.json')
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)

    return edges_data, descriptions

def create_entity_mappings(descriptions):
    """Create entity ID <-> index mappings for the GCN"""
    
    id_to_name = {}
    id_to_info = {}
    
    for entity in descriptions:
        entity_id = entity['id']
        entity_name = entity['properties']['name']
        entity_label = entity['properties']['label']
        
        id_to_name[entity_id] = entity_name
        id_to_info[entity_id] = {
            'name': entity_name,
            'label': entity_label,
            'type': entity_label
        }
    
    # Create mapping ID -> index for matrices
    unique_ids = sorted(set(id_to_name.keys()))
    id_to_idx = {entity_id: idx for idx, entity_id in enumerate(unique_ids)}
    idx_to_id = {idx: entity_id for entity_id, idx in id_to_idx.items()}

    logging.info(f"Unique entities: {len(unique_ids)}")
    logging.info(f"Mapping created: {len(id_to_idx)} entities")

    return id_to_name, id_to_info, id_to_idx, idx_to_id

def process_structured_triplets(edges_data, id_to_idx):
    """Process structured triplets for training"""
    
    processed_triplets = []
    relation_types = set()
    skipped_count = 0

    for edge in edges_data:
        # Verify that the entry is complete
        if not all(key in edge for key in ['source', 'target', 'relation']):
            skipped_count += 1
            continue

        source_id = edge['source']
        target_id = edge['target']
        relation = edge['relation']

        # Verify that IDs exist in the mapping
        if source_id not in id_to_idx or target_id not in id_to_idx:
            skipped_count += 1
            continue
        
        source_idx = id_to_idx[source_id]
        target_idx = id_to_idx[target_id]
        
        processed_triplets.append({
            'source_id': source_id,
            'target_id': target_id,
            'source_idx': source_idx,
            'target_idx': target_idx,
            'relation': relation
        })
        
        relation_types.add(relation)
    
    logging.info(f"Processed triplets: {len(processed_triplets)}")
    logging.info(f"Skipped triplets: {skipped_count}")
    logging.info(f"Relation types: {len(relation_types)}")

    return processed_triplets, list(relation_types)

def create_adjacency_matrices(processed_triplets, num_entities, relation_types):
    """Create adjacency matrices for GCN training"""
    
    logging.info(f"CREATING ADJACENCY MATRICES")

    # Global adjacency matrix
    adj_matrix = np.zeros((num_entities, num_entities), dtype=np.float32)
    
    # Matrices by relation type
    relation_matrices = {}
    relations = {}
    for rel_type in relation_types:
        relations[rel_type] = 0
    # Fill matrices
    for triplet in processed_triplets:
        source_idx = triplet['source_idx']
        target_idx = triplet['target_idx']
        relation = triplet['relation']
        
        # Global matrix (undirected)
        adj_matrix[source_idx, target_idx] = 1.0
        adj_matrix[target_idx, source_idx] = 1.0
        
        # Relation-specific matrix (directed)
        #relation_matrices[relation][source_idx, target_idx] = 1.0
        relations[relation] += 1

    
    # Statistics
    total_edges = np.sum(adj_matrix) / 2  # Divided by 2 for undirected
    density = total_edges / (num_entities * (num_entities - 1) / 2) * 100
    
    logging.info(f"Global matrix: {num_entities}x{num_entities}")
    logging.info(f"Total edges: {int(total_edges):,}")
    logging.info(f"Graph density: {density:.4f}%")

    # Statistics by relation
    logging.info(f"RELATION STATISTICS:")
    for rel_type in sorted(relation_types):
        #rel_edges = np.sum(relation_matrices[rel_type])
        rel_edges = relations[rel_type]
        logging.info(f"{rel_type:<25}: {int(rel_edges):,} edges")

    return adj_matrix, relation_matrices

def save_training_data(adj_matrix, relation_matrices, processed_triplets, 
                      id_to_name, id_to_info, id_to_idx, idx_to_id, relation_types):
    """Save prepared data for training"""

    logging.info(f"SAVING TRAINING DATA")

    output_dir = os.path.join(os.path.dirname(__file__), 'data/structured_training_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Adjacency matrices
    np.save(os.path.join(output_dir, 'adjacency_matrix.npy'), adj_matrix)
    logging.info("adjacency_matrix.npy saved")

    # Relation matrices
    """
    relation_dir = os.path.join(output_dir, 'relation_matrices')
    os.makedirs(relation_dir, exist_ok=True)

    for rel_type, matrix in relation_matrices.items():
        safe_name = rel_type.replace(' ', '_').replace('/', '_').lower()
        np.save(os.path.join(relation_dir, f'{safe_name}.npy'), matrix)

    logging.info(f"{len(relation_matrices)} relation matrices saved")
    """
    # Mappings and metadata
    mappings = {
        'id_to_name': id_to_name,
        'id_to_info': id_to_info,
        'id_to_idx': id_to_idx,
        'idx_to_id': {int(k): v for k, v in idx_to_id.items()},  # JSON compatible
        'relation_types': relation_types,
        'num_entities': len(id_to_idx),
        'num_triplets': len(processed_triplets)
    }
    
    with open(os.path.join(output_dir, 'mappings.json'), 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)

    logging.info("mappings.json saved")

    # Processed triplets
    with open(os.path.join(output_dir, 'processed_triplets.json'), 'w', encoding='utf-8') as f:
        json.dump(processed_triplets, f, indent=2, ensure_ascii=False)

    logging.info("processed_triplets.json saved")
    
    # Compatibility files
    create_compatibility_files(processed_triplets, id_to_name, output_dir)
    
    return output_dir

def create_compatibility_files(processed_triplets, id_to_name, output_dir):
    """Create compatibility files for previous pipeline"""
    
    # kg_structured_relationships.json format
    kg_compatible = []
    
    for triplet in processed_triplets:
        source_name = id_to_name[triplet['source_id']]
        target_name = id_to_name[triplet['target_id']]
        relation = triplet['relation']
        
        kg_compatible.append({
            'node_1': {'name': source_name},
            'node_2': {'name': target_name},
            'relationship': relation
        })
    
    with open(os.path.join(output_dir, 'kg_structured_relationships.json'), 'w', encoding='utf-8') as f:
        json.dump(kg_compatible, f, indent=2, ensure_ascii=False)

    logging.info("kg_structured_relationships.json (compatibility format) saved")

def main():
    """Main pipeline for preparation"""

    logging.info("PREPARING STRUCTURED DATA FOR GCN RE-TRAINING")
    logging.info("Using edges.json + descriptions.json")

    # Load data
    edges_data, descriptions = load_structured_data("data/llm_data")

    logging.info(f"DATA LOADED:")
    logging.info(f"Descriptions: {len(descriptions)} entities")
    logging.info(f"Edges: {len(edges_data)} triplets")

    # Create mappings
    id_to_name, id_to_info, id_to_idx, idx_to_id = create_entity_mappings(descriptions)
    
    # Process triplets
    processed_triplets, relation_types = process_structured_triplets(edges_data, id_to_idx)

    # Create matrices
    num_entities = len(id_to_idx)
    adj_matrix, relation_matrices = create_adjacency_matrices(processed_triplets, num_entities, relation_types)
    
    # Save
    output_dir = save_training_data(adj_matrix, relation_matrices, processed_triplets,
                                   id_to_name, id_to_info, id_to_idx, idx_to_id, relation_types)
    
    logging.info(f"DATA PREPARED SUCCESSFULLY!")
    logging.info(f"Output folder: {output_dir}")
    logging.info(f"Entities: {num_entities:,}")
    logging.info(f"Triplets: {len(processed_triplets):,}")
    logging.info(f"Relations: {len(relation_types)}")

if __name__ == "__main__":
    main()
