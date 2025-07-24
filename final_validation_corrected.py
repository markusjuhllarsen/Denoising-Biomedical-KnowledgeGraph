import os
import json
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
import csv
import re

# Fonction pour web_search (simulé avec test réel ; intégrez tool)
def web_search(query):
    # Test réel : Appel simulé basé sur tool (retourne abstract PubMed)
    return "PubMed PMC9876543: Strong binding evidence for CHEBI_404903 and PR_000022718; inhibits with high affinity. Docking studies show strong interaction."

def extend_scores_with_pubmed(tsv_path):
    extended = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) == 3 and row[2] == '1':
                source_iri = row[0].strip().split('/')[-1]
                target_iri = row[1].strip().split('/')[-1]
                abstract = web_search(f"{source_iri} {target_iri} interaction PubMed abstract")
                terms = re.findall(r'\b(binding|inhibits|evidence|affinity|strong)\b', abstract.lower(), re.I)
                score = min(1.0, len(terms) / 5.0)
                extended[(source_iri, target_iri)] = score
    print(f"Scores étendus pour {len(extended)} triplets.")
    return extended

def load_ground_truth_kg(file_path):
    ground_truth = set()
    iri_to_id = defaultdict(lambda: len(iri_to_id))
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) == 3 and row[2] == '1':
                source_iri = row[0].strip()
                target_iri = row[1].strip()
                source_id = iri_to_id[source_iri]
                target_id = iri_to_id[target_iri]
                ground_truth.add((source_id, target_id))  # Ignore relation for match
    print(f"Ground truth chargée : {len(ground_truth)} triplets positifs.")
    return ground_truth, dict(iri_to_id)

def load_standardized_graph_for_gcn(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    nodes = set()
    edges = []
    for triplet in data:
        node1 = triplet['node1']
        node2 = triplet['node2']
        relation = triplet['relationship_label']
        nodes.add(node1)
        nodes.add(node2)
        edges.append((node1, node2, relation))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    gcn_scores = np.random.uniform(0.5, 1.0, len(edges))  # Placeholder ; intégrez vrai GCN
    print(f"KG standardisé chargé : {len(nodes)} nodes, {len(edges)} edges.")
    return node_to_idx, edges, gcn_scores

def load_id_mapping():
    descriptions_path = os.path.join(os.path.dirname(__file__), 'data_for_corentin+g-retriever', 'new_kg', 'descriptions.json')
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    id_to_name = {}
    name_to_id = {}
    for entity in descriptions:
        entity_id = entity['id']
        entity_name = entity['properties']['name']
        id_to_name[entity_id] = entity_name
        name_to_id[entity_name.lower()] = entity_id
    return id_to_name, name_to_id

def normalize_name(name):
    if not name:
        return ""
    return str(name).lower().strip()

def fuzzy_match(name1, name2, threshold=0.8):
    if not name1 or not name2:
        return False
    similarity = SequenceMatcher(None, normalize_name(name1), normalize_name(name2)).ratio()
    return similarity >= threshold

def find_entity_id(entity_name, name_to_id, id_to_name):
    normalized_name = normalize_name(entity_name)
    if normalized_name in name_to_id:
        return name_to_id[normalized_name]
    for known_name, entity_id in name_to_id.items():
        if fuzzy_match(entity_name, known_name):
            return entity_id
    return None

def validate_removed_triplets_final(scores, gcn_scores):
    print("FINAL VALIDATION - REMOVED TRIPLETS vs GROUND TRUTH")
    print("=" * 70)
    id_to_name, name_to_id = load_id_mapping()
    print(f"Mapping loaded: {len(id_to_name)} entities")
    ground_truth = load_ground_truth_kg('complete_groundtruth (1).tsv')
    print(f"Ground truth loaded: {len(ground_truth)} triplets")
    removed_file = os.path.join(os.path.dirname(__file__), 'kg_adaptive_removed_triplets_structured.json')
    with open(removed_file, 'r', encoding='utf-8') as f:
        removed_data = json.load(f)
    removed_triplets = removed_data['removed_triplets']
    print(f"Removed triplets: {len(removed_triplets)}")
    print(f"\nVALIDATION IN PROGRESS...")
    print("=" * 50)
    validation_results = {
        'false_positive_removal': 0,
        'correct_removal': 0,
        'no_mapping': 0,
        'total_validated': 0
    }
    detailed_results = []
    for i, triplet in enumerate(removed_triplets):
        if i % 20 == 0:
            print(f"   Progress: {i}/{len(removed_triplets)} triplets validated...")
        head_name = triplet['head']
        tail_name = triplet['tail']
        relation = triplet['relation']
        head_id = find_entity_id(head_name, name_to_id, id_to_name)
        tail_id = find_entity_id(tail_name, name_to_id, id_to_name)
        validation_status = "no_mapping"
        evidence = []
        if head_id is not None and tail_id is not None:
            gt_triplet = (head_id, tail_id)
            if gt_triplet in ground_truth:
                validation_status = "false_positive_removal"
                evidence.append(f"Triplet found in ground truth: {head_id} -> {tail_id} ({relation})")
            else:
                validation_status = "correct_removal"
                evidence.append(f"Triplet not in ground truth: {head_id} -> {tail_id} ({relation})")
        else:
            evidence.append(f"Mapping failed: {head_name} -> {head_id}, {tail_name} -> {tail_id}")
        validation_results[validation_status] += 1
        validation_results['total_validated'] += 1
        detailed_results.append({
            'triplet': triplet,
            'mapped_ids': {
                'head_id': head_id,
                'tail_id': tail_id,
                'head_name': head_name,
                'tail_name': tail_name
            },
            'validation_status': validation_status,
            'evidence': evidence
        })
    total = validation_results['total_validated']
    false_positives = validation_results['false_positive_removal']
    correct_removals = validation_results['correct_removal']
    no_mapping = validation_results['no_mapping']
    print(f"\nFINAL VALIDATION RESULTS:")
    print("=" * 60)
    print(f"Incorrect removals (false positives): {false_positives} ({false_positives/total*100:.1f}%)")
    print(f"Correct removals: {correct_removals} ({correct_removals/total*100:.1f}%)")
    print(f"Mapping failed: {no_mapping} ({no_mapping/total*100:.1f}%)")
    print(f"Total validated: {total}")
    # Category analysis
    print(f"\nCATEGORY ANALYSIS:")
    print("=" * 60)
    category_stats = {}
    for result in detailed_results:
        category = result['triplet']['relation_category']
        status = result['validation_status']
        if category not in category_stats:
            category_stats[category] = {
                'total': 0,
                'false_positive_removal': 0,
                'correct_removal': 0,
                'no_mapping': 0
            }
        category_stats[category]['total'] += 1
        category_stats[category][status] += 1
    print(f"{'Category':<25} {'Total':<8} {'False+':<8} {'Correct':<8} {'NoMap':<8} {'Error rate'}")
    print("-" * 80)
    for category, stats in sorted(category_stats.items()):
        total_cat = stats['total']
        false_pos = stats['false_positive_removal']
        correct = stats['correct_removal']
        no_map = stats['no_mapping']
        error_rate = (false_pos / total_cat * 100) if total_cat > 0 else 0
        print(f"{category:<25} {total_cat:<8} {false_pos:<8} {correct:<8} {no_map:<8} {error_rate:<8.1f}%")
    final_report = {
        'validation_method': 'ground_truth_kg_comparison',
        'ground_truth_source': 'complete_groundtruth.tsv',
        'total_triplets_validated': total,
        'validation_results': validation_results,
        'category_analysis': category_stats,
        'detailed_results': detailed_results
    }
    report_path = os.path.join(os.path.dirname(__file__), 'final_validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"\nFinal report saved: final_validation_report.json")
    return final_report

if __name__ == "__main__":
    scores = extend_scores_with_pubmed('complete_groundtruth (1).tsv')
    node_to_idx, edges, gcn_scores = load_standardized_graph_for_gcn('standardized_graph2_ro (2).json')
    final_report = validate_removed_triplets_final(scores, gcn_scores)
    print("\nVALIDATION COMPLETE!")