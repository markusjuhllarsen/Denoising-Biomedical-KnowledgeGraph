import os
import json
from difflib import SequenceMatcher
from collections import defaultdict
import csv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)

def load_id_mapping():
    descriptions_path = os.path.join(os.path.dirname(__file__), 'data/llm_data/descriptions.json')
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

def load_ground_truth_kg(file_path):
    ground_truth_positive = set()
    ground_truth_negative = set()
    iri_to_id = defaultdict(lambda: len(iri_to_id))
    invalid_rows = 0
    total_rows = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(reader):
            total_rows += 1
            if len(row) != 3:
                logging.info(f"Invalid row {i}: {row}")
                invalid_rows += 1
                continue
            source_iri = row[0].strip()
            target_iri = row[1].strip()
            source_id = iri_to_id[source_iri]
            target_id = iri_to_id[target_iri]
            if row[2] == '1':
                ground_truth_positive.add((source_id, target_id))
            elif row[2] == '0':
                ground_truth_negative.add((source_id, target_id))
            else:
                logging.info(f"Unknown label in row {i}: {row}")
                invalid_rows += 1
    logging.info(f"Total rows processed: {total_rows}")
    logging.info(f"Invalid rows (wrong format or unknown label): {invalid_rows}")
    logging.info(f"Ground truth positive: {len(ground_truth_positive)} triplets")
    logging.info(f"Ground truth negative: {len(ground_truth_negative)} triplets")
    return ground_truth_positive, ground_truth_negative, dict(iri_to_id)

def validate_removed_triplets_final():
    logging.info("FINAL VALIDATION - REMOVED TRIPLETS vs GROUND TRUTH")
    id_to_name, name_to_id = load_id_mapping()
    logging.info(f"Mapping loaded: {len(id_to_name)} entities")
    ground_truth_positive, ground_truth_negative, iri_to_id = load_ground_truth_kg('complete_groundtruth (1).tsv')
    logging.info(f"Ground truth loaded: {len(ground_truth_positive)} positive, {len(ground_truth_negative)} negative triplets")
    removed_file = os.path.join(os.path.dirname(__file__), 'kg_adaptive_removed_triplets_structured.json')
    with open(removed_file, 'r', encoding='utf-8') as f:
        removed_data = json.load(f)
    removed_triplets = removed_data['removed_triplets']
    logging.info(f"Removed triplets: {len(removed_triplets)}")
    logging.info(f"VALIDATION IN PROGRESS...")
    validation_results = {
        'false_positive_removal': 0,
        'correct_removal_positive': 0,
        'correct_removal_negative': 0,
        'no_mapping': 0
    }
    detailed_results = []
    for i, triplet in enumerate(removed_triplets):
        if i % 20 == 0:
            logging.info(f"Progress: {i}/{len(removed_triplets)} triplets validated...")
        head_name = triplet['head']
        tail_name = triplet['tail']
        relation = triplet['relation']
        head_id = find_entity_id(head_name, name_to_id, id_to_name)
        tail_id = find_entity_id(tail_name, name_to_id, id_to_name)
        validation_status = "no_mapping"
        evidence = []
        if head_id is not None and tail_id is not None:
            gt_triplet = (head_id, tail_id)
            if gt_triplet in ground_truth_positive:
                validation_status = "false_positive_removal"
                evidence.append(f"Triplet found in ground truth positive: {head_id} -> {tail_id} ({relation})")
            elif gt_triplet in ground_truth_negative:
                validation_status = "correct_removal_negative"
                evidence.append(f"Triplet found in ground truth negative: {head_id} -> {tail_id} ({relation})")
            else:
                validation_status = "correct_removal_positive"
                evidence.append(f"Triplet not in ground truth positive: {head_id} -> {tail_id} ({relation})")
        else:
            evidence.append(f"Mapping failed: {head_name} -> {head_id}, {tail_name} -> {tail_id}")
        validation_results[validation_status] += 1
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
    total = sum(v for k, v in validation_results.items() if k != 'no_mapping') + validation_results['no_mapping']
    false_positives = validation_results['false_positive_removal']
    correct_positive = validation_results['correct_removal_positive']
    correct_negative = validation_results['correct_removal_negative']
    no_mapping = validation_results['no_mapping']
    logging.info(f"FINAL VALIDATION RESULTS:")
    logging.info(f"Incorrect removals (false positives): {false_positives} ({false_positives/total*100:.1f}%)")
    logging.info(f"Correct removals (not in positives): {correct_positive} ({correct_positive/total*100:.1f}%)")
    logging.info(f"Correct removals (in negatives): {correct_negative} ({correct_negative/total*100:.1f}%)")
    logging.info(f"Mapping failed: {no_mapping} ({no_mapping/total*100:.1f}%)")
    logging.info(f"Total validated: {total}")
    category_stats = {}
    for result in detailed_results:
        category = result['triplet']['relation_category']
        status = result['validation_status']
        if category not in category_stats:
            category_stats[category] = {
                'total': 0,
                'false_positive_removal': 0,
                'correct_removal_positive': 0,
                'correct_removal_negative': 0,
                'no_mapping': 0
            }
        category_stats[category]['total'] += 1
        category_stats[category][status] += 1
    logging.info(f"CATEGORY ANALYSIS:")
    logging.info(f"{'Category':<25} {'Total':<8} {'False+':<8} {'Correct+':<8} {'Correct-':<8} {'NoMap':<8} {'Error rate'}")
    for category, stats in sorted(category_stats.items()):
        total_cat = stats['total']
        false_pos = stats['false_positive_removal']
        correct_pos = stats['correct_removal_positive']
        correct_neg = stats['correct_removal_negative']
        no_map = stats['no_mapping']
        error_rate = (false_pos / total_cat * 100) if total_cat > 0 else 0
        logging.info(f"{category:<25} {total_cat:<8} {false_pos:<8} {correct_pos:<8} {correct_neg:<8} {no_map:<8} {error_rate:<8.1f}%")
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
    logging.info(f"Final report saved: final_validation_report.json")
    return final_report

if __name__ == "__main__":
    final_report = validate_removed_triplets_final()