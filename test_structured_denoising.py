"""
Adaptive denoising test using the new structured GCN scores
"""

import os
import json
import numpy as np

def load_denoising_results_structured():
    """Load the original KG and the new structured GCN scores."""
    # Load original KG
    kg_path = os.path.join(os.path.dirname(__file__), 'warith', 'kg_llm_relationships_all.json')
    with open(kg_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    # Load new structured GCN scores
    scores = np.load(os.path.join(os.path.dirname(__file__), 'kg_gcn_scores_structured.npy'))
    # Extract triplets (head, relation, tail)
    triplets = []
    for entry in kg_data:
        h = entry["node_1"]["name"]
        t = entry["node_2"]["name"]
        r = entry["relationship"]
        triplets.append((h, r, t))
    return triplets, scores, kg_data

def classify_relation_type(relation):
    """Classify a relation according to its biomedical type and assign a threshold."""
    relation_lower = relation.lower()
    # Thresholds differentiated by biomedical criticality
    relation_thresholds = {
        'drug_target_critical': {
            'relations': ['treats', 'inhibits', 'activates', 'binds to', 'targets'],
            'threshold': 0.10,
            'rationale': 'Drug-target relations are crucial for drug discovery'
        },
        'genomic_fundamental': {
            'relations': ['encodes', 'expressed in', 'transcribes', 'regulates'],
            'threshold': 0.15,
            'rationale': 'Genomic relations are well-established and should be preserved'
        },
        'pathway_standard': {
            'relations': ['involved in', 'participates in', 'catalyzes', 'metabolizes'],
            'threshold': 0.20,
            'rationale': 'Pathway relations have moderate confidence requirements'
        },
        'anatomical_standard': {
            'relations': ['located in', 'part of', 'contains', 'found in'],
            'threshold': 0.20,
            'rationale': 'Anatomical relations are generally well-documented'
        },
        'causal_strict': {
            'relations': ['causes', 'leads to', 'results in', 'induces'],
            'threshold': 0.25,
            'rationale': 'Causal relations require higher confidence due to complexity'
        },
        'administrative_strict': {
            'relations': ['administered via', 'prescribed for', 'dosed as'],
            'threshold': 0.30,
            'rationale': 'Administrative relations can be more variable'
        },
        'rare_specialized': {
            'relations': ['phosphorylates', 'ubiquitinates', 'methylates', 'acetylates'],
            'threshold': 0.05,
            'rationale': 'Rare biochemical relations should be preserved even if infrequent'
        }
    }
    # Search for a matching pattern
    for category, info in relation_thresholds.items():
        for pattern in info['relations']:
            if pattern in relation_lower:
                return category, info['threshold'], info['rationale']
    # Default: standard conservative threshold for unclassified relations
    return 'default', 0.15, 'Standard threshold for unclassified relations (conservative)'

def adaptive_denoising_structured(triplets, scores):
    """Apply adaptive denoising using the new GCN scores and relation-specific thresholds."""
    keep_mask = np.zeros(len(triplets), dtype=bool)
    for i, (h, r, t) in enumerate(triplets):
        category, threshold, rationale = classify_relation_type(r)
        if scores[i] >= threshold:
            keep_mask[i] = True
    return keep_mask

def test_structured_denoising():
    """Run adaptive denoising using the new GCN scores and print summary statistics."""
    print("ADAPTIVE DENOISING TEST - STRUCTURED GCN SCORES")
    print("=" * 70)
    # Load data with new scores
    triplets, scores, kg_data = load_denoising_results_structured()
    print(f"Loaded data:")
    print(f"   Triplets: {len(triplets):,}")
    print(f"   Scores (new): {len(scores):,}")
    print(f"   Score distribution: min={np.min(scores):.4f}, max={np.max(scores):.4f}, mean={np.mean(scores):.4f}")
    # Apply adaptive denoising
    keep_mask = adaptive_denoising_structured(triplets, scores)
    # Compute statistics
    total_triplets = len(triplets)
    kept_count = int(keep_mask.sum())
    removed_count = total_triplets - kept_count
    print(f"\nResults with new scores:")
    print(f"   Triplets kept: {kept_count:,} ({kept_count/total_triplets*100:.1f}%)")
    print(f"   Triplets removed: {removed_count:,} ({removed_count/total_triplets*100:.1f}%)")
    # Comparison with previous results (static reference)
    print(f"\nComparison:")
    print(f"   BEFORE (original scores): 145 removed (1.5%)")
    print(f"   AFTER (structured scores): {removed_count} removed ({removed_count/total_triplets*100:.1f}%)")
    # Extract removed triplets for validation
    removed_triplets = []
    for i, (h, r, t) in enumerate(triplets):
        if not keep_mask[i]:
            score = float(scores[i])
            category, threshold, rationale = classify_relation_type(r)
            removed_triplets.append({
                'head': h,
                'relation': r,
                'tail': t,
                'score': score,
                'threshold_used': threshold,
                'relation_category': category,
                'rationale': rationale,
                'confidence': 'high' if score > 0.6 else 'medium' if score > 0.4 else 'low'
            })
    # Save removed triplets for validation
    if removed_triplets:
        output = {
            'method': 'adaptive_denoising_structured',
            'description': 'Triplets removed using new structured GCN scores',
            'generation_info': {
                'total_original_triplets': total_triplets,
                'triplets_removed': removed_count,
                'removal_percentage': round(removed_count/total_triplets*100, 2)
            },
            'removed_triplets': removed_triplets
        }
        output_path = os.path.join(os.path.dirname(__file__), 'kg_adaptive_removed_triplets_structured.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nRemoved triplets saved:")
        print(f"   File: kg_adaptive_removed_triplets_structured.json")
        print(f"   Triplets: {len(removed_triplets):,}")
    else:
        print(f"\nNo triplet removed.")
        print("   All triplets have sufficiently high scores.")
    # Score analysis by category
    print(f"\nScore analysis by category:")
    print("-" * 60)
    category_stats = {}
    for i, (h, r, t) in enumerate(triplets):
        category, threshold, _ = classify_relation_type(r)
        score = scores[i]
        if category not in category_stats:
            category_stats[category] = {
                'scores': [],
                'threshold': threshold,
                'above_threshold': 0,
                'below_threshold': 0
            }
        category_stats[category]['scores'].append(score)
        if score >= threshold:
            category_stats[category]['above_threshold'] += 1
        else:
            category_stats[category]['below_threshold'] += 1
    for category, stats in sorted(category_stats.items()):
        total = len(stats['scores'])
        above = stats['above_threshold']
        below = stats['below_threshold']
        mean_score = np.mean(stats['scores'])
        threshold = stats['threshold']
        print(f"{category:<25}: {above:>4}/{total:<4} kept ({above/total*100:>5.1f}%) | Mean score: {mean_score:.3f} | Threshold: {threshold:.2f}")
    return removed_triplets

def main():
    """Main pipeline entry point."""
    test_structured_denoising()

if __name__ == "__main__":
    main()
