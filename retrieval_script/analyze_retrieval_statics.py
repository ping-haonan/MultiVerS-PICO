#!/usr/bin/env python3
"""
Used under the data/scifact-open folder. (After running the get data script to download.)

Analyze the claims_new.jsonl file:
1. Count how many doc_ids each claim has.
2. Compute the proportion of evidence doc_ids that are included in doc_ids.
"""

import json
from collections import defaultdict

def analyze_claims(file_path, max_lines=None):
    """
    Analyze the claims file
    
    Args:
        file_path: Path to the jsonl file
        max_lines: Maximum number of lines to read (None = read all)
    """
    results = []
    stats = {
        'total_claims': 0,
        'claims_with_evidence': 0,
        'claims_without_evidence': 0,
        'total_doc_ids': 0,
        'total_evidence_doc_ids': 0,
        'fully_covered': 0,      # All evidence doc_ids are included in doc_ids
        'partially_covered': 0,  # Some evidence doc_ids are included in doc_ids
        'not_covered': 0,        # None of the evidence doc_ids are included in doc_ids
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Stop if max_lines is specified
            if max_lines is not None and line_num > max_lines:
                break
                
            line = line.strip()
            if not line:
                continue

            claim_data = json.loads(line)
            claim_id = claim_data.get('id')
            
            # doc_ids are integers; convert to a set of ints
            doc_ids = set(int(doc_id) for doc_id in claim_data.get('doc_ids', []))
            evidence = claim_data.get('evidence', {})

            stats['total_claims'] += 1
            stats['total_doc_ids'] += len(doc_ids)

            # Extract evidence doc_ids; keys are strings and must be converted to ints
            evidence_doc_ids = set(int(doc_id) for doc_id in evidence.keys()) if evidence else set()

            if evidence_doc_ids:
                stats['claims_with_evidence'] += 1
                stats['total_evidence_doc_ids'] += len(evidence_doc_ids)

                # Compute coverage
                covered = evidence_doc_ids & doc_ids
                coverage_ratio = len(covered) / len(evidence_doc_ids) if evidence_doc_ids else 0.0

                if coverage_ratio == 1.0:
                    stats['fully_covered'] += 1
                elif coverage_ratio > 0:
                    stats['partially_covered'] += 1
                else:
                    stats['not_covered'] += 1

                results.append({
                    'claim_id': claim_id,
                    'num_doc_ids': len(doc_ids),
                    'num_evidence_doc_ids': len(evidence_doc_ids),
                    'coverage_ratio': coverage_ratio,
                    'covered_doc_ids': len(covered),
                    'missing_doc_ids': list(evidence_doc_ids - doc_ids),
                })
            else:
                stats['claims_without_evidence'] += 1
                results.append({
                    'claim_id': claim_id,
                    'num_doc_ids': len(doc_ids),
                    'num_evidence_doc_ids': 0,
                    'coverage_ratio': None,
                    'covered_doc_ids': 0,
                    'missing_doc_ids': [],
                })

    return results, stats

def print_statistics(results, stats):
    """Print summary statistics"""
    print("=" * 80)
    print("Overall Statistics")
    print("=" * 80)
    print(f"Total number of claims: {stats['total_claims']}")
    print(f"Claims with evidence: {stats['claims_with_evidence']}")
    print(f"Claims without evidence: {stats['claims_without_evidence']}")
    print(f"Total number of doc_ids: {stats['total_doc_ids']}")
    print(f"Total number of evidence doc_ids: {stats['total_evidence_doc_ids']}")
    print()

    if stats['claims_with_evidence'] > 0:
        print("=" * 80)
        print("Coverage Statistics (for claims with evidence)")
        print("=" * 80)
        print(f"Fully covered (100%): {stats['fully_covered']} ({stats['fully_covered']/stats['claims_with_evidence']*100:.2f}%)")
        print(f"Partially covered (0-100%): {stats['partially_covered']} ({stats['partially_covered']/stats['claims_with_evidence']*100:.2f}%)")
        print(f"Not covered (0%): {stats['not_covered']} ({stats['not_covered']/stats['claims_with_evidence']*100:.2f}%)")
        print()

        # Average coverage ratio
        coverage_ratios = [r['coverage_ratio'] for r in results if r['coverage_ratio'] is not None]
        if coverage_ratios:
            avg_coverage = sum(coverage_ratios) / len(coverage_ratios)
            print(f"Average coverage ratio: {avg_coverage*100:.2f}%")
            print()

    # Distribution of doc_ids per claim
    print("=" * 80)
    print("Distribution of number of doc_ids per claim")
    print("=" * 80)
    doc_id_counts = [r['num_doc_ids'] for r in results]
    if doc_id_counts:
        print(f"Minimum doc_ids: {min(doc_id_counts)}")
        print(f"Maximum doc_ids: {max(doc_id_counts)}")
        print(f"Average doc_ids: {sum(doc_id_counts)/len(doc_id_counts):.2f}")
        print(f"Median doc_ids: {sorted(doc_id_counts)[len(doc_id_counts)//2]}")
        print()

    # Example output
    print("=" * 80)
    print("Example entries (first 10 claims with evidence)")
    print("=" * 80)
    examples = [r for r in results if r['coverage_ratio'] is not None][:10]
    for ex in examples:
        print(f"Claim ID: {ex['claim_id']}")
        print(f"  Number of doc_ids: {ex['num_doc_ids']}")
        print(f"  Number of evidence doc_ids: {ex['num_evidence_doc_ids']}")
        print(f"  Coverage ratio: {ex['coverage_ratio']*100:.2f}%")
        if ex['missing_doc_ids']:
            print(f"  Missing doc_ids: {ex['missing_doc_ids']}")
        print()

def save_detailed_results(results, output_path):
    """Save detailed results to a JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {output_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze claims jsonl file')
    parser.add_argument('--input', type=str, default='claims_new.jsonl',
                       help='Input claims jsonl file')
    parser.add_argument('--output', type=str, default='claims_analysis_results.json',
                       help='Output JSON file for detailed results')
    parser.add_argument('--max-lines', type=int, default=None,
                       help='Maximum number of lines to read (for subset testing)')
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    
    if args.max_lines:
        print(f"Starting analysis of {input_file} (first {args.max_lines} lines)...")
    else:
        print(f"Starting analysis of {input_file}...")
    
    results, stats = analyze_claims(input_file, max_lines=args.max_lines)
    
    if args.max_lines:
        print(f"\nNote: Analyzed only first {args.max_lines} lines")
    
    print_statistics(results, stats)
    save_detailed_results(results, output_file)

    print("Analysis complete!")
