#!/usr/bin/env python3
"""
Used under the data/scifact-open folder. (After running the get data script to download.)

Reduce the number of doc_ids per claim from 50 to an average of 35.
All documents referenced in evidence must be included.

This is an ideal retrieval simulation step to test whether reducing the number of
documents while keeping all evidence documents helps downstream performance.
"""

import json
import argparse
import random
from typing import List, Set, Dict
from collections import defaultdict


def load_claims(claims_file: str) -> List[Dict]:
    """Load claims from a JSONL file"""
    claims = []
    with open(claims_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                claims.append(json.loads(line))
    return claims


def get_evidence_doc_ids(claim: Dict) -> Set[int]:
    """Extract all evidence-related doc_ids from a claim"""
    evidence_doc_ids = set()
    if 'evidence' in claim and claim['evidence']:
        # Evidence keys are doc_id strings
        for doc_id_str in claim['evidence'].keys():
            try:
                evidence_doc_ids.add(int(doc_id_str))
            except (ValueError, TypeError):
                # Skip if conversion fails
                continue
    return evidence_doc_ids


def reduce_doc_ids(
    claim: Dict,
    target_avg: int = 35,
    current_count: int = 50,
    seed: int = None
) -> List[int]:
    """
    Reduce the doc_ids list while ensuring all evidence doc_ids are kept.

    Args:
        claim: Claim dictionary containing doc_ids and optional evidence
        target_avg: Target average number of documents
        current_count: Current number of documents per claim (default: 50)
        seed: Random seed

    Returns:
        Reduced list of doc_ids
    """
    if seed is not None:
        random.seed(seed)

    # Original doc_ids
    original_doc_ids = claim.get('doc_ids', [])
    if len(original_doc_ids) == 0:
        return []

    # Evidence doc_ids (must be kept)
    evidence_doc_ids = get_evidence_doc_ids(claim)

    target_count = target_avg

    # If evidence alone already reaches or exceeds the target,
    # we must keep all evidence documents
    if len(evidence_doc_ids) >= target_count:
        # Preserve original order
        evidence_list = [
            doc_id for doc_id in original_doc_ids
            if doc_id in evidence_doc_ids
        ]
        return evidence_list

    # Separate evidence and non-evidence docs
    original_set = set(original_doc_ids)
    evidence_in_original = evidence_doc_ids & original_set
    other_doc_ids = [
        doc_id for doc_id in original_doc_ids
        if doc_id not in evidence_doc_ids
    ]

    # Number of additional docs needed
    needed_count = target_count - len(evidence_in_original)

    # If not enough remaining docs, return everything available
    if len(other_doc_ids) < needed_count:
        result = list(evidence_in_original) + other_doc_ids
        return result

    # Randomly sample remaining docs
    selected_other = random.sample(other_doc_ids, needed_count)

    # Merge results while preserving original order
    result = []
    evidence_list = [
        doc_id for doc_id in original_doc_ids
        if doc_id in evidence_in_original
    ]
    result.extend(evidence_list)

    selected_other_set = set(selected_other)
    other_sorted = [
        doc_id for doc_id in original_doc_ids
        if doc_id in selected_other_set
    ]
    result.extend(other_sorted)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Reduce doc_ids in claims while preserving all evidence documents'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input claims JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output claims JSONL file'
    )
    parser.add_argument(
        '--target-avg',
        type=int,
        default=35,
        help='Target average number of documents (default: 35)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--current-count',
        type=int,
        default=50,
        help='Current number of documents per claim (default: 50)'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load claims
    print(f"Loading claims file: {args.input}")
    claims = load_claims(args.input)
    print(f"Loaded {len(claims)} claims")

    # Statistics
    stats = {
        'total_claims': len(claims),
        'claims_with_evidence': 0,
        'total_evidence_docs': 0,
        'total_docs_before': 0,
        'total_docs_after': 0,
        'evidence_coverage': [],  # Per-claim evidence coverage
    }

    # Process claims
    print(f"\nProcessing claims, target average doc count: {args.target_avg}")
    results = []

    for claim in claims:
        original_count = len(claim.get('doc_ids', []))
        stats['total_docs_before'] += original_count

        evidence_doc_ids = get_evidence_doc_ids(claim)
        if evidence_doc_ids:
            stats['claims_with_evidence'] += 1
            stats['total_evidence_docs'] += len(evidence_doc_ids)

        reduced_doc_ids = reduce_doc_ids(
            claim,
            target_avg=args.target_avg,
            current_count=args.current_count,
            seed=args.seed
        )

        stats['total_docs_after'] += len(reduced_doc_ids)

        # Check evidence coverage
        if evidence_doc_ids:
            reduced_set = set(reduced_doc_ids)
            coverage = len(evidence_doc_ids & reduced_set) / len(evidence_doc_ids)
            stats['evidence_coverage'].append(coverage)
            if coverage < 1.0:
                print(
                    f"Warning: claim {claim.get('id', 'unknown')} "
                    f"has evidence coverage of only {coverage * 100:.1f}%"
                )

        output_claim = claim.copy()
        output_claim['doc_ids'] = reduced_doc_ids
        results.append(output_claim)

    # Save results
    print(f"\nSaving results to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for claim in results:
            f.write(json.dumps(claim, ensure_ascii=False) + '\n')

    # Print statistics
    print("\n" + "=" * 80)
    print("Processing Statistics")
    print("=" * 80)
    print(f"Total claims: {stats['total_claims']}")
    print(f"Claims with evidence: {stats['claims_with_evidence']}")

    print("\nDocument count:")
    print(f"  Average before: {stats['total_docs_before'] / stats['total_claims']:.2f}")
    print(f"  Average after: {stats['total_docs_after'] / stats['total_claims']:.2f}")
    print(f"  Target average: {args.target_avg}")

    if stats['claims_with_evidence'] > 0:
        print("\nEvidence statistics:")
        print(f"  Total evidence docs: {stats['total_evidence_docs']}")
        print(
            f"  Average evidence docs per claim: "
            f"{stats['total_evidence_docs'] / stats['claims_with_evidence']:.2f}"
        )

        if stats['evidence_coverage']:
            avg_coverage = sum(stats['evidence_coverage']) / len(stats['evidence_coverage'])
            min_coverage = min(stats['evidence_coverage'])
            perfect_coverage = (
                sum(1 for c in stats['evidence_coverage'] if c >= 1.0)
                / len(stats['evidence_coverage'])
            )

            print("\nEvidence coverage:")
            print(f"  Average coverage: {avg_coverage * 100:.2f}%")
            print(f"  Minimum coverage: {min_coverage * 100:.2f}%")
            print(f"  Fully covered claims: {perfect_coverage * 100:.2f}%")

    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
