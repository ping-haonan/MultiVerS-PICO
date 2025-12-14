#!/usr/bin/env python3
"""
Used under the data/scifact-open folder. (After running the get data script to download.)

Filter documents from claims.jsonl:
1. Read claims.jsonl which has 50 preprocessed doc_ids per claim
2. Load the corresponding documents from corpus
3. Filter to a specified number of documents using improved methods:
   - Option 1: Dense retrieval + Reranking (two-stage)
   - Option 2: BM25 + Dense + Reranking (hybrid)
   - Option 3: Direct reranking (original, but improved)
4. Output filtered claims with reduced doc_ids

Usage: python filter_claims_docs.py     --claims claims.jsonl     --corpus corpus_candidates.jsonl     --output claims_filtered.jsonl     --top-k 30     --method rerank     --evaluate
"""

import json
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import re

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None  # type: ignore
    print("Warning: rank-bm25 not available. Install with: pip install rank-bm25")


def load_corpus(corpus_file: str) -> Dict[int, Dict]:
    """Load corpus from jsonl file and return as a dictionary keyed by doc_id"""
    corpus_dict = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                corpus_dict[doc['doc_id']] = doc
    return corpus_dict


def load_claims(claims_file: str) -> List[Dict]:
    """Load claims from jsonl file"""
    claims = []
    with open(claims_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                claims.append(json.loads(line))
    return claims


def tokenize(text: str) -> List[str]:
    """Simple tokenization for BM25"""
    return re.findall(r'\b\w+\b', text.lower())


def bm25_retrieval_for_candidates(
    claim: str,
    candidate_docs: List[Tuple[int, Dict]],
    top_k: int = None,
    title_weight: float = 2.0
) -> List[Tuple[int, float]]:
    """
    BM25 retrieval for a subset of candidate documents.
    Returns list of (doc_id, score) tuples.
    """
    if not BM25_AVAILABLE:
        raise ImportError("rank-bm25 is required for BM25 retrieval")
    
    if top_k is None:
        top_k = len(candidate_docs)
    
    # Prepare documents with title weighting
    doc_texts = []
    doc_ids = []
    title_texts = []
    
    for doc_id, doc in candidate_docs:
        title = doc.get('title', '')
        abstract_list = doc.get('abstract', [])
        abstract = ' '.join(abstract_list) if isinstance(abstract_list, list) else abstract_list if abstract_list else ""
        title_texts.append(tokenize(title))
        full_text = title + ' ' + abstract if title else abstract
        doc_texts.append(tokenize(full_text))
        doc_ids.append(doc_id)
    
    # Create BM25 index
    bm25 = BM25Okapi(doc_texts)
    bm25_title = BM25Okapi(title_texts)
    
    # Query
    query_tokens = tokenize(claim)
    scores = bm25.get_scores(query_tokens)
    title_scores = bm25_title.get_scores(query_tokens)
    
    # Combine with title weighting
    combined_scores = scores + title_weight * title_scores
    
    # Get top-k
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    results = [(doc_ids[i], float(combined_scores[i])) for i in top_indices]
    
    return results


def dense_retrieval_for_candidates(
    claim: str,
    candidate_docs: List[Tuple[int, Dict]],
    dense_model: SentenceTransformer,
    top_k: int = None,
    batch_size: int = 32
) -> List[Tuple[int, float]]:
    """
    Dense retrieval for a subset of candidate documents.
    Returns list of (doc_id, score) tuples.
    """
    if top_k is None:
        top_k = len(candidate_docs)
    
    # Prepare documents: title + abstract
    doc_texts = []
    doc_ids = []
    
    for doc_id, doc in candidate_docs:
        title = doc.get('title', '')
        abstract = doc.get('abstract', [])
        abstract_text = ' '.join(abstract) if isinstance(abstract, list) else abstract if abstract else ""
        text = title + ' ' + abstract_text if title else abstract_text
        doc_texts.append(text)
        doc_ids.append(doc_id)
    
    # Compute embeddings
    claim_embedding = dense_model.encode([claim], convert_to_numpy=True, show_progress_bar=False)[0]
    doc_embeddings = dense_model.encode(
        doc_texts, 
        convert_to_numpy=True, 
        batch_size=batch_size,
        show_progress_bar=False
    )
    
    # Normalize and compute cosine similarity
    claim_embedding = claim_embedding / np.linalg.norm(claim_embedding)
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    scores = np.dot(doc_embeddings, claim_embedding)
    
    # Get top-k
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(doc_ids[i], float(scores[i])) for i in top_indices]
    
    return results


def hybrid_retrieval_for_candidates(
    claim: str,
    candidate_docs: List[Tuple[int, Dict]],
    bm25_results: List[Tuple[int, float]],
    dense_results: List[Tuple[int, float]],
    alpha: float = 0.5
) -> List[Tuple[int, float]]:
    """
    Hybrid retrieval using Reciprocal Rank Fusion (RRF) for candidates.
    """
    # Create rank dictionaries (rank starts from 1)
    bm25_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_results, 1)}
    dense_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_results, 1)}
    
    # Get all unique doc_ids
    all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
    max_rank = len(all_doc_ids) + 1
    
    # Compute RRF scores (k=60 is standard)
    k = 60
    combined_scores = {}
    for doc_id in all_doc_ids:
        bm25_rank = bm25_ranks.get(doc_id, max_rank)
        dense_rank = dense_ranks.get(doc_id, max_rank)
        
        # RRF formula: 1 / (k + rank)
        rrf_bm25 = 1.0 / (k + bm25_rank)
        rrf_dense = 1.0 / (k + dense_rank)
        
        # Weighted combination
        combined_scores[doc_id] = alpha * rrf_bm25 + (1 - alpha) * rrf_dense
    
    # Sort by combined score
    results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return results


def rrf_fusion(rankings: List[List[Tuple[int, float]]], k: int = 60) -> List[int]:
    """
    Reciprocal Rank Fusion (RRF) to combine multiple rankings.
    rankings: list of ranked lists, each is [(doc_id, score), ...]
    Returns: list of doc_ids sorted by RRF score
    """
    # Create rank dictionaries for each ranking
    rank_dicts = []
    all_doc_ids = set()
    
    for ranking in rankings:
        rank_dict = {doc_id: rank for rank, (doc_id, _) in enumerate(ranking, 1)}
        rank_dicts.append(rank_dict)
        all_doc_ids.update(rank_dict.keys())
    
    max_rank = len(all_doc_ids) + 1
    
    # Compute RRF scores
    rrf_scores = {}
    for doc_id in all_doc_ids:
        rrf_score = 0.0
        for rank_dict in rank_dicts:
            rank = rank_dict.get(doc_id, max_rank)
            rrf_score += 1.0 / (k + rank)
        rrf_scores[doc_id] = rrf_score
    
    # Sort by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs]


def rerank_with_cross_encoder(
    claim: str,
    candidate_docs: List[Tuple[int, Dict]],
    reranker: CrossEncoder,
    top_k: int,
    batch_size: int = 32
) -> List[Tuple[int, float]]:
    """
    Rerank candidates using a cross-encoder.
    candidate_docs: list of (doc_id, doc_dict) tuples
    """
    # Prepare pairs
    pairs = []
    doc_ids = []
    
    for doc_id, doc in candidate_docs:
        title = doc.get('title', '')
        abstract = doc.get('abstract', [])  # abstract is a list of sentences
        abstract_text = ' '.join(abstract) if isinstance(abstract, list) else abstract if abstract else ""
        text = title + ' ' + abstract_text if title else abstract_text
        pairs.append([claim, text])
        doc_ids.append(doc_id)
    
    # Score pairs
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    
    # Sort by score
    results = list(zip(doc_ids, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


def filter_documents_for_claim(
    claim: Dict,
    corpus_dict: Dict[int, Dict],
    method: str,
    top_k: int,
    dense_model=None,
    reranker=None,
    batch_size: int = 32,
    hybrid_alpha: float = 0.5,
    dense_top_k: int = None  # For two-stage: dense retrieval top-k before reranking
) -> List[int]:
    """
    Filter documents for a single claim using improved methods.
    Takes the preprocessed doc_ids from the claim, loads the documents,
    and filters them using the specified method.
    
    Methods:
    - 'dense_rerank': Dense retrieval first, then rerank top candidates
    - 'hybrid_rerank': BM25 + Dense hybrid, then rerank top candidates
    - 'rerank': Direct reranking (original method)
    
    Returns list of doc_ids (not scores).
    """
    claim_text = claim.get('claim', '')
    preprocessed_doc_ids = claim.get('doc_ids', [])
    
    # If we already have fewer or equal docs than top_k, return as is
    if len(preprocessed_doc_ids) <= top_k:
        return preprocessed_doc_ids
    
    # Get candidate documents from corpus
    candidate_docs = []
    for doc_id in preprocessed_doc_ids:
        if doc_id in corpus_dict:
            candidate_docs.append((doc_id, corpus_dict[doc_id]))
        else:
            print(f"Warning: doc_id {doc_id} not found in corpus")
    
    # If we don't have enough documents in corpus, return what we have
    if len(candidate_docs) <= top_k:
        return [doc_id for doc_id, _ in candidate_docs]
    
    # Apply filtering method
    if method == 'dense_rerank':
        if dense_model is None or reranker is None:
            raise ValueError("dense_rerank requires both dense_model and reranker")
        
        # Strategy: Use RRF to fuse rerank and dense rankings for better stability
        if len(candidate_docs) <= 50:
            # Rerank all documents
            reranked_all = rerank_with_cross_encoder(
                claim_text, candidate_docs, reranker, top_k=len(candidate_docs), batch_size=batch_size
            )
            
            # Get dense ranking for all candidates
            dense_results = dense_retrieval_for_candidates(
                claim_text, candidate_docs, dense_model, top_k=len(candidate_docs), batch_size=batch_size
            )
            
            # Use RRF to fuse the two rankings (more stable than score fusion)
            # Rerank is more accurate, so we can give it more weight by using it twice
            rankings = [reranked_all, reranked_all, dense_results]  # Rerank appears twice for higher weight
            fused_ranking = rrf_fusion(rankings, k=60)
            filtered_doc_ids = fused_ranking[:top_k]
        else:
            # For larger sets, use two-stage approach
            if dense_top_k is None:
                dense_top_k = min(len(candidate_docs), max(top_k * 3, 50))  # Get more candidates
            
            dense_results = dense_retrieval_for_candidates(
                claim_text, candidate_docs, dense_model, top_k=dense_top_k, batch_size=batch_size
            )
            dense_doc_ids = [doc_id for doc_id, _ in dense_results]
            dense_candidate_docs = [(doc_id, corpus_dict[doc_id]) for doc_id in dense_doc_ids if doc_id in corpus_dict]
            
            reranked = rerank_with_cross_encoder(claim_text, dense_candidate_docs, reranker, top_k=top_k, batch_size=batch_size)
            filtered_doc_ids = [doc_id for doc_id, _ in reranked]
    
    elif method == 'hybrid_rerank':
        if not BM25_AVAILABLE or dense_model is None or reranker is None:
            raise ValueError("hybrid_rerank requires BM25, dense_model, and reranker")
        
        # Strategy: Use RRF to fuse rerank, BM25, and Dense rankings
        if len(candidate_docs) <= 50:
            # Rerank all documents
            reranked_all = rerank_with_cross_encoder(
                claim_text, candidate_docs, reranker, top_k=len(candidate_docs), batch_size=batch_size
            )
            
            # Get BM25 and Dense rankings
            bm25_results = bm25_retrieval_for_candidates(claim_text, candidate_docs, top_k=len(candidate_docs))
            dense_results = dense_retrieval_for_candidates(
                claim_text, candidate_docs, dense_model, top_k=len(candidate_docs), batch_size=batch_size
            )
            
            # Use RRF to fuse all three rankings
            # Give rerank more weight by including it multiple times
            rankings = [reranked_all, reranked_all, bm25_results, dense_results]  # Rerank x2 for higher weight
            fused_ranking = rrf_fusion(rankings, k=60)
            filtered_doc_ids = fused_ranking[:top_k]
        else:
            # For larger sets, use two-stage
            bm25_results = bm25_retrieval_for_candidates(claim_text, candidate_docs, top_k=len(candidate_docs))
            dense_results = dense_retrieval_for_candidates(
                claim_text, candidate_docs, dense_model, top_k=len(candidate_docs), batch_size=batch_size
            )
            hybrid_results = hybrid_retrieval_for_candidates(
                claim_text, candidate_docs, bm25_results, dense_results, alpha=hybrid_alpha
            )
            
            rerank_candidate_count = min(len(hybrid_results), max(top_k * 3, 50))
            hybrid_doc_ids = [doc_id for doc_id, _ in hybrid_results[:rerank_candidate_count]]
            hybrid_candidate_docs = [(doc_id, corpus_dict[doc_id]) for doc_id in hybrid_doc_ids if doc_id in corpus_dict]
            
            reranked = rerank_with_cross_encoder(claim_text, hybrid_candidate_docs, reranker, top_k=top_k, batch_size=batch_size)
            filtered_doc_ids = [doc_id for doc_id, _ in reranked]
    
    elif method == 'rerank':
        if reranker is None:
            raise ValueError("rerank requires reranker")
        
        # Direct reranking: rerank all candidates to ensure best selection
        # For better results, we can also combine with a simple ranking based on original order
        # as a fallback signal
        reranked_all = rerank_with_cross_encoder(
            claim_text, candidate_docs, reranker, top_k=len(candidate_docs), batch_size=batch_size
        )
        
        # Use original order as a secondary signal (documents earlier in the list might be more relevant)
        original_order = [(doc_id, len(candidate_docs) - idx) for idx, (doc_id, _) in enumerate(candidate_docs)]
        original_order.sort(key=lambda x: x[1], reverse=True)  # Sort by position (earlier = better)
        
        # Fuse rerank with original order using RRF
        rankings = [reranked_all, original_order]
        fused_ranking = rrf_fusion(rankings, k=60)
        filtered_doc_ids = fused_ranking[:top_k]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return filtered_doc_ids


def main():
    parser = argparse.ArgumentParser(description='Filter documents from claims.jsonl using improved methods')
    parser.add_argument('--claims', type=str, default='claims.jsonl',
                       help='Input claims jsonl file (with preprocessed 50 doc_ids)')
    parser.add_argument('--corpus', type=str, required=True,
                       help='Corpus jsonl file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output claims jsonl file with filtered doc_ids')
    parser.add_argument('--top-k', type=int, required=True,
                       help='Number of documents to retrieve per claim (must be <= 50)')
    parser.add_argument('--method', type=str,
                       choices=['dense_rerank', 'hybrid_rerank', 'rerank'],
                       default='hybrid_rerank',
                       help='Filtering method: dense_rerank (best quality), hybrid_rerank (balanced), rerank (fastest)')
    parser.add_argument('--dense-model', type=str,
                       default='sentence-transformers/multi-qa-mpnet-base-dot-v1',
                       help='Dense retrieval model name. Better options: '
                            'multi-qa-mpnet-base-dot-v1 (best for Q&A, default), '
                            'all-mpnet-base-v2 (general purpose, high quality), '
                            'all-MiniLM-L6-v2 (fast but lower quality)')
    parser.add_argument('--reranker-model', type=str,
                       default='BAAI/bge-reranker-v2-m3',
                       help='Cross-encoder reranker model name. Recommended models (best to good): '
                            'BAAI/bge-reranker-v2-m3 (best quality, default, ~1.3GB), '
                            'BAAI/bge-reranker-large (very good, ~1.3GB), '
                            'BAAI/bge-reranker-base (good, ~400MB), '
                            'cross-encoder/ms-marco-MiniLM-L-12-v2 (balanced, ~400MB), '
                            'cross-encoder/ms-marco-electra-base (good, ~400MB), '
                            'cross-encoder/ms-marco-MiniLM-L-6-v2 (fastest, ~100MB)')
    parser.add_argument('--hybrid-alpha', type=float, default=0.5,
                       help='Weight for BM25 in hybrid RRF (0.5 = equal weight)')
    parser.add_argument('--dense-top-k', type=int, default=None,
                       help='Number of candidates for dense retrieval before reranking (default: top_k * 2)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for model inference')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate retrieval quality (requires evidence in claims)')
    
    args = parser.parse_args()
    
    if args.top_k > 50:
        print("Warning: top-k is greater than 50. The input claims have at most 50 doc_ids.")
        print("Setting top-k to 50.")
        args.top_k = 50
    
    # Load data
    print(f"Loading corpus from {args.corpus}...")
    corpus_dict = load_corpus(args.corpus)
    print(f"Loaded {len(corpus_dict)} documents")
    
    print(f"Loading claims from {args.claims}...")
    claims = load_claims(args.claims)
    print(f"Loaded {len(claims)} claims")
    
    # Load models
    dense_model = None
    reranker = None
    
    if args.method in ['dense_rerank', 'hybrid_rerank']:
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers required for this method")
        print(f"Loading dense retrieval model: {args.dense_model}...")
        dense_model = SentenceTransformer(args.dense_model)
        print("Dense model loaded")
    
    if args.method in ['dense_rerank', 'hybrid_rerank', 'rerank']:
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers required for reranking")
        print(f"Loading reranker model: {args.reranker_model}...")
        print("Note: BGE reranker models may take longer to download on first use (~1.3GB)")
        reranker = CrossEncoder(args.reranker_model)
        print("Reranker loaded successfully")
    
    # Filter documents for each claim
    print(f"\nFiltering documents to top-{args.top_k} for each claim using method: {args.method}...")
    
    results = []
    stats = {
        'total_claims': 0,
        'claims_with_evidence': 0,
        'total_filtered_docs': 0,
        'evidence_coverage': [],
    }
    
    for claim in tqdm(claims, desc="Filtering"):
        stats['total_claims'] += 1
        
        # Filter documents
        filtered_doc_ids = filter_documents_for_claim(
            claim,
            corpus_dict,
            args.method,
            args.top_k,
            dense_model=dense_model,
            reranker=reranker,
            batch_size=args.batch_size,
            hybrid_alpha=args.hybrid_alpha,
            dense_top_k=args.dense_top_k
        )
        
        stats['total_filtered_docs'] += len(filtered_doc_ids)
        
        # Evaluate if evidence is available
        if args.evaluate and claim.get('evidence'):
            stats['claims_with_evidence'] += 1
            evidence_doc_ids = set(int(doc_id) for doc_id in claim['evidence'].keys())
            filtered_set = set(filtered_doc_ids)
            coverage = len(evidence_doc_ids & filtered_set) / len(evidence_doc_ids) if evidence_doc_ids else 0.0
            stats['evidence_coverage'].append(coverage)
        
        # Create output claim
        output_claim = claim.copy()
        output_claim['doc_ids'] = filtered_doc_ids
        results.append(output_claim)
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for claim in results:
            f.write(json.dumps(claim, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n" + "="*80)
    print("Filtering Statistics")
    print("="*80)
    print(f"Method: {args.method}")
    print(f"Total claims: {stats['total_claims']}")
    print(f"Average documents per claim: {stats['total_filtered_docs'] / stats['total_claims']:.2f}")
    print(f"Target top-k: {args.top_k}")
    
    if args.evaluate and stats['claims_with_evidence'] > 0:
        avg_coverage = sum(stats['evidence_coverage']) / len(stats['evidence_coverage'])
        min_coverage = min(stats['evidence_coverage'])
        coverage_95 = sum(1 for c in stats['evidence_coverage'] if c >= 0.95) / len(stats['evidence_coverage'])
        coverage_100 = sum(1 for c in stats['evidence_coverage'] if c >= 1.0) / len(stats['evidence_coverage'])
        
        print(f"\nEvidence Coverage (for {stats['claims_with_evidence']} claims with evidence):")
        print(f"  Average coverage: {avg_coverage*100:.2f}%")
        print(f"  Minimum coverage: {min_coverage*100:.2f}%")
        print(f"  Claims with ≥95% coverage: {coverage_95*100:.2f}%")
        print(f"  Claims with 100% coverage: {coverage_100*100:.2f}%")
    
    print("\nFiltering complete!")


if __name__ == '__main__':
    main()

