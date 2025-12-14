#!/usr/bin/env python3
"""
Used under the data/scifact-open folder. (After running the get data script to download.)

Improved retrieval system.

Methods:
1. Dense retrieval with sentence transformers (replaces BM25)
2. Cross-encoder reranking (replaces neural re-ranker)
3. Hybrid: BM25 + Dense + Reranking

(compared with rerank_base_50.py, this script is retrieving directly from the full corpus instead of a fixed 50 docs)
"""

import json
import argparse
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    torch = None  # type: ignore
    print("Warning: transformers not available. Install with: pip install transformers torch")

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None  # type: ignore
    print("Warning: rank-bm25 not available. Install with: pip install rank-bm25")

import re


def load_corpus(corpus_file: str) -> List[Dict]:
    """Load corpus from jsonl file"""
    corpus = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    return corpus


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


def bm25_retrieval(
    claim: str,
    corpus: List[Dict],
    top_k: int = 100,
    title_weight: float = 2.0
) -> List[Tuple[int, float]]:
    """
    Improved BM25 retrieval with title weighting.
    Returns list of (doc_id, score) tuples.
    """
    if not BM25_AVAILABLE:
        raise ImportError("rank-bm25 is required for BM25 retrieval")
    
    # Prepare documents with title weighting
    doc_texts = []
    doc_ids = []
    title_texts = []
    
    for doc in corpus:
        # Title separately (will be weighted)
        title = doc.get('title', '')
        abstract_list = doc.get('abstract', [])  # abstract is a list of sentences
        abstract = ' '.join(abstract_list) if isinstance(abstract_list, list) else abstract_list if abstract_list else ""
        title_texts.append(tokenize(title))
        # Full text for main index
        full_text = title + ' ' + abstract if title else abstract
        doc_texts.append(tokenize(full_text))
        doc_ids.append(doc['doc_id'])
    
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


def dense_retrieval(
    claim: str,
    corpus: List[Dict],
    model: SentenceTransformer,
    top_k: int = 100,
    batch_size: int = 32,
    title_weight: float = 0.0,
    model_name: str = None
) -> List[Tuple[int, float]]:
    """
    Dense retrieval using sentence transformers.
    Returns list of (doc_id, score) tuples.
    """
    # Prepare documents: title + abstract
    doc_texts = []
    doc_ids = []
    
    for doc in corpus:
        title = doc.get('title', '')
        abstract = doc.get('abstract', [])  # abstract is a list of sentences
        # Join sentences
        abstract_text = ' '.join(abstract) if isinstance(abstract, list) else abstract if abstract else ""
        # Combine title and abstract
        text = title + ' ' + abstract_text if title else abstract_text
        doc_texts.append(text)
        doc_ids.append(doc['doc_id'])
    
    # Compute embeddings
    claim_embedding = model.encode([claim], convert_to_numpy=True, show_progress_bar=False)[0]
    doc_embeddings = model.encode(
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


def hybrid_retrieval(
    claim: str,
    corpus: List[Dict],
    bm25_results: List[Tuple[int, float]],
    dense_results: List[Tuple[int, float]],
    alpha: float = 0.5
) -> List[Tuple[int, float]]:
    """
    Simple and effective hybrid retrieval using Reciprocal Rank Fusion (RRF).
    This is more stable and proven to work better than complex score normalization.
    alpha: weight for BM25 RRF score (1-alpha for dense RRF score)
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


def retrieve_documents(
    claim: str,
    corpus: List[Dict],
    method: str,
    top_k: int,
    bm25_model=None,
    dense_model=None,
    reranker=None,
    hybrid_alpha: float = 0.5,
    rerank_top_k: int = 100,
    dense_model_name: str = None
) -> List[int]:
    """
    Main retrieval function.
    Returns list of doc_ids (not scores).
    """
    corpus_dict = {doc['doc_id']: doc for doc in corpus}
    
    if method == 'bm25':
        if not BM25_AVAILABLE:
            raise ValueError("BM25 requires rank-bm25 package")
        results = bm25_retrieval(claim, corpus, top_k=min(rerank_top_k, top_k * 2), title_weight=2.0)
        doc_ids = [doc_id for doc_id, _ in results[:top_k]]
    
    elif method == 'dense':
        if not ST_AVAILABLE or dense_model is None:
            raise ValueError("Dense retrieval requires sentence-transformers")
        results = dense_retrieval(claim, corpus, dense_model, top_k=min(rerank_top_k, top_k * 2), model_name=dense_model_name)
        doc_ids = [doc_id for doc_id, _ in results[:top_k]]
    
    elif method == 'hybrid':
        if not BM25_AVAILABLE or not ST_AVAILABLE or dense_model is None:
            raise ValueError("Hybrid requires both rank-bm25 and sentence-transformers")
        bm25_results = bm25_retrieval(claim, corpus, top_k=rerank_top_k, title_weight=2.0)
        dense_results = dense_retrieval(claim, corpus, dense_model, top_k=rerank_top_k, model_name=dense_model_name)
        combined = hybrid_retrieval(claim, corpus, bm25_results, dense_results, alpha=hybrid_alpha)
        doc_ids = [doc_id for doc_id, _ in combined[:top_k]]
    
    elif method == 'dense_rerank':
        if not ST_AVAILABLE or dense_model is None or reranker is None:
            raise ValueError("Dense+rerank requires sentence-transformers with CrossEncoder")
        # First stage: dense retrieval - retrieve more for better coverage
        candidates = dense_retrieval(claim, corpus, dense_model, top_k=min(rerank_top_k * 2, len(corpus)), model_name=dense_model_name)
        # Get document dicts
        candidate_docs = [(doc_id, corpus_dict[doc_id]) for doc_id, _ in candidates[:rerank_top_k] if doc_id in corpus_dict]
        # Rerank
        reranked = rerank_with_cross_encoder(claim, candidate_docs, reranker, top_k=top_k)
        doc_ids = [doc_id for doc_id, _ in reranked]
    
    elif method == 'hybrid_rerank':
        if not BM25_AVAILABLE or not ST_AVAILABLE or dense_model is None or reranker is None:
            raise ValueError("Hybrid+rerank requires rank-bm25 and sentence-transformers with CrossEncoder")
        # First stage: hybrid retrieval - retrieve MORE candidates for better coverage
        # Use 4-5x more candidates to ensure we don't miss evidence docs
        max_candidates = min(max(rerank_top_k * 4, 300), len(corpus))
        bm25_results = bm25_retrieval(claim, corpus, top_k=max_candidates, title_weight=2.0)
        dense_results = dense_retrieval(claim, corpus, dense_model, top_k=max_candidates, model_name=dense_model_name)
        combined = hybrid_retrieval(claim, corpus, bm25_results, dense_results, alpha=hybrid_alpha)
        # Get document dicts - take top rerank_top_k for reranking (keep more for reranking)
        candidate_docs = [(doc_id, corpus_dict[doc_id]) for doc_id, _ in combined[:rerank_top_k] if doc_id in corpus_dict]
        # Rerank to get final top_k
        reranked = rerank_with_cross_encoder(claim, candidate_docs, reranker, top_k=top_k)
        doc_ids = [doc_id for doc_id, _ in reranked]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return doc_ids


def main():
    parser = argparse.ArgumentParser(description='Improved retrieval system')
    parser.add_argument('--claims', type=str, required=True, help='Input claims jsonl file')
    parser.add_argument('--corpus', type=str, required=True, help='Corpus jsonl file')
    parser.add_argument('--output', type=str, required=True, help='Output claims jsonl file')
    parser.add_argument('--method', type=str, 
                       choices=['bm25', 'dense', 'hybrid', 'dense_rerank', 'hybrid_rerank'],
                       default='hybrid_rerank',
                       help='Retrieval method')
    parser.add_argument('--top-k', type=int, default=35,
                       help='Number of documents to retrieve per claim')
    parser.add_argument('--rerank-top-k', type=int, default=200,
                       help='Number of candidates for reranking (more = better coverage, but slower)')
    parser.add_argument('--dense-model', type=str,
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Dense retrieval model name. Good options: '
                            'all-MiniLM-L6-v2 (fast), '
                            'all-mpnet-base-v2 (better quality), '
                            'multi-qa-mpnet-base-dot-v1 (tuned for Q&A retrieval)')
    parser.add_argument('--reranker-model', type=str,
                       default='cross-encoder/ms-marco-MiniLM-L-6-v2',
                       help='Cross-encoder reranker model name (using smaller model for speed)')
    parser.add_argument('--hybrid-alpha', type=float, default=0.5,
                       help='Weight for BM25 in hybrid RRF (0.5 = equal weight)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for model inference')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate retrieval quality (requires evidence in claims)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading corpus from {args.corpus}...")
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} documents")
    
    print(f"Loading claims from {args.claims}...")
    claims = load_claims(args.claims)
    print(f"Loaded {len(claims)} claims")
    
    # Load models
    dense_model = None
    reranker = None
    
    if args.method in ['dense', 'hybrid', 'dense_rerank', 'hybrid_rerank']:
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers required for this method")
        print(f"Loading dense retrieval model: {args.dense_model}...")
        dense_model = SentenceTransformer(args.dense_model)
        print("Dense model loaded")
    
    if args.method in ['dense_rerank', 'hybrid_rerank']:
        print(f"Loading reranker model: {args.reranker_model}...")
        reranker = CrossEncoder(args.reranker_model)
        print("Reranker loaded")
    
    # Retrieve documents for each claim
    print(f"\nRetrieving documents using method: {args.method}")
    print(f"Top-k: {args.top_k}")
    
    results = []
    stats = {
        'total_claims': 0,
        'claims_with_evidence': 0,
        'total_retrieved_docs': 0,
        'evidence_coverage': [],
    }
    
    for claim in tqdm(claims, desc="Retrieving"):
        stats['total_claims'] += 1
        
        # Retrieve documents (NO evidence information used)
        rerank_k = args.rerank_top_k
        
        retrieved_doc_ids = retrieve_documents(
            claim['claim'],
            corpus,
            args.method,
            args.top_k,
            bm25_model=None,  # BM25 doesn't need pre-loaded model
            dense_model=dense_model,
            reranker=reranker,
            hybrid_alpha=args.hybrid_alpha,
            rerank_top_k=rerank_k,
            dense_model_name=args.dense_model
        )
        
        stats['total_retrieved_docs'] += len(retrieved_doc_ids)
        
        # Evaluate if evidence is available
        if args.evaluate and claim.get('evidence'):
            stats['claims_with_evidence'] += 1
            evidence_doc_ids = set(int(doc_id) for doc_id in claim['evidence'].keys())
            retrieved_set = set(retrieved_doc_ids)
            coverage = len(evidence_doc_ids & retrieved_set) / len(evidence_doc_ids) if evidence_doc_ids else 0.0
            stats['evidence_coverage'].append(coverage)
        
        # Create output claim
        output_claim = claim.copy()
        output_claim['doc_ids'] = retrieved_doc_ids
        results.append(output_claim)
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for claim in results:
            f.write(json.dumps(claim, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n" + "="*80)
    print("Retrieval Statistics")
    print("="*80)
    print(f"Total claims: {stats['total_claims']}")
    print(f"Average documents per claim: {stats['total_retrieved_docs'] / stats['total_claims']:.2f}")
    
    if args.evaluate and stats['claims_with_evidence'] > 0:
        avg_coverage = sum(stats['evidence_coverage']) / len(stats['evidence_coverage'])
        min_coverage = min(stats['evidence_coverage'])
        coverage_95 = sum(1 for c in stats['evidence_coverage'] if c >= 0.95) / len(stats['evidence_coverage'])
        
        print(f"\nEvidence Coverage (for {stats['claims_with_evidence']} claims with evidence):")
        print(f"  Average coverage: {avg_coverage*100:.2f}%")
        print(f"  Minimum coverage: {min_coverage*100:.2f}%")
        print(f"  Claims with ≥95% coverage: {coverage_95*100:.2f}%")
    
    print("\nRetrieval complete!")


if __name__ == '__main__':
    main()

