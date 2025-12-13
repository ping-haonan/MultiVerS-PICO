#!/usr/bin/env python3
"""
Preprocess SciFact data using BioELECTRA-PICO to generate features for MultiVerS training.

This script extracts:
1. PICO token tags (aligned to Longformer)
2. PICO span embeddings (768-dim vectors from BioELECTRA)
3. Statistical features (counts, scores)

It saves a single .pt file containing a dictionary mapping {id: features}.
"""

import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
import numpy as np
from collections import defaultdict

# Constants
TAG_MAP = {"Participants": 1, "Intervention": 2, "Outcome": 3}  # 0 is reserved for O/Padding
BIOELECTRA_PATH = "/root/autodl-tmp/models/BioELECTRA-PICO"
LONGFORMER_NAME = "allenai/longformer-base-4096"  # Or the one used in MultiVerS

def load_bioelectra(device):
    print("Loading BioELECTRA-PICO...")
    tokenizer = AutoTokenizer.from_pretrained(BIOELECTRA_PATH, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(BIOELECTRA_PATH, local_files_only=True)
    model.to(device)
    model.eval()
    return model, tokenizer

def extract_features_for_text(text, model, tokenizer, device):
    """
    Run BioELECTRA on text and return span embeddings and tags.
    """
    if not text:
        return None

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    offsets = inputs.pop("offset_mapping")[0].cpu().numpy()
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    logits = outputs.logits[0]  # [seq_len, num_labels]
    hidden_states = outputs.hidden_states[-1][0]  # [seq_len, 768]
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    
    id2label = model.config.id2label
    
    # Collect spans
    spans = []
    curr_span = None
    
    for idx, (pred, offset) in enumerate(zip(preds, offsets)):
        if offset[0] == offset[1]: continue # Skip special tokens
        
        label = id2label[pred]
        if label == 'O':
            if curr_span:
                spans.append(curr_span)
                curr_span = None
            continue
            
        tag_type = label.split('-')[-1] # B-Patient -> Patient
        if tag_type not in TAG_MAP:
            if curr_span:
                spans.append(curr_span)
                curr_span = None
            continue
            
        # Simple logic: contiguous same tags form a span
        if curr_span and curr_span['type'] == tag_type:
            curr_span['end'] = idx
            curr_span['tokens'].append(idx)
        else:
            if curr_span: spans.append(curr_span)
            curr_span = {
                'type': tag_type,
                'start': idx,
                'end': idx,
                'tokens': [idx],
                'score': probs[idx, pred]
            }
            
    if curr_span: spans.append(curr_span)
    
    # Aggregate span embeddings (Mean Pooling over tokens in span)
    span_embeddings = []
    span_types = []
    span_scores = []
    
    for span in spans:
        indices = span['tokens']
        emb = hidden_states[indices].mean(dim=0) # [768]
        span_embeddings.append(emb.cpu())
        span_types.append(TAG_MAP[span['type']])
        span_scores.append(span['score'])
        
    return {
        "span_embeddings": torch.stack(span_embeddings) if span_embeddings else torch.zeros(0, 768),
        "span_types": torch.tensor(span_types, dtype=torch.long) if span_types else torch.zeros(0, dtype=torch.long),
        "span_scores": torch.tensor(span_scores, dtype=torch.float) if span_scores else torch.zeros(0),
        "raw_text": text
    }

def process_file(input_path, model, tokenizer, device, mode="scifact_claim"):
    data_map = {}
    
    with open(input_path) as f:
        lines = f.readlines()
        
    print(f"Processing {len(lines)} lines from {input_path} in mode {mode}...")
    
    for line in tqdm(lines):
        item = json.loads(line)
        
        if mode == "corpus":
            # Process abstract sentences from corpus file
            doc_id = item['doc_id']
            abstract_sentences = item['abstract']
            sent_features = []
            for sent in abstract_sentences:
                feat = extract_features_for_text(sent, model, tokenizer, device)
                sent_features.append(feat)
            data_map[doc_id] = sent_features
            
        elif mode == "scifact_claim":
            # Process claim only (SciFact style)
            doc_id = item['id']
            claim = item['claim']
            feat = extract_features_for_text(claim, model, tokenizer, device)
            data_map[doc_id] = feat
            
        elif mode == "self_contained":
            # Process BOTH claim and sentences (FEVER/PubMedQA style)
            # We need to store them separately or together?
            # The DataModule expects:
            # - claims_pico.pt -> {claim_id: claim_features}
            # - corpus_pico.pt -> {doc_id: [sent_features]}
            # But for FEVER/PubMedQA, doc_id is tied to claim_id or abstract_id.
            # Let's see how Reader handles it.
            # Reader sets abstract_id = -1 for FEVER.
            # Reader sets abstract_id = instance["abstract_id"] for PubMedQA.
            
            # We will output TWO dictionaries from this mode if possible, 
            # or run this function twice (once for claim, once for sentences).
            # To keep it simple, let's just handle one type at a time based on flag.
            pass

    return data_map

def process_self_contained(input_path, model, tokenizer, device, target="claim"):
    """
    Extract features from self-contained datasets (FEVER/PubMedQA).
    target="claim" -> extract claim features, key=id
    target="sentences" -> extract sentence features, key=abstract_id (or id if FEVER)
    """
    data_map = {}
    with open(input_path) as f:
        lines = f.readlines()
        
    print(f"Processing {len(lines)} lines from {input_path} for target {target}...")
    
    for line in tqdm(lines):
        item = json.loads(line)
        claim_id = item['id']
        
        # FEVER doesn't have abstract_id, so we use claim_id as abstract_id (Reader sets abstract_id=-1 but maybe we should align)
        # Wait, if Reader sets abstract_id=-1, how does it look up corpus?
        # Answer: FEVERReader doesn't use Corpus! It reads sentences directly from jsonl.
        # So SciFactPICODataset needs to know how to fetch features.
        # If abstract_id is -1, it might fail to look up corpus_pico.pt.
        
        # Actually, for FEVER/PubMedQA, the "abstract_id" in the dataset might be just the claim_id 
        # or we need to mock it.
        # Let's assume for FEVER, we map claim_id -> sentences features in "corpus_pico.pt".
        
        if "abstract_id" in item:
            doc_id = item["abstract_id"]
        else:
            # For FEVER, map doc_id to claim_id (since 1 doc per claim in this preprocessed format)
            doc_id = claim_id
            
        if target == "claim":
            claim = item['claim']
            feat = extract_features_for_text(claim, model, tokenizer, device)
            data_map[claim_id] = feat
            
        elif target == "sentences":
            sentences = item['sentences']
            sent_features = []
            for sent in sentences:
                feat = extract_features_for_text(sent, model, tokenizer, device)
                sent_features.append(feat)
            data_map[doc_id] = sent_features
            
    return data_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/pico_features")

    parser.add_argument("--claims_path", type=str, default=None, help="Path to claims jsonl file")
    parser.add_argument("--corpus_path", type=str, default=None, help="Path to corpus jsonl file")

    parser.add_argument("--claims_output_name", type=str, default="claims_pico.pt")
    parser.add_argument("--corpus_output_name", type=str, default="corpus_pico.pt")
    parser.add_argument("--mode", type=str, default="scifact", choices=["scifact", "self_contained"],
                        help="Mode: scifact (separate corpus) or self_contained (FEVER/PubMedQA)")
    parser.add_argument("--self_contained_target", type=str, default="both", choices=["claim", "sentences", "both"],
                        help="For self_contained mode: extract 'claim' features, 'sentences' features, or 'both'")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_bioelectra(device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "scifact":
        # 1. Process Claims
        if args.claims_path:
            print(f"Processing Claims from {args.claims_path}...")
            claims_feats = process_file(args.claims_path, model, tokenizer, device, mode="scifact_claim")
            out_path = os.path.join(args.output_dir, args.claims_output_name)
            torch.save(claims_feats, out_path)
            print(f"Saved claims features to {out_path}")
        
        # 2. Process Corpus
        if args.corpus_path:
            print(f"Processing Corpus from {args.corpus_path}...")
            corpus_feats = process_file(args.corpus_path, model, tokenizer, device, mode="corpus")
            out_path = os.path.join(args.output_dir, args.corpus_output_name)
            torch.save(corpus_feats, out_path)
            print(f"Saved corpus features to {out_path}")
            
    elif args.mode == "self_contained":
        if not args.claims_path:
            raise ValueError("claims_path is required for self_contained mode")
            
        if args.self_contained_target in ["claim", "both"]:
            print(f"Processing Claims (Self-Contained) from {args.claims_path}...")
            claims_feats = process_self_contained(args.claims_path, model, tokenizer, device, target="claim")
            out_path = os.path.join(args.output_dir, args.claims_output_name)
            torch.save(claims_feats, out_path)
            print(f"Saved claims features to {out_path}")
            
        if args.self_contained_target in ["sentences", "both"]:
            print(f"Processing Sentences (Self-Contained) from {args.claims_path}...")
            corpus_feats = process_self_contained(args.claims_path, model, tokenizer, device, target="sentences")
            out_path = os.path.join(args.output_dir, args.corpus_output_name)
            torch.save(corpus_feats, out_path)
            print(f"Saved corpus features to {out_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()