#!/bin/bash

# ===========================================================================================
# Evaluation Script for SciFact-20 Ablation Models on SciFact-Open
# ===========================================================================================
# This script evaluates both PICO-only and PICO+LoRA models trained on SciFact-20
# on the SciFact-Open test set, providing both document-level and sentence-level metrics.
# ===========================================================================================

set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate multivers

cd /root/autodl-tmp/code/multivers-main

# ===========================================================================================
# Configuration
# ===========================================================================================

# Model checkpoints (best F1 from training)
PICO_ONLY_CHECKPOINT="results/ablation_logs/ablation_scifact20_pico_20251204_102006/checkpoint/epoch=0-valid_f1=0.5434-valid_precision=0.7402-valid_recall=0.4292.ckpt"
PICO_LORA_CHECKPOINT="results/ablation_logs/ablation_scifact20_pico_lora_20251204_123255/checkpoint/epoch=1-valid_f1=0.5491-valid_precision=0.7283-valid_recall=0.4406.ckpt"

# Data paths - Use SciFact-Open corpus (NOT SciFact corpus!)
INPUT_FILE="data/scifact-open/claims_new.jsonl"
CORPUS_FILE="data/scifact-open/corpus_candidates.jsonl"
# IMPORTANT: Use the same PICO features as training!
PICO_FEATURE_DIR="data/pico_features/hybrid_scifact20_open"

# Encoder
ENCODER_NAME="/root/autodl-tmp/models/longformer-base-4096"

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="results/ablation_eval_scifact20_$TIMESTAMP"
mkdir -p $OUTPUT_BASE

# Create symlink for module import
if [ ! -d "multivers_phn" ]; then
    ln -sf multivers-phn multivers_phn
fi

# ===========================================================================================
# Step 0: Check PICO Features
# ===========================================================================================

echo "=========================================="
echo "Step 0: Checking PICO Features"
echo "=========================================="

# Claims PICO features - Use claims_dev_pico.pt from hybrid directory (same as training validation)
CLAIMS_PICO="$PICO_FEATURE_DIR/claims_dev_pico.pt"
if [ ! -f "$CLAIMS_PICO" ]; then
    echo "ERROR: Claims PICO features not found: $CLAIMS_PICO"
    exit 1
else
    echo "✓ Claims PICO features found: $CLAIMS_PICO"
fi

# Corpus PICO features - Use the same file as training (corpus_pico.pt in hybrid directory)
CORPUS_PICO="$PICO_FEATURE_DIR/corpus_pico.pt"
if [ ! -f "$CORPUS_PICO" ]; then
    echo "ERROR: Corpus PICO features not found: $CORPUS_PICO"
    exit 1
else
    echo "✓ Corpus PICO features found: $CORPUS_PICO"
fi

# ===========================================================================================
# Function: Run Evaluation for a Model
# ===========================================================================================

run_evaluation() {
    local MODEL_TYPE=$1      # "pico" or "pico_lora"
    local CHECKPOINT=$2
    local OUTPUT_DIR=$3
    
    mkdir -p $OUTPUT_DIR
    
    echo ""
    echo "=========================================="
    echo "Evaluating: $MODEL_TYPE"
    echo "Checkpoint: $CHECKPOINT"
    echo "Output: $OUTPUT_DIR"
    echo "=========================================="
    
    # Check checkpoint exists
    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Checkpoint not found: $CHECKPOINT"
        return 1
    fi

    # Run prediction
    python3 <<EOF
import torch
import json
import os
from argparse import Namespace
from tqdm import tqdm

# Import based on model type
MODEL_TYPE = "$MODEL_TYPE"
if MODEL_TYPE == "pico_lora":
    from multivers_phn.model_pico_lora import MultiVerSPICOLoRAModel as Model
else:
    from multivers_phn.model_pico import MultiVerSPICOModel as Model

from multivers_phn.data_train_pico import SciFactPICODataset, SciFactPICOCollator
from multivers.data_train import get_tokenizer
from multivers.util import load_jsonl
from multivers.data_verisci import Corpus
from torch.utils.data import DataLoader

# Configuration
checkpoint_path = "$CHECKPOINT"
input_file = "$INPUT_FILE"
corpus_file = "$CORPUS_FILE"
output_file = "$OUTPUT_DIR/predictions.jsonl"
pico_claims_path = "$CLAIMS_PICO"
pico_corpus_path = "$CORPUS_PICO"
encoder_name = "$ENCODER_NAME"

print(f"Loading model from {checkpoint_path}")

# Load checkpoint to extract hparams
ckpt = torch.load(checkpoint_path, map_location='cpu')
saved_hparams = ckpt.get('hyper_parameters', {})

# Handle nested hparams structure (PyTorch Lightning saves as hparams.hparams)
if 'hparams' in saved_hparams and hasattr(saved_hparams['hparams'], '__dict__'):
    hparams = saved_hparams['hparams']
    print(f"Using nested hparams from checkpoint")
else:
    hparams = Namespace(**saved_hparams)
    print(f"Using flat hparams from checkpoint")

# Only add missing parameters that have safe defaults
default_pico_params = {
    'pico_span_projection_dim': 1024,  # Match training config
    'pico_span_feature_dim': 768,
    'pico_dropout': 0.2,
    'pico_token_dim': 0,
    'pico_sentence_feature_dim': 0,
    'pico_claim_feature_dim': 0,
    'pico_num_tags': 5,
}
for k, v in default_pico_params.items():
    if not hasattr(hparams, k):
        setattr(hparams, k, v)
        print(f"  Added missing param: {k}={v}")

# Load model with hparams
model = Model.load_from_checkpoint(checkpoint_path, hparams=hparams, strict=False)
model.eval()
model.cuda()
print("✓ Model loaded")

# Load tokenizer
tokenizer = get_tokenizer(type('Args', (), {'encoder_name': encoder_name}))

# Load corpus
print("Loading corpus...")
corpus = Corpus.from_jsonl(corpus_file)
doc_map = {doc.id: doc for doc in corpus.documents}
print(f"✓ Loaded {len(doc_map)} documents")

# Load claims
print("Loading claims...")
raw_claims = load_jsonl(input_file)
print(f"✓ Loaded {len(raw_claims)} claims")

# Create dataset entries
print("Creating dataset entries...")
entries = []
for raw in tqdm(raw_claims, desc="Preparing data"):
    doc_ids = raw.get("doc_ids", [])
    claim_id = raw["id"]
    claim_text = raw["claim"]
    
    for doc_id in doc_ids:
        try:
            doc_id_int = int(doc_id)
        except:
            doc_id_int = doc_id
            
        doc = doc_map.get(doc_id_int) or doc_map.get(str(doc_id))
        
        if doc is None:
            continue
            
        entry = {
            "claim_id": claim_id,
            "abstract_id": doc.id,
            "negative_sample_id": 0,
            "weight": 1.0,
            "to_tensorize": {
                "claim": claim_text,
                "sentences": doc.sentences,
                "label": "NOT ENOUGH INFO",
                "rationales": [],
                "title": doc.title
            }
        }
        entries.append(entry)

print(f"✓ Created {len(entries)} claim-document pairs")

# Create dataset with PICO features
print("Creating dataset with PICO features...")
dataset = SciFactPICODataset(
    entries,
    tokenizer,
    "scifact-open",
    1.0,
    pico_claims_cache=pico_claims_path,
    pico_corpus_cache=pico_corpus_path
)

loader = DataLoader(
    dataset,
    batch_size=4,  # Same as training eval_batch_size
    shuffle=False,
    num_workers=4,
    collate_fn=SciFactPICOCollator(tokenizer),
    pin_memory=True
)

# Inference
predictions_map = {}

print("Running Inference...")
with torch.no_grad():
    for batch in tqdm(loader, desc="Predicting"):
        device = next(model.parameters()).device
        
        def move_to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(device, non_blocking=True)
            elif isinstance(obj, dict):
                return {k: move_to_device(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_to_device(v) for v in obj]
            elif hasattr(obj, 'to'):
                return obj.to(device)
            return obj
            
        batch = move_to_device(batch)
        
        # PICO kwargs
        pico_kwargs = {}
        for key in ["pico_span_features", "pico_span_mask", "claim_span_features", 
                    "claim_span_mask", "pico_sentence_features", "claim_pico_features", "pico_token_ids"]:
            if key in batch:
                pico_kwargs[key] = batch[key]
        
        # Forward
        output = model(batch["tokenized"], batch["abstract_sent_idx"], **pico_kwargs)
        
        # Process outputs
        label_probs = output["label_probs"]
        label_preds = label_probs.argmax(dim=1)
        rationale_probs = torch.sigmoid(output["rationale_logits"])
        
        claim_ids = batch["claim_id"]
        abstract_ids = batch["abstract_id"]
        
        for i in range(len(claim_ids)):
            c_id = claim_ids[i].item()
            a_id = abstract_ids[i].item()
            lbl_idx = label_preds[i].item()
            
            label_map = {0: "CONTRADICT", 1: "NEI", 2: "SUPPORT"}
            pred_label = label_map[lbl_idx]
            
            if pred_label == "NEI":
                continue
                
            rat_probs = rationale_probs[i]
            valid_mask = batch["abstract_sent_idx"][i] > 0
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            
            if len(valid_indices) == 0:
                pred_sentences = []
            else:
                valid_probs = rat_probs[valid_indices]
                k = min(3, len(valid_indices))
                topk_vals, topk_idx_in_valid = torch.topk(valid_probs, k)
                mask = topk_vals > 0.005 
                final_idx_in_valid = topk_idx_in_valid[mask]
                pred_sentences = valid_indices[final_idx_in_valid].cpu().tolist()
            
            if c_id not in predictions_map:
                predictions_map[c_id] = {}
            
            predictions_map[c_id][str(a_id)] = {
                "label": pred_label,
                "sentences": pred_sentences
            }

# Write output
print(f"Writing predictions to {output_file}...")
with open(output_file, 'w') as f:
    for c_id, evidence in predictions_map.items():
        out_obj = {"id": c_id, "evidence": evidence}
        f.write(json.dumps(out_obj) + "\n")

print(f"✓ Saved predictions for {len(predictions_map)} claims")
EOF

    # Run evaluation
    echo ""
    echo "Running evaluation..."
    python script/eval_multivers.py \
        --gold_file $INPUT_FILE \
        --pred_file $OUTPUT_DIR/predictions.jsonl \
        --input_file $INPUT_FILE | tee $OUTPUT_DIR/eval_results.txt
    
    echo "✓ Evaluation complete for $MODEL_TYPE"
}

# ===========================================================================================
# Run Evaluations
# ===========================================================================================

echo ""
echo "=========================================="
echo "SciFact-20 Ablation Evaluation on SciFact-Open"
echo "=========================================="

# 1. PICO-only model
run_evaluation "pico" "$PICO_ONLY_CHECKPOINT" "$OUTPUT_BASE/pico_only"

# 2. PICO+LoRA model
run_evaluation "pico_lora" "$PICO_LORA_CHECKPOINT" "$OUTPUT_BASE/pico_lora"

# ===========================================================================================
# Summary
# ===========================================================================================

echo ""
echo "=========================================="
echo "EVALUATION SUMMARY"
echo "=========================================="
echo ""
echo "=== PICO-only Model ==="
echo "Checkpoint: $PICO_ONLY_CHECKPOINT"
cat $OUTPUT_BASE/pico_only/eval_results.txt 2>/dev/null || echo "Results not available"

echo ""
echo "=== PICO+LoRA Model ==="
echo "Checkpoint: $PICO_LORA_CHECKPOINT"
cat $OUTPUT_BASE/pico_lora/eval_results.txt 2>/dev/null || echo "Results not available"

echo ""
echo "=========================================="
echo "Results saved to: $OUTPUT_BASE"
echo "=========================================="

