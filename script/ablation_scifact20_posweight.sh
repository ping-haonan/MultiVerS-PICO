#!/bin/bash

# ===========================================================================================
# Ablation Study: Fine-tuning on SciFact-20 with pos_weight
# 
# 基于 ablation_scifact20_finetune.sh，添加 pos_weight 参数来处理类别不平衡
#
# 对比两种微调策略:
# 1. PICO only (冻结 encoder) + pos_weight
# 2. PICO + LoRA (冻结 encoder + LoRA adapters) + pos_weight
#
# 核心改进：
# - rationale_pos_weight = 10.0 (证据句子稀疏，约 1:10 比例)
# - label_pos_weight = 2.0 (1:20 负采样后，SUPPORT/REFUTE 仍较少)
#
# 基础模型: stage1_pico (在 Evidence Inference 上预训练的 PICO 模型)
# 训练集: scifact_20 (1:20 负采样)
# 验证集: scifact_open (使用正确的 corpus_candidates.jsonl)
# ===========================================================================================

set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate multivers

# Configuration
STARTING_CHECKPOINT="results/ablation_logs/stage1_pico/ablation_pico_pretrain_20251202_125323/checkpoint/epoch=2-valid_f1=0.7059-valid_precision=0.6511-valid_recall=0.7708.ckpt"
TRAIN_DATASET="scifact_20"
VAL_DATASET="scifact_open"  # SciFact-Open for validation (correct corpus)
MAX_EPOCHS=3

# PICO Feature Directories
PICO_FEATURE_DIR_TRAIN="data/pico_features/scifact_20"
PICO_FEATURE_DIR_VAL="data/pico_features/scifact-open"

# Create hybrid PICO feature directory for mixed train/val
HYBRID_PICO_DIR="data/pico_features/hybrid_scifact20_scifactopen"
mkdir -p $HYBRID_PICO_DIR

# ===========================================================================================
# Key Addition: pos_weight for class imbalance
# ===========================================================================================
RATIONALE_POS_WEIGHT=10.0  # 证据句子稀疏 (约 1:10)
LABEL_POS_WEIGHT=2.0       # SUPPORT/REFUTE vs NEI (1:20 负采样后仍不平衡)

# Hyperparameters
GPUS=1
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
ACCUMULATE_GRAD_BATCHES=8  # Effective batch size = 32
LR=2e-5
PRECISION=16

# LoRA Hyperparameters
LORA_RANK=8
LORA_ALPHA=16
LORA_NUM_LAYERS=6
LORA_DROPOUT=0.1

mkdir -p logs

echo "============================================================"
echo "Ablation: SciFact-20 Fine-tuning with pos_weight"
echo "============================================================"
echo "Starting Checkpoint: $STARTING_CHECKPOINT"
echo "Train Dataset: $TRAIN_DATASET"
echo "Val Dataset: $VAL_DATASET (correct corpus!)"
echo "Max Epochs: $MAX_EPOCHS"
echo ""
echo ">>> pos_weight Settings <<<"
echo "  rationale_pos_weight: $RATIONALE_POS_WEIGHT"
echo "  label_pos_weight: $LABEL_POS_WEIGHT"
echo ""

# ===========================================================================================
# Step 0: Prepare SciFact-Open PICO features with correct corpus
# ===========================================================================================
prepare_scifact_open_pico() {
    echo ""
    echo "============================================================"
    echo "Step 0: Preparing SciFact-Open PICO Features"
    echo "============================================================"
    
    SCIFACT_OPEN_CORPUS="data/scifact-open/corpus_candidates.jsonl"
    SCIFACT_OPEN_CLAIMS="data/scifact-open/claims_new.jsonl"
    
    # Check if we need to generate corpus PICO features for SciFact-Open corpus
    NEED_CORPUS_PICO=false
    
    if [ -L "$PICO_FEATURE_DIR_VAL/corpus_pico.pt" ]; then
        echo "⚠️  corpus_pico.pt is a symlink"
        LINK_TARGET=$(readlink -f "$PICO_FEATURE_DIR_VAL/corpus_pico.pt")
        if [[ "$LINK_TARGET" == *"scifact/corpus_pico"* ]]; then
            echo "❌ Points to SciFact corpus - missing 90% of SciFact-Open docs!"
            rm "$PICO_FEATURE_DIR_VAL/corpus_pico.pt"
            NEED_CORPUS_PICO=true
        fi
    elif [ ! -f "$PICO_FEATURE_DIR_VAL/corpus_pico.pt" ]; then
        NEED_CORPUS_PICO=true
    fi
    
    if [ "$NEED_CORPUS_PICO" = true ]; then
        echo "Generating PICO features for SciFact-Open corpus (12K docs)..."
        echo "This will take a few minutes..."
        python script/preprocess_pico_features_phn.py \
            --output_dir $PICO_FEATURE_DIR_VAL \
            --corpus_path $SCIFACT_OPEN_CORPUS \
            --corpus_output_name corpus_pico.pt
        echo "✓ Corpus PICO features generated"
    else
        echo "✓ Corpus PICO features exist"
    fi
    
    # Check claims features
    if [ ! -f "$PICO_FEATURE_DIR_VAL/claims_test_pico.pt" ]; then
        echo "Generating claims PICO features..."
        python script/preprocess_pico_features_phn.py \
            --output_dir $PICO_FEATURE_DIR_VAL \
            --claims_path $SCIFACT_OPEN_CLAIMS \
            --claims_output_name claims_test_pico.pt
    else
        echo "✓ Claims PICO features exist"
    fi
    
    # Create hybrid directory with proper links
    echo "Setting up hybrid PICO feature directory..."
    
    # Training features from SciFact-20
    ln -sf $(pwd)/$PICO_FEATURE_DIR_TRAIN/claims_train_pico.pt $HYBRID_PICO_DIR/claims_train_pico.pt
    ln -sf $(pwd)/$PICO_FEATURE_DIR_TRAIN/corpus_pico.pt $HYBRID_PICO_DIR/corpus_pico.pt
    
    # Validation features from SciFact-Open (use test as dev)
    ln -sf $(pwd)/$PICO_FEATURE_DIR_VAL/claims_test_pico.pt $HYBRID_PICO_DIR/claims_dev_pico.pt
    
    echo "✓ Hybrid PICO feature directory ready"
}

# ===========================================================================================
# Strategy 1: PICO Only (Frozen Encoder) + pos_weight
# ===========================================================================================
finetune_pico_only() {
    echo ""
    echo "============================================================"
    echo "Strategy 1: PICO Only + pos_weight"
    echo "  rationale_pos_weight: $RATIONALE_POS_WEIGHT"
    echo "  label_pos_weight: $LABEL_POS_WEIGHT"
    echo "============================================================"
    
    EXPERIMENT_NAME="ablation_scifact20_pico_posweight_$(date +%Y%m%d_%H%M%S)"
    
    python -m multivers-phn.train_pico \
        --experiment_name $EXPERIMENT_NAME \
        --result_dir results/ablation_logs/scifact20_pico_posweight \
        --datasets "$TRAIN_DATASET,$VAL_DATASET" \
        --pico_feature_dir $HYBRID_PICO_DIR \
        --starting_checkpoint $STARTING_CHECKPOINT \
        --gpus $GPUS \
        --precision $PRECISION \
        --gradient_clip_val 1.0 \
        --max_epochs $MAX_EPOCHS \
        --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
        --val_check_interval 0.5 \
        --log_every_n_steps 10 \
        --lr $LR \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --num_workers 4 \
        --pico_num_tags 5 \
        --pico_token_dim 0 \
        --pico_span_feature_dim 768 \
        --pico_sentence_feature_dim 0 \
        --pico_claim_feature_dim 0 \
        --pico_dropout 0.2 \
        --rationale_pos_weight $RATIONALE_POS_WEIGHT \
        --label_pos_weight $LABEL_POS_WEIGHT \
        --encoder_name /root/autodl-tmp/models/longformer-base-4096 \
        --monitor valid_f1 \
        2>&1 | tee logs/ablation_scifact20_pico_posweight_${EXPERIMENT_NAME}.log
    
    echo "✓ PICO Only + pos_weight fine-tuning completed"
    
    # Run final evaluation on SciFact-Open
    echo ""
    echo "Running final SciFact-Open evaluation..."
    BEST_CKPT=$(ls -t results/ablation_logs/scifact20_pico_posweight/${EXPERIMENT_NAME}/checkpoint/*.ckpt | head -1)
    if [ -n "$BEST_CKPT" ]; then
        evaluate_on_scifact_open "$BEST_CKPT" "pico_posweight"
    fi
}

# ===========================================================================================
# Strategy 2: PICO + LoRA + pos_weight
# ===========================================================================================
finetune_pico_lora() {
    echo ""
    echo "============================================================"
    echo "Strategy 2: PICO + LoRA + pos_weight"
    echo "  rationale_pos_weight: $RATIONALE_POS_WEIGHT"
    echo "  label_pos_weight: $LABEL_POS_WEIGHT"
    echo "============================================================"
    
    EXPERIMENT_NAME="ablation_scifact20_pico_lora_posweight_$(date +%Y%m%d_%H%M%S)"
    
    python -m multivers-phn.train_lora \
        --experiment_name $EXPERIMENT_NAME \
        --result_dir results/ablation_logs/scifact20_pico_lora_posweight \
        --model_type pico_lora \
        --datasets "$TRAIN_DATASET,$VAL_DATASET" \
        --pico_feature_dir $HYBRID_PICO_DIR \
        --starting_checkpoint $STARTING_CHECKPOINT \
        --gpus $GPUS \
        --precision $PRECISION \
        --gradient_clip_val 1.0 \
        --max_epochs $MAX_EPOCHS \
        --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
        --val_check_interval 0.5 \
        --log_every_n_steps 10 \
        --lr $LR \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --num_workers 4 \
        --pico_num_tags 5 \
        --pico_token_dim 0 \
        --pico_span_feature_dim 768 \
        --pico_sentence_feature_dim 0 \
        --pico_claim_feature_dim 0 \
        --pico_dropout 0.2 \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_num_layers $LORA_NUM_LAYERS \
        --lora_dropout $LORA_DROPOUT \
        --rationale_pos_weight $RATIONALE_POS_WEIGHT \
        --label_pos_weight $LABEL_POS_WEIGHT \
        --encoder_name /root/autodl-tmp/models/longformer-base-4096 \
        --monitor valid_f1 \
        2>&1 | tee logs/ablation_scifact20_pico_lora_posweight_${EXPERIMENT_NAME}.log
    
    echo "✓ PICO + LoRA + pos_weight fine-tuning completed"
    
    # Run final evaluation on SciFact-Open
    echo ""
    echo "Running final SciFact-Open evaluation..."
    BEST_CKPT=$(ls -t results/ablation_logs/scifact20_pico_lora_posweight/${EXPERIMENT_NAME}/checkpoint/*.ckpt | head -1)
    if [ -n "$BEST_CKPT" ]; then
        evaluate_on_scifact_open "$BEST_CKPT" "pico_lora_posweight"
    fi
}

# ===========================================================================================
# Final Evaluation on SciFact-Open (same as eval_multivers.py)
# ===========================================================================================
evaluate_on_scifact_open() {
    CHECKPOINT=$1
    MODEL_TYPE=$2  # "pico_posweight" or "pico_lora_posweight"
    
    echo ""
    echo "============================================================"
    echo "Final Evaluation on SciFact-Open: $MODEL_TYPE"
    echo "Checkpoint: $CHECKPOINT"
    echo "============================================================"
    
    INPUT_FILE="data/scifact-open/claims_new.jsonl"
    CORPUS_FILE="data/scifact-open/corpus_candidates.jsonl"
    OUTPUT_DIR="results/predictions/final_eval_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p $OUTPUT_DIR
    
    python3 <<EOF
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from multivers.data_train import get_tokenizer
from multivers.util import load_jsonl
from multivers.data_verisci import Corpus

# Load model based on type
model_type = "$MODEL_TYPE"
checkpoint_path = "$CHECKPOINT"

if "lora" in model_type:
    from multivers_phn.model_pico_lora import MultiVerSPICOLoRAModel
    model = MultiVerSPICOLoRAModel.load_from_checkpoint(checkpoint_path, strict=False)
else:
    from multivers_phn.model_pico_attn import MultiVerSPICOAttnModel
    model = MultiVerSPICOAttnModel.load_from_checkpoint(checkpoint_path, strict=False)

model.eval()
model.cuda()
print("✓ Model loaded")

# Setup
tokenizer = get_tokenizer(type('Args', (), {'encoder_name': '/root/autodl-tmp/models/longformer-base-4096'}))
corpus = Corpus.from_jsonl("$CORPUS_FILE")
doc_map = {doc.id: doc for doc in corpus.documents}
raw_claims = load_jsonl("$INPUT_FILE")
print(f"✓ Loaded {len(raw_claims)} claims, {len(doc_map)} documents")

# Prepare data
from multivers_phn.data_train_pico import SciFactPICOCollator, SciFactPICODataset
entries = []
for raw in tqdm(raw_claims, desc="Preparing"):
    for doc_id in raw.get("doc_ids", []):
        try: doc_id_int = int(doc_id)
        except: doc_id_int = doc_id
        doc = doc_map.get(doc_id_int) or doc_map.get(str(doc_id))
        if doc is None: continue
        entries.append({
            "claim_id": raw["id"], "claim": raw["claim"], "abstract_id": doc.id,
            "abstract": doc.sentences, "paragraph": " ".join(doc.sentences),
            "negative_sample_id": 0, "weight": 1.0,
            "to_tensorize": {"claim": raw["claim"], "sentences": doc.sentences,
                           "label": "NOT ENOUGH INFO", "rationales": [], "title": doc.title}
        })

print(f"✓ {len(entries)} claim-document pairs")
dataset = SciFactPICODataset(entries, tokenizer, "scifact-open", 1.0,
    pico_claims_cache="$PICO_FEATURE_DIR_VAL/claims_test_pico.pt",
    pico_corpus_cache="$PICO_FEATURE_DIR_VAL/corpus_pico.pt")
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4,
    collate_fn=SciFactPICOCollator(tokenizer), pin_memory=True)

# Inference
predictions_map = {}
print("Running inference...")
with torch.no_grad():
    for batch in tqdm(loader):
        device = next(model.parameters()).device
        def move(obj):
            if isinstance(obj, torch.Tensor): return obj.to(device, non_blocking=True)
            elif isinstance(obj, dict): return {k: move(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [move(v) for v in obj]
            elif hasattr(obj, 'to'): return obj.to(device, non_blocking=True)
            return obj
        batch = move(batch)
        
        pico_kwargs = {k: batch[k] for k in ["pico_span_features", "pico_span_mask",
            "claim_span_features", "claim_span_mask", "pico_sentence_features",
            "claim_pico_features", "pico_token_ids"] if k in batch}
        output = model(batch["tokenized"], batch["abstract_sent_idx"], **pico_kwargs)
        
        label_preds = output["label_probs"].argmax(dim=1)
        rationale_probs = torch.sigmoid(output["rationale_logits"])
        
        for i in range(len(batch["claim_id"])):
            c_id, a_id = batch["claim_id"][i].item(), batch["abstract_id"][i].item()
            lbl = {0: "CONTRADICT", 1: "NEI", 2: "SUPPORT"}[label_preds[i].item()]
            if lbl == "NEI": continue
            
            valid_mask = batch["abstract_sent_idx"][i] > 0
            valid_idx = valid_mask.nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0: pred_sent = []
            else:
                k = min(3, len(valid_idx))
                topk_vals, topk_idx = torch.topk(rationale_probs[i][valid_idx], k)
                pred_sent = valid_idx[topk_idx[topk_vals > 0.005]].cpu().tolist()
            
            if c_id not in predictions_map: predictions_map[c_id] = {}
            predictions_map[c_id][str(a_id)] = {"label": lbl, "sentences": pred_sent}

with open("$OUTPUT_DIR/predictions.jsonl", 'w') as f:
    for c_id, ev in predictions_map.items():
        f.write(json.dumps({"id": c_id, "evidence": ev}) + "\n")
print("✓ Predictions saved")
EOF

    echo ""
    echo "Final Evaluation Results ($MODEL_TYPE):"
    python script/eval_multivers.py \
        --gold_file $INPUT_FILE \
        --pred_file $OUTPUT_DIR/predictions.jsonl \
        --input_file $INPUT_FILE
}

# ===========================================================================================
# Main Execution
# ===========================================================================================

# Always prepare PICO features first
prepare_scifact_open_pico

case "${1:-all}" in
    pico)
        finetune_pico_only
        ;;
    lora)
        finetune_pico_lora
        ;;
    all)
        finetune_pico_only
        finetune_pico_lora
        ;;
    eval)
        # Just run evaluation on a given checkpoint
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 eval <checkpoint_path> <model_type>"
            exit 1
        fi
        evaluate_on_scifact_open "$2" "$3"
        ;;
    *)
        echo "Usage: $0 {pico|lora|all|eval}"
        echo "  pico - Run PICO only + pos_weight fine-tuning"
        echo "  lora - Run PICO + LoRA + pos_weight fine-tuning"
        echo "  all  - Run both (default)"
        echo "  eval <ckpt> <type> - Only run evaluation"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "All tasks completed!"
echo "============================================================"

echo ""
echo "Results saved to:"
echo "  - results/ablation_logs/scifact20_pico_posweight/"
echo "  - results/ablation_logs/scifact20_pico_lora_posweight/"
echo ""
echo "Key hyperparameters used:"
echo "  - rationale_pos_weight: $RATIONALE_POS_WEIGHT"
echo "  - label_pos_weight: $LABEL_POS_WEIGHT"

