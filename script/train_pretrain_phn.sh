#!/bin/bash

# ===========================================================================================
# MultiVerS-PICO Pre-training Script (Phase 1)
# Dataset: Evidence Inference (Large biomedical dataset for PICO reasoning)
# Goal: Pretrain the PICO attention mechanism on a large, relevant dataset before fine-tuning on SciFact.
# ===========================================================================================

set -e  # Exit on error

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate multivers

# Configuration
EXPERIMENT_NAME="pretrain_evidence_inference_$(date +%Y%m%d_%H%M%S)"
DATASET="evidence_inference"
STARTING_CHECKPOINT="checkpoints/scifact.ckpt"
PICO_FEATURE_DIR="data/pico_features/evidence_inference"

# WandB Configuration
WANDB_PROJECT="multivers-pico"
WANDB_ENTITY=""
WANDB_NAME="pretrain_phase1_evinf"
WANDB_TAGS=("pretrain" "evidence_inference" "pico")

# Hyperparameters
GPUS=1
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
# Effective Batch Size = 4 * 16 = 64 (Good for stable pretraining)
ACCUMULATE_GRAD_BATCHES=16
MAX_EPOCHS=5
LR=3e-5
PRECISION=16

# PICO Model Hyperparameters
PICO_NUM_TAGS=5
PICO_TOKEN_DIM=0
PICO_SPAN_FEATURE_DIM=768
PICO_SENTENCE_FEATURE_DIM=0
PICO_CLAIM_FEATURE_DIM=0
# Standard dropout for pretraining (data is large enough)
PICO_DROPOUT=0.1

# ===========================================================================================
# Pre-flight checks
# ===========================================================================================

echo "=========================================="
echo "MultiVerS PICO Pre-training (Phase 1)"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Dataset: $DATASET"
echo "PICO Features: $PICO_FEATURE_DIR"
echo ""

mkdir -p logs

# Check if PICO features exist
if [ ! -f "$PICO_FEATURE_DIR/corpus_pico.pt" ]; then
    echo "ERROR: corpus_pico.pt not found in $PICO_FEATURE_DIR"
    echo "Please run the preprocessing steps first!"
    exit 1
fi

if [ ! -f "$PICO_FEATURE_DIR/claims_train_pico.pt" ]; then
    echo "ERROR: claims_train_pico.pt not found in $PICO_FEATURE_DIR"
    exit 1
fi

echo "✓ Pre-flight checks passed"
echo ""

# ===========================================================================================
# Training
# ===========================================================================================

# Fix TQDM output clutter
export TQDM_MININTERVAL=5

python -m multivers-phn.train_pico \
    --datasets $DATASET \
    --starting_checkpoint $STARTING_CHECKPOINT \
    --experiment_name $EXPERIMENT_NAME \
    --result_dir results/pretrain_logs \
    --pico_feature_dir $PICO_FEATURE_DIR \
    \
    --wandb_project $WANDB_PROJECT \
    $(if [ -n "$WANDB_NAME" ]; then echo "--wandb_name $WANDB_NAME"; fi) \
    $(if [ ${#WANDB_TAGS[@]} -gt 0 ]; then echo "--wandb_tags ${WANDB_TAGS[@]}"; fi) \
    \
    --encoder_name longformer-large-science \
    --num_labels 3 \
    \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --num_workers 8 \
    \
    --lr $LR \
    --frac_warmup 0.1 \
    --max_epochs $MAX_EPOCHS \
    \
    --label_weight 1.0 \
    --rationale_weight 15.0 \
    \
    --pico_num_tags $PICO_NUM_TAGS \
    --pico_token_dim $PICO_TOKEN_DIM \
    --pico_span_feature_dim $PICO_SPAN_FEATURE_DIM \
    --pico_sentence_feature_dim $PICO_SENTENCE_FEATURE_DIM \
    --pico_claim_feature_dim $PICO_CLAIM_FEATURE_DIM \
    --pico_dropout $PICO_DROPOUT \
    \
    --gpus $GPUS \
    --precision $PRECISION \
    --gradient_clip_val 1.0 \
    \
    --monitor valid_f1 \
    --val_check_interval 0.25 \
    --log_every_n_steps 50 \
    \
    2>&1 | tee logs/train_pretrain_${EXPERIMENT_NAME}.log

echo ""
echo "=========================================="
echo "Pre-training completed!"
echo "=========================================="
echo "Next Step: Use the best checkpoint from results/pretrain_logs/$EXPERIMENT_NAME/checkpoint/"
echo "as the --starting_checkpoint for Phase 2 (SciFact Fine-tuning)."

