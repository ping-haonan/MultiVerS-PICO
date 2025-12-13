# MultiVerS-PICO: Scientific Claim Verification with PICO Features and LoRA

This repository contains the enhanced MultiVerS model with PICO (Population, Intervention, Comparator, Outcome) feature integration and LoRA (Low-Rank Adaptation) for scientific claim verification.

## Project Structure

```
multivers-main/
├── multivers/                    # Enhanced model code
│   ├── model_pico_attn.py            # PICO attention model (frozen encoder)
│   ├── model_pico_lora.py            # PICO + LoRA model
│   ├── model_lora.py                 # LoRA layer definitions
│   ├── train_pico.py                 # PICO model training entry
│   ├── train_lora.py                 # LoRA model training entry
│   ├── data_train_pico.py            # PICO-enhanced data loader
│   ├── data_train.py                 # Base data loader
│   └── metrics.py                    # Evaluation metrics
├── script/
│   ├── get_data.sh                   # Download target datasets
│   ├── get_data_train.sh             # Download pre-training datasets
│   ├── preprocess_pico_features_phn.py   # PICO feature extraction
│   ├── prepare_evidence_inference.py     # Evidence Inference preprocessing
│   ├── train_pretrain_phn.sh         # Stage 1: Pre-training script
│   ├── ablation_scifact20_finetune.sh    # Stage 2: Fine-tuning (PICO vs LoRA)
│   ├── ablation_scifact20_posweight.sh   # Stage 2: Fine-tuning with class weight
│   ├── eval_ablation_scifact20.sh        # Evaluation script
│   └── eval_multivers.py             # Evaluation metrics computation
└── data/
    └── pico_features/                # Pre-computed PICO features
```

## Prerequisites

- Python 3.8+
- CUDA-enabled GPU (16GB+ VRAM recommended)
- Conda environment

### Required Models (Download from HuggingFace)

| Model | Path | Description |
|-------|------|-------------|
| BioELECTRA-PICO | `xlreator/BioELECTRA-PICO` | PICO entity extraction |
| Longformer | `allenai/longformer-base-4096` | Document encoder |

## Installation

```bash
# Create conda environment
conda create --name multivers python=3.8
conda activate multivers

# Install dependencies
pip install -r requirements.txt

# Setup module path
conda develop .
```

## Configuration

Before running, update the model paths in the following files:

**1. `script/preprocess_pico_features_phn.py`** (Line 24):
```python
BIOELECTRA_PATH = "/path/to/your/BioELECTRA-PICO"
```

**2. Training scripts** (e.g., `script/train_pretrain_phn.sh`):
```bash
--encoder_name /path/to/your/longformer-base-4096
```

## Quick Start

### Step 1: Download Data

```bash
# Download target datasets (SciFact, etc.)
bash script/get_data.sh

# Download pre-training datasets (Evidence Inference, etc.)
bash script/get_data_train.sh
```

### Step 2: Preprocess PICO Features

```bash
# Prepare Evidence Inference data format
python script/prepare_evidence_inference.py

# Extract PICO features for Evidence Inference
python script/preprocess_pico_features_phn.py \
    --output_dir data/pico_features/evidence_inference \
    --corpus_path data/evidence_inference_processed/corpus.jsonl \
    --corpus_output_name corpus_pico.pt

python script/preprocess_pico_features_phn.py \
    --output_dir data/pico_features/evidence_inference \
    --claims_path data/evidence_inference_processed/claims_train.jsonl \
    --claims_output_name claims_train_pico.pt

python script/preprocess_pico_features_phn.py \
    --output_dir data/pico_features/evidence_inference \
    --claims_path data/evidence_inference_processed/claims_dev.jsonl \
    --claims_output_name claims_dev_pico.pt

# Extract PICO features for SciFact-20
python script/preprocess_pico_features_phn.py \
    --output_dir data/pico_features/scifact_20 \
    --corpus_path data/scifact/corpus.jsonl \
    --corpus_output_name corpus_pico.pt

python script/preprocess_pico_features_phn.py \
    --output_dir data/pico_features/scifact_20 \
    --claims_path data_train/target/scifact_20/claims_train.jsonl \
    --claims_output_name claims_train_pico.pt

# Extract PICO features for SciFact-Open (evaluation)
python script/preprocess_pico_features_phn.py \
    --output_dir data/pico_features/scifact-open \
    --corpus_path data/scifact-open/corpus_candidates.jsonl \
    --corpus_output_name corpus_pico.pt

python script/preprocess_pico_features_phn.py \
    --output_dir data/pico_features/scifact-open \
    --claims_path data/scifact-open/claims_new.jsonl \
    --claims_output_name claims_test_pico.pt
```

### Step 3: Pre-train PICO Module (Stage 1)

```bash
bash script/train_pretrain_phn.sh
```

**Output:** `results/pretrain_logs/<experiment_name>/checkpoint/*.ckpt`

### Step 4: Fine-tune on SciFact-20 (Stage 2)

```bash
# Option A: PICO only vs PICO+LoRA ablation
bash script/ablation_scifact20_finetune.sh all

# Option B: With positive class weighting
bash script/ablation_scifact20_posweight.sh all
```

**Output:** `results/ablation_logs/scifact20_*/`

### Step 5: Evaluate on SciFact-Open

```bash
# Evaluate models without pos_weight
bash script/eval_ablation_scifact20.sh

# Evaluate models with pos_weight
bash script/eval_ablation_scifact20_posweight.sh
```

## Training Configuration

### Stage 1: Pre-training on Evidence Inference

| Parameter | Value |
|-----------|-------|
| Dataset | Evidence Inference |
| Epochs | 3 |
| Batch Size | 64 (4 × 16 accumulation) |
| Learning Rate | 3e-5 |
| rationale_weight | 15.0 |
| Encoder | Frozen |

### Stage 2: Fine-tuning on SciFact-20

| Parameter | PICO Only | PICO + LoRA |
|-----------|-----------|-------------|
| Dataset | SciFact-20 (1:20 neg sampling) |
| Epochs | 3 |
| Batch Size | 32 (4 × 8 accumulation) |
| Learning Rate | 2e-5 |
| LoRA Rank | - | 8 |
| LoRA Layers | - | 6 |
| Encoder | Frozen | Frozen + LoRA |

### With Positive Class Weighting

| Parameter | Value |
|-----------|-------|
| rationale_pos_weight | 10.0 |
| label_pos_weight | 2.0 |

## Model Variants

| Model | File | Description |
|-------|------|-------------|
| PICO Attention | `model_pico_attn.py` | Frozen encoder + PICO attention |
| PICO + LoRA | `model_pico_lora.py` | Frozen encoder + LoRA + PICO attention |
| LoRA Only | `model_lora.py` | Frozen encoder + LoRA (no PICO) |

## Evaluation Metrics

- **Abstract-level**: Precision, Recall, F1 (document-level label prediction)
- **Sentence-level**: Precision, Recall, F1 (evidence sentence selection)

## Results on SciFact-Open

| Model | Abstract F1 | Sentence F1 |
|-------|-------------|-------------|
| MultiVerS (original) | 52.38% | 42.19% |
| + PICO | 52.45% | 46.89% |
| + PICO + Class Weight | 53.11% | 48.85% |
| + PICO + LoRA | 53.24% | 49.75% |

## File Dependencies

```
multivers/
├── model_pico_lora.py      ← Main model (imports from below)
│   ├── model.py            ← Base MultiVerS
│   ├── model_lora.py       ← LoRA layers
│   └── metrics.py          ← Evaluation metrics
├── train_lora.py           ← Training entry (imports from below)
│   ├── train_pico.py       ← PICO training logic
│   └── data_train_pico.py  ← PICO data loader
└── data_train_pico.py      ← Data loader (imports from below)
    ├── data_train.py       ← Base data loader
    └── data_verisci.py     ← Corpus utilities
```
