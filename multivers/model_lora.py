"""
MultiVerS model with LoRA (Low-Rank Adaptation) for Longformer encoder.

This model bypasses the Longformer inplace operation error by:
1. Keeping the encoder frozen (no gradient flow through attention layers)
2. Adding LoRA adapters that operate on the OUTPUT of the encoder
3. Only training the LoRA adapters + classification heads

Key insight: We don't need gradients to flow THROUGH the encoder.
We extract hidden states with torch.no_grad(), then apply trainable LoRA adapters.

This allows the model to learn task-specific representations while avoiding
the Longformer gradient issues.
"""

import torch
from torch import nn
from torch.nn import functional as F
import math

from multivers.model import MultiVerSModel
from multivers.allennlp_nn_util import batched_index_select
from multivers.allennlp_feedforward import FeedForward
from .metrics import ClaimDocumentMetrics


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.
    
    Applies a low-rank update to input: output = x + scale * (up_proj(down_proj(x)))
    
    Args:
        in_features: Input dimension
        out_features: Output dimension (usually same as in_features)
        rank: Rank of the low-rank decomposition (smaller = fewer parameters)
        alpha: Scaling factor for LoRA output
        dropout: Dropout rate
    """
    def __init__(self, in_features, out_features=None, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank decomposition: W = B @ A where B is (out, rank), A is (rank, in)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize: down with normal, up with zeros (so initial output = 0)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
    def forward(self, x):
        # LoRA adjustment: scale * up(down(x))
        lora_out = self.lora_up(self.lora_down(self.dropout(x)))
        return x + self.scaling * lora_out


class MultiLayerLoRA(nn.Module):
    """
    Applies LoRA adapters to multiple layers of hidden states.
    
    This is applied AFTER the encoder outputs hidden states.
    """
    def __init__(self, hidden_size, num_layers=4, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LoRALayer(hidden_size, hidden_size, rank, alpha, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class MultiVerSLoRAModel(MultiVerSModel):
    """
    MultiVerS model with LoRA adapters for learning task-specific representations.
    
    The Longformer encoder is frozen, but LoRA adapters allow the model to
    learn adjustments to the hidden states without triggering gradient errors.
    """
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MultiVerSModel.add_model_specific_args(parent_parser)
        
        # LoRA hyperparameters
        parser.add_argument("--lora_rank", type=int, default=8,
                            help="Rank for LoRA decomposition (smaller = fewer params)")
        parser.add_argument("--lora_alpha", type=int, default=16,
                            help="Scaling factor for LoRA (usually 2*rank)")
        parser.add_argument("--lora_num_layers", type=int, default=4,
                            help="Number of LoRA adapter layers to stack")
        parser.add_argument("--lora_dropout", type=float, default=0.1,
                            help="Dropout for LoRA layers")
        
        # Loss weighting (same as PICO model for fair comparison)
        parser.add_argument("--rationale_pos_weight", type=float, default=1.0,
                            help="Positive weight for rationale loss (recall booster).")
        parser.add_argument("--label_pos_weight", type=float, default=1.0,
                            help="Positive weight for label loss.")

        return parser

    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.rationale_pos_weight = getattr(hparams, "rationale_pos_weight", 1.0)
        self.label_pos_weight = getattr(hparams, "label_pos_weight", 1.0)

        # === CRITICAL: Freeze encoder ===
        print("=" * 60)
        print("LoRA Model: Freezing Longformer encoder")
        print("Adding LoRA adapters for task-specific learning")
        print("=" * 60)
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        
        # === LoRA Adapters ===
        lora_rank = getattr(hparams, "lora_rank", 8)
        lora_alpha = getattr(hparams, "lora_alpha", 16)
        lora_num_layers = getattr(hparams, "lora_num_layers", 4)
        lora_dropout = getattr(hparams, "lora_dropout", 0.1)
        
        # LoRA for hidden states (applied to all token representations)
        self.lora_hidden = MultiLayerLoRA(
            hidden_size, lora_num_layers, lora_rank, lora_alpha, lora_dropout
        )
        
        # LoRA for pooled output (CLS representation)
        self.lora_pooled = LoRALayer(
            hidden_size, hidden_size, lora_rank, lora_alpha, lora_dropout
        )
        
        # Re-initialize classifiers (they'll be trained from scratch)
        activations = [nn.GELU(), nn.Identity()]
        dropouts = [self.dropout.p, 0]
        
        self.label_classifier = FeedForward(
            input_dim=hidden_size,
            num_layers=2,
            hidden_dims=[hidden_size, hparams.num_labels],
            activations=activations,
            dropout=dropouts
        )
        
        self.rationale_classifier = FeedForward(
            input_dim=2 * hidden_size,
            num_layers=2,
            hidden_dims=[hidden_size, 1],
            activations=activations,
            dropout=dropouts
        )
        
        # Print parameter counts
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for p in self.lora_hidden.parameters()) + \
                      sum(p.numel() for p in self.lora_pooled.parameters())
        
        print(f"✓ Encoder frozen: {encoder_params:,} parameters")
        print(f"✓ LoRA parameters: {lora_params:,}")
        print(f"✓ Total trainable: {trainable_params:,} parameters")
        print(f"✓ Trainable ratio: {trainable_params/encoder_params*100:.2f}%")

        # === Metrics (same as PICO model) ===
        fold_names = ["train", "valid"]
        claim_doc_metrics = {}
        for name in fold_names:
            claim_doc_metrics[f"claim_doc_metrics_{name}"] = ClaimDocumentMetrics(compute_on_step=False)
        self.claim_doc_metrics = nn.ModuleDict(claim_doc_metrics)

    def forward(self, tokenized, abstract_sent_idx, **kwargs):
        """
        Forward pass with LoRA adaptation.
        
        Key: Encoder runs in no_grad mode, LoRA adapters are trainable.
        """
        # Clone inputs to avoid any view issues
        tokenized = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in tokenized.items()
        }

        # === Encoder pass (NO gradients) ===
        with torch.no_grad():
            encoded = self.encoder(**tokenized)
            hidden_states = encoded.last_hidden_state.clone()
            pooled_output = encoded.pooler_output.clone() if encoded.pooler_output is not None else None

        # Detach to ensure no gradient flow to encoder
        hidden_states = hidden_states.detach()
        if pooled_output is not None:
            pooled_output = pooled_output.detach()

        # === Apply LoRA adapters (trainable) ===
        hidden_states = self.lora_hidden(hidden_states)
        if pooled_output is not None:
            pooled_output = self.lora_pooled(pooled_output)
        else:
            pooled_output = self.lora_pooled(hidden_states.mean(dim=1))

        # Apply dropout
        hidden_states = self.dropout(hidden_states).contiguous()
        pooled_output = self.dropout(pooled_output)

        # === Label Classification ===
        label_logits = self.label_classifier(pooled_output)
        label_probs = F.softmax(label_logits, dim=1).detach()
        
        if self.label_threshold is None:
            predicted_labels = label_logits.argmax(dim=1)
        else:
            label_probs_truncated = label_probs.clone()
            label_probs_truncated[:, self.nei_label] = self.label_threshold
            predicted_labels = label_probs_truncated.argmax(dim=1)

        # === Rationale Classification ===
        sentence_states = batched_index_select(hidden_states, abstract_sent_idx)
        pooled_rep = pooled_output.unsqueeze(1).expand_as(sentence_states)
        rationale_input = torch.cat([pooled_rep, sentence_states], dim=2)
        rationale_logits = self.rationale_classifier(rationale_input).squeeze(2)
        
        rationale_probs = torch.sigmoid(rationale_logits).detach()
        predicted_rationales = (rationale_probs >= self.rationale_threshold).to(torch.int64)

        return {
            "label_logits": label_logits,
            "rationale_logits": rationale_logits,
            "label_probs": label_probs,
            "rationale_probs": rationale_probs,
            "predicted_labels": predicted_labels,
            "predicted_rationales": predicted_rationales,
        }

    def training_step(self, batch, batch_idx):
        """Training step with same loss as PICO model."""
        res = self(batch["tokenized"], batch["abstract_sent_idx"])
        
        # Label loss
        label_weights = torch.tensor(
            [self.label_pos_weight, 1.0, self.label_pos_weight], 
            device=batch["label"].device
        )
        label_loss_per_sample = F.cross_entropy(
            res["label_logits"], batch["label"], weight=label_weights, reduction="none"
        )
        label_loss = (batch["weight"] * label_loss_per_sample).sum()
        
        # Rationale loss with pos_weight
        logits = res["rationale_logits"]
        targets = batch["rationale"].float()
        weights = batch["weight"]
        dataset_rationale_mask = batch["rationale_mask"].float()
        
        sentence_mask = (targets != -1).float()
        clean_targets = targets.clone()
        clean_targets[clean_targets == -1] = 0
        
        pos_weight = torch.tensor(self.rationale_pos_weight, device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, clean_targets, pos_weight=pos_weight, reduction="none"
        )
        
        final_mask = sentence_mask * dataset_rationale_mask.unsqueeze(1)
        weighted_loss = bce_loss * final_mask
        instance_loss = weighted_loss.sum(dim=1)
        rationale_loss = (instance_loss * weights).sum()
        
        # Total loss
        loss = self.label_weight * label_loss + self.rationale_weight * rationale_loss
        
        # Debug output
        if batch_idx % 50 == 0:
            batch_size = len(batch["label"])
            loss_mean = loss / batch_size
            print(f"\n[LoRA Batch {batch_idx}] Sum: {loss.item():.2f} (Mean: {loss_mean.item():.2f}) "
                  f"| Label: {label_loss.item():.2f} | Rationale: {rationale_loss.item():.2f}")
        
        self.log("train_label_loss", label_loss, on_step=False, on_epoch=True)
        self.log("train_rationale_loss", rationale_loss, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        self._invoke_metrics(res, batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        res = self(batch["tokenized"], batch["abstract_sent_idx"])
        
        # Same loss calculation as training
        label_weights = torch.tensor(
            [self.label_pos_weight, 1.0, self.label_pos_weight], 
            device=batch["label"].device
        )
        label_loss = F.cross_entropy(
            res["label_logits"], batch["label"], weight=label_weights, reduction="none"
        )
        label_loss = (batch["weight"] * label_loss).sum()
        
        logits = res["rationale_logits"]
        targets = batch["rationale"].float()
        weights = batch["weight"]
        dataset_rationale_mask = batch["rationale_mask"].float()
        
        sentence_mask = (targets != -1).float()
        clean_targets = targets.clone()
        clean_targets[clean_targets == -1] = 0
        
        pos_weight = torch.tensor(self.rationale_pos_weight, device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, clean_targets, pos_weight=pos_weight, reduction="none"
        )
        
        final_mask = sentence_mask * dataset_rationale_mask.unsqueeze(1)
        weighted_loss = bce_loss * final_mask
        instance_loss = weighted_loss.sum(dim=1)
        rationale_loss = (instance_loss * weights).sum()
        
        loss = self.label_weight * label_loss + self.rationale_weight * rationale_loss
        
        self.log("val_label_loss", label_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rationale_loss", rationale_loss, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self._invoke_metrics(res, batch, "valid")
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        self(batch["tokenized"], batch["abstract_sent_idx"])
    
    def _invoke_metrics(self, pred, batch, fold):
        if fold in ["train", "valid"]:
            detached = {k: v.detach() for k, v in pred.items()}
            self.claim_doc_metrics[f"claim_doc_metrics_{fold}"](detached, batch)
    
    def _log_metrics(self, fold):
        if fold in ["train", "valid"]:
            the_metric = self.claim_doc_metrics[f"claim_doc_metrics_{fold}"]
            to_log = the_metric.compute()
            the_metric.reset()
            for k, v in to_log.items():
                self.log(f"{fold}_{k}", v)

