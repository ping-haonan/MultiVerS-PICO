"""
MultiVerS model with BOTH LoRA adapters AND PICO features.

This combines:
1. LoRA adapters - Allow the model to learn task-specific representations
2. PICO features - Domain-specific medical entity information

The encoder is frozen, but:
- LoRA adapters learn adjustments to hidden states
- PICO attention aggregates entity features
- Both contribute to the final classification
"""

import torch
from torch import nn
from torch.nn import functional as F
import math

from multivers.model import MultiVerSModel
from multivers.allennlp_nn_util import batched_index_select
from multivers.allennlp_feedforward import FeedForward
from .metrics import ClaimDocumentMetrics
from .model_lora import LoRALayer, MultiLayerLoRA


class MultiVerSPICOLoRAModel(MultiVerSModel):
    """
    MultiVerS with LoRA + PICO feature fusion.
    
    This is the most complete version:
    - LoRA adapters for task-specific learning (bypasses Longformer gradient issues)
    - PICO attention for domain-specific entity information
    """
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MultiVerSModel.add_model_specific_args(parent_parser)

        # === LoRA hyperparameters ===
        parser.add_argument("--lora_rank", type=int, default=8)
        parser.add_argument("--lora_alpha", type=int, default=16)
        parser.add_argument("--lora_num_layers", type=int, default=4)
        parser.add_argument("--lora_dropout", type=float, default=0.1)
        
        # === PICO hyperparameters ===
        parser.add_argument("--pico_num_tags", type=int, default=5)
        parser.add_argument("--pico_token_dim", type=int, default=0)
        parser.add_argument("--pico_span_feature_dim", type=int, default=768)
        parser.add_argument("--pico_sentence_feature_dim", type=int, default=0)
        parser.add_argument("--pico_claim_feature_dim", type=int, default=0)
        parser.add_argument("--pico_dropout", type=float, default=0.1)
        
        # === Loss weighting ===
        parser.add_argument("--rationale_pos_weight", type=float, default=1.0)
        parser.add_argument("--label_pos_weight", type=float, default=1.0)

        return parser

    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.rationale_pos_weight = getattr(hparams, "rationale_pos_weight", 1.0)
        self.label_pos_weight = getattr(hparams, "label_pos_weight", 1.0)

        # === Freeze encoder ===
        print("=" * 60)
        print("PICO + LoRA Model: Freezing Longformer encoder")
        print("Adding LoRA adapters + PICO feature fusion")
        print("=" * 60)
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        
        # === LoRA Adapters ===
        lora_rank = getattr(hparams, "lora_rank", 8)
        lora_alpha = getattr(hparams, "lora_alpha", 16)
        lora_num_layers = getattr(hparams, "lora_num_layers", 4)
        lora_dropout = getattr(hparams, "lora_dropout", 0.1)
        
        self.lora_hidden = MultiLayerLoRA(
            hidden_size, lora_num_layers, lora_rank, lora_alpha, lora_dropout
        )
        self.lora_pooled = LoRALayer(
            hidden_size, hidden_size, lora_rank, lora_alpha, lora_dropout
        )
        
        # === PICO Feature Layers ===
        pico_dropout = getattr(hparams, "pico_dropout", 0.1)
        
        self.use_span_features = getattr(hparams, "pico_span_feature_dim", 768) > 0
        if self.use_span_features:
            self.pico_span_proj = nn.Sequential(
                nn.Linear(hparams.pico_span_feature_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(pico_dropout)
            )
            self.pico_span_out_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(pico_dropout)
            )
        
        self.use_sentence_features = getattr(hparams, "pico_sentence_feature_dim", 0) > 0
        if self.use_sentence_features:
            self.pico_sentence_stat_proj = nn.Sequential(
                nn.Linear(hparams.pico_sentence_feature_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(pico_dropout)
            )
        
        self.use_claim_features = getattr(hparams, "pico_claim_feature_dim", 0) > 0
        if self.use_claim_features:
            self.pico_claim_stat_proj = nn.Sequential(
                nn.Linear(hparams.pico_claim_feature_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(pico_dropout)
            )
        
        # === Classifiers ===
        # Rationale input: [pooled, sentence, (pico_span), (pico_sentence)]
        rationale_input_dim = hidden_size * 2 # 1536
        if self.use_span_features:
            rationale_input_dim += hidden_size # 1536 + 768 = 2304
        if self.use_sentence_features:
            rationale_input_dim += hidden_size # 2304 + 768 = 3072

        self.rationale_classifier = FeedForward(
            input_dim=rationale_input_dim,
            num_layers=2,
            hidden_dims=[hidden_size, 1],
            activations=[nn.GELU(), nn.Identity()],
            dropout=[self.dropout.p, 0]
        )

        # Label input: [pooled, (claim_span), (claim_features)]
        label_input_dim = hidden_size
        if self.use_span_features:
            label_input_dim += hidden_size
        if self.use_claim_features:
            label_input_dim += hidden_size

        self.label_classifier = FeedForward(
            input_dim=label_input_dim,
            num_layers=2,
            hidden_dims=[hidden_size, hparams.num_labels],
            activations=[nn.GELU(), nn.Identity()],
            dropout=[self.dropout.p, 0]
        )

        # Zero buffer for padding
        self.register_buffer("_zero_vec", torch.zeros(1, 1, hidden_size), persistent=False)
        
        # === Metrics ===
        fold_names = ["train", "valid"]
        claim_doc_metrics = {}
        for name in fold_names:
            claim_doc_metrics[f"claim_doc_metrics_{name}"] = ClaimDocumentMetrics(compute_on_step=False)
        self.claim_doc_metrics = nn.ModuleDict(claim_doc_metrics)
        
        # Print stats
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ Encoder frozen: {encoder_params:,} parameters")
        print(f"✓ Total trainable: {trainable_params:,} parameters")

    def _attention_aggregate(self, query, span_features, span_mask):
        """PICO span attention aggregation (same as model_pico_attn.py)."""
        b, s, n, d_raw = span_features.shape # (batch_size, num_sentences, num_spans, feature_dim)
        hidden_size = query.size(-1) # 768 by default
        
        span_features = span_features.contiguous()
        flat_feats = span_features.reshape(b * s, n, d_raw) # (b * s, n, d_raw)
        keys = self.pico_span_proj(flat_feats)
        values = keys
        
        flat_query = query.reshape(b * s, 1, hidden_size) # (b, s, 768）-> (b * s, 1, 768)
        scores = torch.bmm(flat_query, keys.transpose(1, 2))
        scores = scores / math.sqrt(hidden_size)
        
        if span_mask is not None:
            mask = span_mask.contiguous().view(b * s, 1, n)
            scores = torch.where(mask > 0, scores, torch.tensor(-1e4, device=scores.device, dtype=scores.dtype))
        
        scores = torch.clamp(scores, min=-10.0, max=10.0)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, values) # weighted sum of sentence semantic features
        context = context.view(b, s, hidden_size)
        context = self.pico_span_out_proj(context)
        
        return context

    def forward(
        self,
        tokenized,
        abstract_sent_idx,
        pico_token_ids=None,
        pico_sentence_features=None,
        claim_pico_features=None,
        pico_span_features=None,
        pico_span_mask=None,
        claim_span_features=None,
        claim_span_mask=None,
        **kwargs
    ):
        tokenized = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in tokenized.items()
        }

        # === Encoder (NO gradients) ===
        with torch.no_grad():
            encoded = self.encoder(**tokenized)
            hidden_states = encoded.last_hidden_state.clone()
            pooled_output = encoded.pooler_output.clone() if encoded.pooler_output is not None else None

        hidden_states = hidden_states.detach()
        if pooled_output is not None:
            pooled_output = pooled_output.detach()

        # === Apply LoRA ===
        hidden_states = self.lora_hidden(hidden_states)
        if pooled_output is not None:
            pooled_output = self.lora_pooled(pooled_output)
        else:
            pooled_output = self.lora_pooled(hidden_states.mean(dim=1))

        hidden_states = self.dropout(hidden_states).contiguous()
        pooled_output = self.dropout(pooled_output)

        # === Rationale Features ===
        sentence_states = batched_index_select(hidden_states, abstract_sent_idx) # [B, S, 768]
        pooled_rep = pooled_output.unsqueeze(1).expand_as(sentence_states) # [B, S, 768]
        rationale_inputs = [pooled_rep, sentence_states]

        # PICO span features for sentences
        if self.use_span_features and pico_span_features is not None:
            n_sents_query = sentence_states.size(1)
            n_sents_pico = pico_span_features.size(1)
            
            if n_sents_query != n_sents_pico:
                if n_sents_pico > n_sents_query:
                    pico_span_features = pico_span_features[:, :n_sents_query, :, :]
                    if pico_span_mask is not None:
                        pico_span_mask = pico_span_mask[:, :n_sents_query, :]
                else:
                    diff = n_sents_query - n_sents_pico
                    b, _, n_spans, dim = pico_span_features.shape
                    padding_feats = torch.zeros(b, diff, n_spans, dim, device=pico_span_features.device, dtype=pico_span_features.dtype)
                    pico_span_features = torch.cat([pico_span_features, padding_feats], dim=1)
                    if pico_span_mask is not None:
                        padding_mask = torch.zeros(b, diff, n_spans, device=pico_span_mask.device, dtype=pico_span_mask.dtype)
                        pico_span_mask = torch.cat([pico_span_mask, padding_mask], dim=1)

            span_context = self._attention_aggregate(sentence_states, pico_span_features, pico_span_mask) # [B, S, 768]
            rationale_inputs.append(span_context)
        elif self.use_span_features:
            rationale_inputs.append(self._zero_vec.expand_as(sentence_states))

        # External manual features, not available now
        if self.use_sentence_features and pico_sentence_features is not None:
            stat_context = self.pico_sentence_stat_proj(pico_sentence_features)
            rationale_inputs.append(stat_context)
        elif self.use_sentence_features:
            rationale_inputs.append(self._zero_vec.expand_as(sentence_states))

        rationale_cat = torch.cat(rationale_inputs, dim=2) # [B, S, 2304]
        rationale_logits = self.rationale_classifier(rationale_cat).squeeze(2)
        rationale_probs = torch.sigmoid(rationale_logits).detach()
        predicted_rationales = (rationale_probs >= self.rationale_threshold).to(torch.int64)

        # === Label Features ===
        label_inputs = [pooled_output]

        if self.use_span_features and claim_span_features is not None:
            query_claim = pooled_output.unsqueeze(1)
            claim_span_ctx = self._attention_aggregate(query_claim, claim_span_features, claim_span_mask).squeeze(1)
            label_inputs.append(claim_span_ctx)
        elif self.use_span_features:
            label_inputs.append(self._zero_vec.squeeze(1).expand_as(pooled_output))

        # External manual features, not available now
        if self.use_claim_features and claim_pico_features is not None:
            claim_stat_ctx = self.pico_claim_stat_proj(claim_pico_features)
            label_inputs.append(claim_stat_ctx)
        elif self.use_claim_features:
            label_inputs.append(self._zero_vec.squeeze(1).expand_as(pooled_output))

        label_cat = torch.cat(label_inputs, dim=1)
        label_logits = self.label_classifier(label_cat)
        label_probs = F.softmax(label_logits, dim=1).detach()

        if self.label_threshold is None:
            predicted_labels = label_logits.argmax(dim=1)
        else:
            label_probs_truncated = label_probs.clone()
            label_probs_truncated[:, self.nei_label] = self.label_threshold
            predicted_labels = label_probs_truncated.argmax(dim=1)

        return {
            "label_logits": label_logits,
            "rationale_logits": rationale_logits,
            "label_probs": label_probs,
            "rationale_probs": rationale_probs,
            "predicted_labels": predicted_labels,
            "predicted_rationales": predicted_rationales,
        }

    def training_step(self, batch, batch_idx):
        pico_kwargs = {}
        for key in ["pico_span_features", "pico_span_mask", "claim_span_features", 
                    "claim_span_mask", "pico_sentence_features", "claim_pico_features", "pico_token_ids"]:
            if key in batch:
                pico_kwargs[key] = batch[key]
        
        res = self(batch["tokenized"], batch["abstract_sent_idx"], **pico_kwargs)
        
        # Label loss
        label_weights = torch.tensor(
            [self.label_pos_weight, 1.0, self.label_pos_weight], 
            device=batch["label"].device
        )
        label_loss = F.cross_entropy(res["label_logits"], batch["label"], weight=label_weights, reduction="none")
        label_loss = (batch["weight"] * label_loss).sum()
        
        # Rationale loss
        logits = res["rationale_logits"]
        targets = batch["rationale"].float()
        sentence_mask = (targets != -1).float()
        clean_targets = targets.clone()
        clean_targets[clean_targets == -1] = 0
        
        pos_weight = torch.tensor(self.rationale_pos_weight, device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, clean_targets, pos_weight=pos_weight, reduction="none")
        
        final_mask = sentence_mask * batch["rationale_mask"].float().unsqueeze(1)
        rationale_loss = (bce_loss * final_mask).sum(dim=1)
        rationale_loss = (rationale_loss * batch["weight"]).sum()
        
        loss = self.label_weight * label_loss + self.rationale_weight * rationale_loss
        
        if batch_idx % 50 == 0:
            print(f"\n[PICO+LoRA Batch {batch_idx}] Loss: {loss.item():.2f} | Label: {label_loss.item():.2f} | Rationale: {rationale_loss.item():.2f}")
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self._invoke_metrics(res, batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        pico_kwargs = {}
        for key in ["pico_span_features", "pico_span_mask", "claim_span_features", 
                    "claim_span_mask", "pico_sentence_features", "claim_pico_features", "pico_token_ids"]:
            if key in batch:
                pico_kwargs[key] = batch[key]
        
        res = self(batch["tokenized"], batch["abstract_sent_idx"], **pico_kwargs)
        
        label_weights = torch.tensor([self.label_pos_weight, 1.0, self.label_pos_weight], device=batch["label"].device)
        label_loss = F.cross_entropy(res["label_logits"], batch["label"], weight=label_weights, reduction="none")
        label_loss = (batch["weight"] * label_loss).sum()
        
        logits = res["rationale_logits"]
        targets = batch["rationale"].float()
        sentence_mask = (targets != -1).float()
        clean_targets = targets.clone()
        clean_targets[clean_targets == -1] = 0
        
        pos_weight = torch.tensor(self.rationale_pos_weight, device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, clean_targets, pos_weight=pos_weight, reduction="none")
        
        final_mask = sentence_mask * batch["rationale_mask"].float().unsqueeze(1)
        rationale_loss = (bce_loss * final_mask).sum(dim=1)
        rationale_loss = (rationale_loss * batch["weight"]).sum()
        
        loss = self.label_weight * label_loss + self.rationale_weight * rationale_loss
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._invoke_metrics(res, batch, "valid")
    
    def test_step(self, batch, batch_idx):
        pass
    
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

